import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class SEBlock(nn.Module):
    # SEBlock：注意力机制关注瓶子特征
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class SpatialSoftmax(nn.Module):
    # SpatialSoftmax：将卷积特征转换为空间坐标，关注瓶子位置
    def __init__(self, height, width, channel, temperature=0.5):
        super(SpatialSoftmax, self).__init__()
        self.height, self.width, self.channel = height, width, channel
        self.temperature = temperature

        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1, 1, self.height),
            torch.linspace(-1, 1, self.width),
            indexing='ij'
        )
        self.register_buffer('pos_x', pos_x.reshape(-1))
        self.register_buffer('pos_y', pos_y.reshape(-1))

    def forward(self, x):
        b, c, h, w = x.size()
        logits = x.view(b, c, -1) / self.temperature
        probs = F.softmax(logits, dim=-1)
        expected_x = torch.sum(probs * self.pos_x, dim=-1)  # [B, 512, 49] -> [B, 512]
        expected_y = torch.sum(probs * self.pos_y, dim=-1)
        return torch.cat([expected_x, expected_y], dim=-1)
    

class BCNet_Transformer(nn.Module):
    def __init__(self, mode="resnet", joint_dim=25, action_dim=8, nhead=8, num_layers=2, window_size=4):
        super(BCNet_Transformer, self).__init__()
        self.mode = mode
        # 由于 DINOv2 预训练模型只接受 3 通道，我们添加一个初始卷积层将 4 通道转为 3 通道
        # 或者对于 ResNet 模式，直接修改第一层卷积
        self.input_adapter = nn.Conv2d(4, 3, kernel_size=1)

        # 使用预训练的dinov2代替resnet18，提取更丰富的视觉特征
        if mode == "vits14":
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            for param in self.backbone.parameters(): param.requires_grad = False
            for param in self.backbone.blocks[-2:].parameters(): param.requires_grad = True
            curr_channels, feat_size = 384, 16 
        else:
            res18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(res18.children())[:7])  # [B, 256, 14, 14]
            curr_channels = 256
            feat_size = 14        

        self.d_model = 512
        input_dim = curr_channels * 2 + curr_channels + 128 # 视觉坐标 + 语义特征 + 关节状态特征
        self.deal_transformer_input = nn.Linear(input_dim, self.d_model)

        self.shared_se = SEBlock(curr_channels)
        # 位置信息提取
        self.pos_extractor = SpatialSoftmax(feat_size, feat_size, curr_channels)
        # 语义信息提取
        self.obj_extractor = nn.AdaptiveAvgPool2d((1, 1))

        # 位置编码
        self.pos_emb = nn.Parameter(torch.zeros(1, window_size, self.d_model))
        self.state_encoder = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.PReLU()
        )
        
        # Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=nhead, 
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fusion = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.PReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, img_seq, state_seq, depth_seq):
        # img_seq: [B, T, 3, 224, 224]
        # state_seq: [B, T, 24]
        B, T, _, H, W = img_seq.size()
        img_in = torch.cat([img_seq, depth_seq], dim=2) 
        img_in = img_in.view(B * T, 4, H, W)    # 包含rgbd
        state_seq = state_seq.view(B * T, -1)   # (B*T, 24)

        img_in = self.input_adapter(img_in)  # (B*T, 3, H, W)
        if self.mode == "vits14":
            with torch.no_grad(): # 冻结 backbone
                feat_dict = self.backbone.forward_features(img_in)
                x = feat_dict["x_norm_patchtokens"] # (B*T, 256, 384)
                x = x.reshape(B * T, 16, 16, 384).permute(0, 3, 1, 2).contiguous() # (B*T, 384, 16, 16)
        else:
            x = self.backbone(img_in)

        x = self.shared_se(x)
        pos_feat = self.pos_extractor(x)  # (B*T, curr_channels*2)
        obj_feat = self.obj_extractor(x).view(x.size(0), -1)    # (B*T, curr_channels)

        v_feat = torch.cat([pos_feat, obj_feat], dim=1)
        s_feat = self.state_encoder(state_seq)
        combined = torch.cat([v_feat, s_feat], dim=1)
        combined = self.deal_transformer_input(combined)   # (B*T, d_model)
        combined = combined.view(B, T, -1)
        combined = combined + self.pos_emb[:, :T, :]  # 加入位置编码
        
        # Transformer: 让每一帧都吸收到其他帧的信息
        trafo_out = self.transformer_encoder(combined)  # (B, T, d_model)
        
        # 取最后一帧预测
        return self.fusion(trafo_out[:, -1, :])
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCNet_Transformer(mode="vits14").to(device)
    x = torch.randn(4, 4, 3, 224, 224).to(device)  # (B, T, C, H, W)
    state = torch.randn(4, 4, 25).to(device)        # (B, T, joint_dim)
    out = model(x, state)
    print("Output shape:", out.shape)  # [B, action_dim]