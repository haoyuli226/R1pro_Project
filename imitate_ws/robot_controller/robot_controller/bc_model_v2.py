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
    def __init__(self, height, width, channel):
        super(SpatialSoftmax, self).__init__()
        self.height, self.width, self.channel = height, width, channel
        # 使用 indexing='ij' 确保第一个张量是行(y)，第二个是列(x)
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1, 1, self.height),
            torch.linspace(-1, 1, self.width),
            indexing='ij'
        )
        self.register_buffer('pos_x', pos_x.reshape(-1))
        self.register_buffer('pos_y', pos_y.reshape(-1))

    def forward(self, x):
        b, c, h, w = x.size()
        probs = F.softmax(x.view(b, c, -1), dim=-1)
        expected_x = torch.sum(probs * self.pos_x, dim=-1)  # [B, 512, 49] -> [B, 512]
        expected_y = torch.sum(probs * self.pos_y, dim=-1)
        return torch.cat([expected_x, expected_y], dim=-1)
    
class BCNet(nn.Module):
    def __init__(self, joint_dim=24, action_dim=8):
        super(BCNet, self).__init__()
        
        res18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(res18.children())[:6])  # [B, 128, 28, 28] 
        for param in self.backbone.parameters():
            param.requires_grad = False  # 冻结预训练权重

        curr_channels = 128
        feat_size = 28

        self.shared_se = SEBlock(curr_channels)
        # 位置信息提取
        self.pos_extractor = SpatialSoftmax(feat_size, feat_size, curr_channels)
        # 语义信息提取
        self.obj_extractor = nn.AdaptiveAvgPool2d((1, 1))

        self.state_encoder = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.PReLU(),
            nn.Linear(128, 256),
            nn.PReLU()
        )

        # 视觉坐标输出 512*2=1024 维
        self.fusion = nn.Sequential(
            nn.Linear(256+128+256, 256),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, action_dim)
        )

    def forward(self, img, state):
        x = self.backbone(img)        # (B, 128, 28, 28)
        x = self.shared_se(x)        # (B, 128, 28, 28)
        pos_feat = self.pos_extractor(x)  # (B, 256)
        obj_feat = self.obj_extractor(x).view(x.size(0), -1)  # (B, 128)
        v_feat = torch.cat([pos_feat, obj_feat], dim=1)  #

        s_feat = self.state_encoder(state) # (B, 256)
        
        combined = torch.cat([v_feat, s_feat], dim=1)
        return self.fusion(combined)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCNet().to(device)
    x = torch.randn(4, 3, 224, 224).to(device)
    state = torch.randn(4, 24).to(device)
    out = model(x, state)
    print("Output shape:", out.shape)  # [B, 8]

    # optimizer = optim.Adam([
    #     {'params': model.backbone.parameters(), 'lr': 1e-5},
    #     {'params': model.shared_se.parameters(), 'lr': 1e-4},
    #     {'params': model.pos_extractor.parameters(), 'lr': 1e-4},
    #     {'params': model.fusion.parameters(), 'lr': 1e-4}
    # ])