import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class SpatialSoftmax(nn.Module):
    # SpatialSoftmax：将卷积特征转换为空间坐标，关注瓶子位置
    def __init__(self, height, width, channel):
        super(SpatialSoftmax, self).__init__()
        self.height, self.width, self.channel = height, width, channel
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
        self.backbone_low = nn.Sequential(*list(res18.children())[:6]) # 128x28x28
        self.backbone_high = nn.Sequential(*list(res18.children())[6:-2]) # 512x7x7
        
        # 将高层语义和低层几何特征融合
        self.up_sample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.compress = nn.Conv2d(256 + 128, 256, kernel_size=1)
        self.reduce = nn.Conv2d(512,256,1)
        
        # 此时特征图为 256x28x28，再做 SpatialSoftmax 精度会高得多
        self.pos_extractor = SpatialSoftmax(28, 28, 256) 
        
        self.state_encoder = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.LayerNorm(128),
            nn.PReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(256*2 + 128, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, img, state):
        feat_low = self.backbone_low(img)   # (B, 128, 28, 28)
        feat_high = self.backbone_high(feat_low) # (B, 512, 7, 7)
        feat_high = self.reduce(feat_high)  # (B, 256, 7, 7)
        
        # 融合高低层特征
        merged = torch.cat([self.up_sample(feat_high), feat_low], dim=1) # (B, 256+128, 28, 28)
        x = self.compress(merged) # (B, 256, 28, 28)
        
        v_feat = self.pos_extractor(x) # (B, 512)
        s_feat = self.state_encoder(state)
        
        return self.fusion(torch.cat([v_feat, s_feat], dim=1))
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCNet().to(device)
    x = torch.randn(4, 3, 224, 224).to(device)
    state = torch.randn(4, 24).to(device)
    out = model(x, state)
    print("Output shape:", out.shape)  # [B, 8]