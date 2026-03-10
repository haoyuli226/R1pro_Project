import torch
import torch.nn as nn
import torchvision.models as models

class BCModel(nn.Module):

    def __init__(self, freeze_backbone=True):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # 只保留特征提取层
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 512

        # 是否冻结
        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # 特征归一化
        self.norm = nn.LayerNorm(self.feature_dim)

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 8)
        )

    def forward(self, x):

        # CNN feature
        feat = self.cnn(x)
        feat = feat.view(feat.size(0), -1)

        # Normalize
        feat = self.norm(feat)

        # Action regression
        action = self.head(feat)

        return action