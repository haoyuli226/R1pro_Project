import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class SEBlock(nn.Module):
    """SEBlock：通道注意力机制，自动筛选对抓取任务最关键的视觉特征通道"""
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
    """SpatialSoftmax：将抽象的卷积特征图强制转换为直观的空间二维坐标 (X, Y)"""
    def __init__(self, height, width, channel, temperature=0.5):
        super(SpatialSoftmax, self).__init__()
        self.height, self.width, self.channel = height, width, channel
        self.temperature = temperature

        # 生成固定的坐标网格 [-1, 1]
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
        # 计算特征图的概率重心
        expected_x = torch.sum(probs * self.pos_x, dim=-1)  
        expected_y = torch.sum(probs * self.pos_y, dim=-1)
        return torch.cat([expected_x, expected_y], dim=-1) # [B, C*2]
    

class BCNet_Transformer(nn.Module):
    """
    基于 Transformer 的时序动作块预测模型 (Action Chunking)
    完美适配 8维状态 -> 8维动作 的笛卡尔空间控制
    """
    def __init__(self, mode="resnet", state_dim=8, action_dim=8, nhead=8, 
                 num_layers=2, window_size=4, chunk_size=8):
        super(BCNet_Transformer, self).__init__()
        self.mode = mode
        self.chunk_size = chunk_size
        
        # 将 RGB (3通道) + Depth (1通道) 降维到骨干网络需要的 3通道
        self.input_adapter = nn.Conv2d(4, 3, kernel_size=1)

        # 视觉骨干网络选择
        if mode == "vits14":
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            # 冻结大部分参数，只微调最后两层，防止小规模模仿数据导致灾难性遗忘
            for param in self.backbone.parameters(): param.requires_grad = False
            for param in self.backbone.blocks[-2:].parameters(): param.requires_grad = True
            curr_channels, feat_size = 384, 16 
        else:
            res18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            # 截取 ResNet18 的前 7 层，保留空间分辨率
            self.backbone = nn.Sequential(*list(res18.children())[:7])  
            curr_channels = 256
            feat_size = 14        

        self.d_model = 512
        # 视觉坐标 (C*2) + 全局语义 (C) + 状态特征 (128)
        input_dim = curr_channels * 2 + curr_channels + 128 
        self.deal_transformer_input = nn.Linear(input_dim, self.d_model)

        self.shared_se = SEBlock(curr_channels)
        self.pos_extractor = SpatialSoftmax(feat_size, feat_size, curr_channels)
        self.obj_extractor = nn.AdaptiveAvgPool2d((1, 1))

        # 历史帧时间位置编码
        self.pos_emb = nn.Parameter(torch.zeros(1, window_size, self.d_model))
        
        # 状态编码器 (完美吸收 8维位姿信息)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.PReLU()
        )

        # Learnable Action Queries (预测未来的探测器)
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, self.d_model))
        nn.init.trunc_normal_(self.action_queries, std=.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=nhead, 
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 动作解码器
        self.fusion = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.PReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, img_seq, depth_seq, state_seq):
        """
        img_seq: [B, T, 3, 224, 224]
        depth_seq: [B, T, 1, 224, 224]
        state_seq: [B, T, 8]  -> [xyz(3), quat(4), gripper(1)]
        """
        B, T, _, H, W = img_seq.size()
        
        # 1. 多模态输入拼接
        img_in = torch.cat([img_seq, depth_seq], dim=2) 
        img_in = img_in.view(B * T, 4, H, W)    
        state_seq = state_seq.view(B * T, -1)   

        img_in = self.input_adapter(img_in)  
        
        # 2. 视觉特征提取
        if self.mode == "vits14":
            with torch.no_grad(): 
                feat_dict = self.backbone.forward_features(img_in)
                x = feat_dict["x_norm_patchtokens"] 
                x = x.reshape(B * T, 16, 16, 384).permute(0, 3, 1, 2).contiguous() 
        else:
            x = self.backbone(img_in)

        # 3. 视觉注意力与坐标映射
        x = self.shared_se(x)
        pos_feat = self.pos_extractor(x)  
        obj_feat = self.obj_extractor(x).view(x.size(0), -1)    

        # 4. 状态特征编码与全局特征拼接
        v_feat = torch.cat([pos_feat, obj_feat], dim=1)
        s_feat = self.state_encoder(state_seq)
        
        combined = torch.cat([v_feat, s_feat], dim=1)
        combined = self.deal_transformer_input(combined)   # (B*T, d_model)
        combined = combined.view(B, T, -1)
        
        # 加入历史帧时间位置编码
        combined = combined + self.pos_emb[:, :T, :]  
        
        # 5. Transformer Action Chunking
        queries = self.action_queries.expand(B, -1, -1) 
        full_seq = torch.cat([combined, queries], dim=1)
        
        trafo_out = self.transformer_encoder(full_seq) 
        
        # 剥离出最后的 Action Queries 输出
        action_outputs = trafo_out[:, T:, :] 
        
        # 6. 解码预测动作
        raw_out = self.fusion(action_outputs) # [B, chunk_size, 8]

        # ==========================================
        # 核心物理约束：四元数增量 L2 归一化投影
        # ==========================================
        pos_pred = raw_out[:, :, 0:3]
        quat_pred = raw_out[:, :, 3:7]
        gripper_pred = raw_out[:, :, 7:8]

        # 强行将四元数拉回单位超球面，保证物理合法性
        quat_pred = F.normalize(quat_pred, p=2, dim=-1)

        # 重新无损拼接
        out = torch.cat([pos_pred, quat_pred, gripper_pred], dim=-1)

        return out
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")
    
    # 单元测试：实例化模型
    model = BCNet_Transformer(mode="resnet", state_dim=8, action_dim=8).to(device)
    
    # 模拟 BCDataset 提供的数据流
    batch_size = 4
    window_size = 4
    chunk_size = 8
    
    dummy_img = torch.randn(batch_size, window_size, 3, 224, 224).to(device)      
    dummy_depth = torch.randn(batch_size, window_size, 1, 224, 224).to(device)  
    dummy_state = torch.randn(batch_size, window_size, 8).to(device)            
    
    # 网络前向传播
    out = model(dummy_img, dummy_depth, dummy_state)
    
    print("\n[+] Network Check Passed!")
    print(f"    Expected Output Shape: [{batch_size}, {chunk_size}, 8]")
    print(f"    Actual Output Shape:   {list(out.shape)}")
    
    # 终极检验：测试四元数是否严格满足单位长度
    sample_quat = out[0, 0, 3:7]
    sq_sum = torch.sum(sample_quat ** 2).item()
    print(f"\n[+] Physics Constraint Check:")
    print(f"    Quaternion L2 Norm: {sq_sum:.6f} (Must be 1.0)")