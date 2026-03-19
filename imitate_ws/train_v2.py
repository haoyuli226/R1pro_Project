import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np

from pkl_to_dataset_v4 import BCDataset 
from bc_model_v7 import BCNet_Transformer

# ==========================================
# 核心亮点：定制化位姿联合损失函数 (Pose Action Loss)
# ==========================================
class PoseActionLoss(nn.Module):
    """
    针对 8维动作空间 (XYZ + 四元数 + Gripper) 定制的联合损失函数
    包含抗双重覆盖 (Antipodal-Aware) 的四元数距离惩罚
    """
    def __init__(self, pos_weight=1.0, rot_weight=1.0, grip_weight=1.0):
        super(PoseActionLoss, self).__init__()
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.grip_weight = grip_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # pred / target shape: [Batch, chunk_size, 8]
        # 拆分特征
        pos_pred, quat_pred, grip_pred = pred[..., 0:3], pred[..., 3:7], pred[..., 7:8]
        pos_gt, quat_gt, grip_gt = target[..., 0:3], target[..., 3:7], target[..., 7:8]

        # 1. 平移损失 (MSE)
        loss_pos = self.mse(pos_pred, pos_gt)

        # 2. 夹爪损失 (MSE)
        loss_grip = self.mse(grip_pred, grip_gt)

        # 3. 旋转损失 (Antipodal-Aware Loss)
        # 计算两个四元数的内积 (Dot Product)
        # 注意：因为 Dataset 中保护了四元数不被标准化，这里的 quat_gt 依然是合法的单位四元数
        dot_product = torch.sum(quat_pred * quat_gt, dim=-1) 
        # 使用 1 - (内积的平方) 作为 Loss。当 q1 = q2 或 q1 = -q2 时，内积平方均为 1，Loss 为 0。
        loss_rot = 1.0 - torch.mean(dot_product ** 2)

        # 总损失
        loss_total = self.pos_weight * loss_pos + self.rot_weight * loss_rot + self.grip_weight * loss_grip

        return loss_total, loss_pos, loss_rot, loss_grip

# ==========================================

pkl_dir = "/home/nvidia/imitation_data_pipeline/imitate_ws/output"
batch_size = 64          
num_epochs = 60          
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = "/home/nvidia/imitation_data_pipeline/imitate_ws/bc_model_v6.pth"
val_split = 0.2         

if __name__ == "__main__":
    print(f"[*] Using device: {device}")
    
    # 训练集数据增强
    train_transform = T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证集变换 (无随机性)
    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_pkl_files = sorted([
        os.path.join(pkl_dir, f) for f in os.listdir(pkl_dir) if f.endswith(".pkl")
    ])
    
    np.random.seed(42)
    np.random.shuffle(all_pkl_files)

    split_idx = int(len(all_pkl_files) * (1 - val_split))
    train_files = all_pkl_files[:split_idx]
    val_files = all_pkl_files[split_idx:]

    print(f"[*] Training on {len(train_files)} demos, Validating on {len(val_files)} demos.")

    # 实例化 Dataset
    train_dataset = BCDataset(train_files, window_size=4, chunk_size=8, transform=train_transform)
    val_dataset = BCDataset(val_files, window_size=4, chunk_size=8, transform=val_transform, stats=train_dataset.stats)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 实例化模型：传入正确的 8 维状态和动作参数
    model = BCNet_Transformer(
        mode="resnet", # 如果显存够，可以换成 "vits14"
        state_dim=train_dataset[0]['state'].shape[-1],   # 应为 8
        action_dim=train_dataset[0]['action'].shape[-1], # 应为 8
        window_size=4,
        chunk_size=8
    ).to(device)

    # 使用我们定制的联合损失函数
    criterion = PoseActionLoss(pos_weight=1.0, rot_weight=1.0, grip_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)

    best_val_loss = float('inf')
    start_time = time.time()

    print("\n[+] Start training...")
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        train_running_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            images = batch["image"].to(device) 
            depths = batch["depth"].to(device) # 新增：深度图
            states = batch["state"].to(device) # 修正：改为 state
            actions = batch["action"].to(device)

            optimizer.zero_grad()
            
            # 修正：参数顺序对齐 model 的 forward(img_seq, depth_seq, state_seq)
            outputs = model(images, depths, states) 
            
            # 计算联合 Loss
            loss, l_pos, l_rot, l_grip = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

        avg_train_loss = train_running_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_running_loss = 0.0
        val_pos, val_rot, val_grip = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                depths = batch["depth"].to(device)
                states = batch["state"].to(device)
                actions = batch["action"].to(device)

                outputs = model(images, depths, states)
                loss, l_pos, l_rot, l_grip = criterion(outputs, actions)
                
                val_running_loss += loss.item()
                val_pos += l_pos.item()
                val_rot += l_rot.item()
                val_grip += l_grip.item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        
        # 打印详细的 Loss 构成，方便 debug
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Total: {avg_val_loss:.4f} "
              f"(Pos: {val_pos/len(val_loader):.4f}, Rot: {val_rot/len(val_loader):.4f}, Grip: {val_grip/len(val_loader):.4f})")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Best model saved with Val Loss: {best_val_loss:.6f}")

    total_time = time.time() - start_time
    print(f"\nTraining finished. Total time: {total_time/60:.2f} min")