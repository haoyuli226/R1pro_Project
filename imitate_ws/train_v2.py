import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pkl_2_dataset_v2 import BCDataset
import time
from bc_model_v5 import BCNet_Transformer
from torch.utils.data import Subset
import torchvision.transforms as T
import numpy as np


pkl_dir = "/home/nvidia/imitation_data_pipeline/imitate_ws/output"

batch_size = 64          
num_epochs = 60          
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = "/home/nvidia/imitation_data_pipeline/imitate_ws/bc_model_v4.pth"
val_split = 0.2         # 验证集比例

if __name__ == "__main__":
    train_transform = T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #  定义验证集变换 (仅 Resize, 无随机性)
    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_pkl_files = sorted([
        os.path.join(pkl_dir, f)
        for f in os.listdir(pkl_dir)
        if f.endswith(".pkl")
    ])
    
    np.random.seed(42)
    np.random.shuffle(all_pkl_files)

    # 按 Demo 数量划分
    split_idx = int(len(all_pkl_files) * (1 - val_split))
    train_files = all_pkl_files[:split_idx]
    val_files = all_pkl_files[split_idx:]

    print(f"Training on {len(train_files)} demos, Validating on {len(val_files)} demos.")

    train_dataset = BCDataset(train_files, window_size=4, transform=train_transform)
    # 验证集必须使用训练集的 stats，否则数据分布会错位
    val_dataset = BCDataset(val_files, window_size=4, transform=val_transform, stats=train_dataset.stats)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    
    model = BCNet_Transformer(joint_dim=train_dataset[0]['joint_state'].shape[-1], 
                  action_dim=train_dataset[-1]['action'].shape[0]).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    start_time = time.time()

    print("Start training...")
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        train_running_loss = 0.0
        for i, batch in enumerate(train_loader):
            images = batch["image"].to(device)  # (B, T, C, H, W)
            joints = batch["joint_state"].to(device)
            actions = batch["action"].to(device)
            depths = batch["depth"].to(device)

            optimizer.zero_grad()
            outputs = model(images, joints, depths)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

        avg_train_loss = train_running_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                joints = batch["joint_state"].to(device)
                actions = batch["action"].to(device)
                depths = batch["depth"].to(device)

                outputs = model(images, joints, depths)
                loss = criterion(outputs, actions)
                val_running_loss += loss.item()
        
        avg_val_loss = val_running_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # 如果验证集表现最好，保存该模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Best model saved with Val Loss: {best_val_loss:.6f}")

    total_time = time.time() - start_time
    print(f"Training finished. Total time: {total_time/60:.2f} min")