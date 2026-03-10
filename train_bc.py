import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from bc_model import BCModel
import numpy as np

# ===============================
# 1️⃣ 设备
# ===============================
device = torch.device("cpu")
print("Using device:", device)

# ===============================
# 2️⃣ 加载数据
# ===============================
data = torch.load("./processed_data/bc_dataset.pt")

images = data["images"]
actions = data["actions"]

print("Dataset loaded.")
print("Images shape:", images.shape)
print("Actions shape:", actions.shape)

print("Action mean:", actions.mean(dim=0))
print("Action std :", actions.std(dim=0))

dataset = TensorDataset(images, actions)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

print("Train size:", len(train_set))
print("Val size:", len(val_set))

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)

# ===============================
# 3️⃣ 加权MSE Loss
# ===============================
class WeightedMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.joint_weight = 1.0
        self.gripper_weight = 5.0

    def forward(self, pred, target):

        joint_loss = (pred[:, :7] - target[:, :7]) ** 2
        gripper_loss = (pred[:, 7] - target[:, 7]) ** 2

        loss = (
            self.joint_weight * joint_loss.mean() +
            self.gripper_weight * gripper_loss.mean()
        )

        return loss

criterion = WeightedMSE()

# ===============================
# 4️⃣ 初始化模型（阶段1冻结）
# ===============================
model = BCModel(freeze_backbone=True).to(device)

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Trainable params (Stage 1):", count_trainable(model))

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

# ===============================
# 5️⃣ 训练函数
# ===============================
def run_epoch(loader, train=True):

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    joint_loss_total = 0
    gripper_loss_total = 0

    for img, act in loader:

        img = img.to(device)
        act = act.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(img)

            joint_loss = ((pred[:, :7] - act[:, :7]) ** 2).mean()
            gripper_loss = ((pred[:, 7] - act[:, 7]) ** 2).mean()

            loss = 1.0 * joint_loss + 5.0 * gripper_loss

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item()
        joint_loss_total += joint_loss.item()
        gripper_loss_total += gripper_loss.item()

    return (
        total_loss / len(loader),
        joint_loss_total / len(loader),
        gripper_loss_total / len(loader)
    )

# ===============================
# 6️⃣ 两阶段训练
# ===============================

best_val = 1e9

# ---------- Stage 1 ----------
print("\n========== Stage 1: Train Head Only ==========")

for epoch in range(20):

    train_loss, train_joint, train_grip = run_epoch(train_loader, True)
    val_loss, val_joint, val_grip = run_epoch(val_loader, False)

    print(f"[Stage1][Epoch {epoch}]")
    print(f"  Train Loss: {train_loss:.5f} | Joint: {train_joint:.5f} | Grip: {train_grip:.5f}")
    print(f"  Val   Loss: {val_loss:.5f} | Joint: {val_joint:.5f} | Grip: {val_grip:.5f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "bc_model_stage1_best.pt")
        print("  ✔ Saved best Stage1 model")

# ---------- Stage 2 ----------
print("\n========== Stage 2: Fine-tune Backbone ==========")

for param in model.cnn.parameters():
    param.requires_grad = True

print("Trainable params (Stage 2):", count_trainable(model))

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=1e-4
)

for epoch in range(20):

    train_loss, train_joint, train_grip = run_epoch(train_loader, True)
    val_loss, val_joint, val_grip = run_epoch(val_loader, False)

    print(f"[Stage2][Epoch {epoch}]")
    print(f"  Train Loss: {train_loss:.5f} | Joint: {train_joint:.5f} | Grip: {train_grip:.5f}")
    print(f"  Val   Loss: {val_loss:.5f} | Joint: {val_joint:.5f} | Grip: {val_grip:.5f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "bc_model_best.pt")
        print("  ✔ Saved best overall model")

print("\nTraining complete.")
print("Best validation loss:", best_val)