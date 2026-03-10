import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as T
import random

stat_path = "/home/nvidia/imitation_data_pipeline/imitate_ws/stats.pkl"

def caculate_stats(samples):
    joint_states = np.array([s["obs"]["joint_state"] for s in samples])
    actions = np.array([s["action"] for s in samples])

    eps = 1e-3
    joint_std = np.std(joint_states, axis=0)+1e-9
    # 如果 std 小于 eps，则强行置为 1.0（即不缩放）或 eps
    joint_std = np.where(joint_std < eps, 1.0, joint_std)

    stats = {
        "joint_mean": joint_states.mean(axis=0),
        "joint_std": joint_std,
        "action_mean": actions.mean(axis=0),
        "action_std": actions.std(axis=0) + 1e-9
    }
    return stats

class BCDataset(Dataset):
    """
    Behavioral Cloning Dataset
    基于处理后的 pkl 文件：
      sample = {
        'obs': {
            'image': latest_image.copy(),
            'joint_state': combined_obs_state
        },
        'action': action_vec # 7维右臂增量 + 1维夹爪位置
      }
    支持 image_size 参数来减小图像分辨率，加快训练
    """

    def __init__(self, pkl_files, transform=None, stats=None):
        """
        Args:
            pkl_dir: 存放多个 .pkl demo 的目录
            image_size: (H, W) 训练时 resize 图像
        """
        self.samples = []

        if not isinstance(pkl_files, list) or len(pkl_files) == 0:
            raise RuntimeError(f"pkl_files must be a non-empty list. Got: {pkl_files}")

        print(f"[BCDataset] Loading {len(pkl_files)} demos...")
        total = 0
        for pkl_path in pkl_files: # 直接遍历文件列表
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                self.samples.extend(data)
                total += len(data)
        print(f"[BCDataset] Total samples: {total}")

        # 图像增强设置
        self.transform = transform if transform is not None else T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        if stats is not None:
            self.stats = stats
        else:
            self.stats = caculate_stats(self.samples)
            with open(stat_path, "wb") as f:
                pickle.dump(self.stats, f)
        print("[BCDataset] Calculated stats:")
        print("  - Joint state mean:", self.stats['joint_mean'])
        print("  - Joint state std:", self.stats['joint_std'])
        print("  - Action mean:", self.stats['action_mean'])
        print("  - Action std:", self.stats['action_std'])
        with open(stat_path, "wb") as f:
            pickle.dump(self.stats, f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_np = sample["obs"]["image"]
        image_tensor = self.transform(image_np)

        joint = sample["obs"]["joint_state"].copy()
        action = sample["action"].copy()

        joint = (joint - self.stats['joint_mean']) / self.stats['joint_std']
        joint_tensor = torch.from_numpy(joint).float()

        action = (action - self.stats['action_mean']) / self.stats['action_std']
        action_tensor = torch.from_numpy(action).float()

        # # 24 维里前 18 维是弧度 (7右臂+4躯干+7左臂)，后 6 维是底盘 (补0)，暂不使用
        # # 粗略归一化：将弧度除以 3.14，使其分布在 [-1, 1]
        # joint[:18] = joint[:18] / 3.14159
        # joint_tensor = torch.from_numpy(joint).float()
        
        # # 关键：由于 delta 很小 (0.1以内)，乘以 10 使其量级接近 1.0，利于收敛
        # action_scaled = action.copy()
        # action_scaled[0:7] = action[0:7] * 10.0 # 放大增量动作
        # action_scaled[7] = action[7] / 100.0    # 缩小夹爪到 [0, 1]
        
        # # ======== 注意：输出到topic前的动作需要进行反归一化 ========

        # action_tensor = torch.from_numpy(action_scaled).float()

        return {
            "image": image_tensor, 
            "joint_state": joint_tensor, 
            "action": action_tensor
        }
    
if __name__ == "__main__":
    # 简单测试数据加载
    pkl_dir = "/home/nvidia/imitation_data_pipeline/imitate_ws/output"
    all_pkl_files = sorted([
        os.path.join(pkl_dir, f)
        for f in os.listdir(pkl_dir)
        if f.endswith(".pkl")
    ])
    data = BCDataset(all_pkl_files)
