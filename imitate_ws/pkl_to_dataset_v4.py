import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as T
import random
import torchvision.transforms.functional as TF

stat_path = "/home/nvidia/imitation_data_pipeline/imitate_ws/stats.pkl"

def caculate_stats(samples):
    states = np.array([s["obs"]["state"] for s in samples])
    actions = np.array([s["action"] for s in samples])

    eps = 1e-3
    state_std = np.std(states, axis=0) + 1e-9
    state_std = np.where(state_std < eps, 1.0, state_std)
    
    action_std = np.std(actions, axis=0) + 1e-9
    action_std = np.where(action_std < eps, 1.0, action_std)

    stats = {
        "state_mean": states.mean(axis=0),
        "state_std": state_std,
        "action_mean": actions.mean(axis=0),
        "action_std": action_std
    }
    
    # ================= 核心修正：保护四元数 =================
    # state: [x, y, z, qx, qy, qz, qw, gripper] (索引 3,4,5,6 是四元数)
    # action: [dx, dy, dz, dqx, dqy, dqz, dqw, gripper] (索引 3,4,5,6 是四元数)
    # 强行将四元数部分的均值设为0，标准差设为1，使其免受标准化影响
    stats["state_mean"][3:7] = 0.0
    stats["state_std"][3:7] = 1.0
    
    stats["action_mean"][3:7] = 0.0
    stats["action_std"][3:7] = 1.0
    # ========================================================

    return stats


class BCDataset(Dataset):
    """
    Behavioral Cloning Dataset
    支持观测窗口 (window_size) 和 动作分块 (chunk_size)
    """

    def __init__(self, pkl_files, window_size=4, chunk_size=8, transform=None, stats=None):
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.demos = []
        self.indices = [] 

        if not isinstance(pkl_files, list) or len(pkl_files) == 0:
            raise RuntimeError(f"pkl_files must be a non-empty list. Got: {pkl_files}")

        print(f"[BCDataset] Loading {len(pkl_files)} demos...")
        total = 0
        
        # 修复 Bug：删除了重复的循环，只保留一个
        for pkl_path in pkl_files:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                self.demos.append(data)
                current_demo_idx = len(self.demos) - 1
                for i in range(len(data)):
                    self.indices.append((current_demo_idx, i))
                total += len(data)
                print(f"  - {os.path.basename(pkl_path)}: {len(data)} samples")
                
        print(f"[BCDataset] Total samples: {total}")

        self.transform = transform if transform is not None else T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        
        if stats is not None:
            self.stats = stats
        else:
            flat_samples = [s for demo in self.demos for s in demo]
            self.stats = caculate_stats(flat_samples)
            os.makedirs(os.path.dirname(stat_path), exist_ok=True)
            with open(stat_path, "wb") as f:
                pickle.dump(self.stats, f)

        print("[BCDataset] Calculated stats:")
        print("  - State mean:", np.round(self.stats['state_mean'], 4))
        print("  - State std: ", np.round(self.stats['state_std'], 4))
        print("  - Action mean:", np.round(self.stats['action_mean'], 4))
        print("  - Action std: ", np.round(self.stats['action_std'], 4))
        
    def apply_sync_transform(self, rgb_np, depth_np):
        """同步对 RGB 和 Depth 应用变换"""
        img = T.ToPILImage()(rgb_np)
        depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=3.0, neginf=0.0)
        depth = torch.from_numpy(depth_np).unsqueeze(0).float()

        if self.transform is None:
            img = TF.to_tensor(img)
            depth = TF.resize(depth, (224, 224), interpolation=T.InterpolationMode.NEAREST)
            return img, depth

        for t in self.transform.transforms:
            if isinstance(t, T.RandomResizedCrop):
                i, j, h, w = t.get_params(img, t.scale, t.ratio)
                img = TF.resized_crop(img, i, j, h, w, t.size, t.interpolation)
                depth = TF.resized_crop(depth, i, j, h, w, t.size, T.InterpolationMode.NEAREST)
            elif isinstance(t, T.Resize):
                img = TF.resize(img, t.size, t.interpolation)
                depth = TF.resize(depth, t.size, T.InterpolationMode.NEAREST)
            elif isinstance(t, T.ColorJitter):
                img = t(img)
            elif isinstance(t, T.ToTensor):
                img = TF.to_tensor(img)
            elif isinstance(t, T.Normalize):
                img = TF.normalize(img, t.mean, t.std)

        # 深度图基础预处理：米限制在 [0, 3]，归一化到 [0, 1]
        depth = torch.clamp(depth, 0.0, 3.0) / 3.0  
        return img, depth


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        demo_idx, frame_idx = self.indices[idx]
        demo = self.demos[demo_idx]

        # 1. 构建观测历史序列 (Observation History)
        seq_samples = []
        for i in range(frame_idx - self.window_size + 1, frame_idx + 1):
            if i < 0:
                seq_samples.append(demo[0])  # Padding
            else:
                seq_samples.append(demo[i])
        
        images, depths, states = [], [], []
        seed = random.randint(0, 99999)  
        
        for s in seq_samples:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            image_np = s["obs"]["image"]
            depth_np = s["obs"]["depth"]

            img, depth = self.apply_sync_transform(image_np, depth_np)
            images.append(img)
            depths.append(depth)

            state_val = s["obs"]["state"].copy()
            # 这里的归一化极其安全，因为我们在算 stats 时已经保护了四元数维度
            state_val = (state_val - self.stats['state_mean']) / self.stats['state_std']
            states.append(torch.from_numpy(state_val).float())

        images = torch.stack(images)      # (window_size, C, H, W)
        depths = torch.stack(depths)      # (window_size, 1, H, W)
        states = torch.stack(states)      # (window_size, state_dim)

        # 2. 构建未来动作序列 (Action Chunking)
        actions = []
        for i in range(frame_idx, frame_idx + self.chunk_size):
            target_i = min(i, len(demo) - 1) # Padding
            act = demo[target_i]["action"].copy()
            # 同样受到四元数免归一化保护
            act = (act - self.stats['action_mean']) / self.stats['action_std']
            actions.append(torch.from_numpy(act).float())
        
        actions = torch.stack(actions)  # (chunk_size, action_dim)

        return {
            "image": images, 
            "depth": depths,
            "state": states, 
            "action": actions
        }
    
if __name__ == "__main__":
    pkl_dir = "/home/nvidia/imitation_data_pipeline/imitate_ws/output"
    # 添加一个保险机制，以防目录不存在
    if not os.path.exists(pkl_dir):
        print(f"Warning: Directory {pkl_dir} does not exist. Creating mock directory for testing.")
        os.makedirs(pkl_dir, exist_ok=True)
        
    all_pkl_files = sorted([
        os.path.join(pkl_dir, f) for f in os.listdir(pkl_dir) if f.endswith(".pkl")
    ])
    
    if len(all_pkl_files) > 0:
        dataset = BCDataset(all_pkl_files, window_size=4, chunk_size=8)
        print(f"Dataset length: {len(dataset)}")
        
        # 拿出一个样本看看形状对不对
        sample = dataset[0]
        print("Sample shapes:")
        print(f"  - Image: {sample['image'].shape}")
        print(f"  - Depth: {sample['depth'].shape}")
        print(f"  - State: {sample['state'].shape}")
        print(f"  - Action: {sample['action'].shape}")
    else:
        print("No .pkl files found in the directory to test.")