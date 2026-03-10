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

    def __init__(self, pkl_files, window_size=4, transform=None, stats=None):
        """
        Args:
            pkl_dir: 存放多个 .pkl demo 的目录
            image_size: (H, W) 训练时 resize 图像
        """
        self.window_size = window_size
        self.demos = []
        self.indices = []  # 存储每个样本 (demo_idx, frame_idx)

        if not isinstance(pkl_files, list) or len(pkl_files) == 0:
            raise RuntimeError(f"pkl_files must be a non-empty list. Got: {pkl_files}")

        print(f"[BCDataset] Loading {len(pkl_files)} demos...")
        total = 0
        for pkl_path in pkl_files: # 直接遍历文件列表
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                self.demos.append(data)
                # 记录该 demo 在 self.demos 中的索引
                current_demo_idx = len(self.demos) - 1
                for i in range(len(data)):
                    self.indices.append((current_demo_idx, i))
                total += len(data)

        print(f"[BCDataset] Loading {len(pkl_files)} demos...")
        total = 0
        for pkl_path in pkl_files:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                self.demos.append(data)
                for i in range(len(data)):
                    self.indices.append((len(self.demos)-1, i))
                total += len(data)
                print(f"  - {os.path.basename(pkl_path)}: {len(data)} samples")
        print(f"[BCDataset] Total samples: {total}")

        # 图像增强设置
        self.transform = transform if transform is not None else T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        # 验证集必须使用训练集的 stats，否则数据分布会错位
        if stats is not None:
            self.stats = stats
        else:
            flat_samples = [s for demo in self.demos for s in demo]
            self.stats = caculate_stats(flat_samples)
            with open(stat_path, "wb") as f:
                pickle.dump(self.stats, f)

        print("[BCDataset] Calculated stats:")
        print("  - Joint state mean:", self.stats['joint_mean'])
        print("  - Joint state std:", self.stats['joint_std'])
        print("  - Action mean:", self.stats['action_mean'])
        print("  - Action std:", self.stats['action_std'])
        
    def apply_sync_transform(self, rgb_np, depth_np):
        """
        核心：同步对 RGB 和 Depth 应用变换
        """
        img = T.ToPILImage()(rgb_np)
        # 深度图转为 Tensor 并在首位增加通道维度 (1, H, W)
        # 先处理可能存在的非法值
        depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=3.0, neginf=0.0)
        depth = torch.from_numpy(depth_np).unsqueeze(0).float()

        if self.transform is None:
            img = TF.to_tensor(img)
            depth = TF.resize(depth, (224, 224), interpolation=T.InterpolationMode.NEAREST)
            return img, depth

        # 遍历外部传入的 Compose，手动同步
        # 注意：这里假设 transform 是 T.Compose
        for t in self.transform.transforms:
            if isinstance(t, T.RandomResizedCrop):
                i, j, h, w = t.get_params(img, t.scale, t.ratio)
                img = TF.resized_crop(img, i, j, h, w, t.size, t.interpolation)
                depth = TF.resized_crop(depth, i, j, h, w, t.size, T.InterpolationMode.NEAREST)
            
            elif isinstance(t, T.Resize):
                img = TF.resize(img, t.size, t.interpolation)
                depth = TF.resize(depth, t.size, T.InterpolationMode.NEAREST)
            
            elif isinstance(t, T.ColorJitter):
                # 深度图不需要颜色抖动，只对 RGB 做
                img = t(img)
            
            elif isinstance(t, T.ToTensor):
                img = TF.to_tensor(img)
            
            elif isinstance(t, T.Normalize):
                # 深度图不适用 RGB 的 Normalize
                img = TF.normalize(img, t.mean, t.std)

        # 深度图基础预处理：米限制在 [0, 3] 
        depth = torch.clamp(depth, 0.0, 3.0) / 3.0  # 归一化到 [0, 1]
        
        return img, depth


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        demo_idx, frame_idx = self.indices[idx]
        demo = self.demos[demo_idx]

        seq_samples = []
        for i in range(frame_idx - self.window_size + 1, frame_idx + 1):
            if i < 0:
                seq_samples.append(demo[0])  # 用第一帧填充前面不足的部分
            else:
                seq_samples.append(demo[i])
        
        images = []
        joint_states = []
        depths = []
        seed = random.randint(0, 99999)  # 每个样本一个随机种子，保证同一序列内变换一致
        for s in seq_samples:
            # 固定随机种子
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            image_np = s["obs"]["image"]
            depth_np = s["obs"]["depth"]

            img, depth = self.apply_sync_transform(image_np, depth_np)
            images.append(img)
            depths.append(depth)

            joint = s["obs"]["joint_state"].copy()
            joint = (joint - self.stats['joint_mean']) / self.stats['joint_std']
            joint_tensor = torch.from_numpy(joint).float()
            joint_states.append(joint_tensor)


        images = torch.stack(images)  # (T, C, H, W)
        joint_states = torch.stack(joint_states)  # (T, joint_dim)
        depths = torch.stack(depths)  # (T, 1, H, W)

        action = seq_samples[-1]["action"].copy()   # 最后一帧的动作
        action = (action - self.stats['action_mean']) / self.stats['action_std']
        action_tensor = torch.from_numpy(action).float()

        return {
            "image": images, 
            "depth": depths,
            "joint_state": joint_states, 
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
    dataset = BCDataset(all_pkl_files, window_size=4)
    print(f"Dataset length: {len(dataset)}")