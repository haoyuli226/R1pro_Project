import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import torch
import sys
import os
import cv2
from collections import deque
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time
from bc_model_v2 import BCNet
from torchvision import transforms as T
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
stat_path = "/home/nvidia/imitation_data_pipeline/imitate_ws/stats.pkl"

class BCControlloer(Node):
    def __init__(self):
        super().__init__('bc_controller')
        
        self.bridge = CvBridge()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"using device: {self.device}")
        
        self.model = self.load_model()
        
        self.latest_image = None
        self.image_lock = False
        
        self.current_joint_state = None
        self.last_target_q = None     # 内部指令累加器
        
        
        self.frame_count = 0
        self.last_log_time = time.time()

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.image_sub = self.create_subscription(
            Image,
            '/hdas/camera_head/rgb/image_rect_color',
            self.image_callback,
            qos_profile
        )
        
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            qos_profile
        )

        self.arm_publisher = self.create_publisher(
            JointState,
            '/motion_target/target_joint_state_arm_right',
            10
        )
        
        self.gripper_publisher = self.create_publisher(
            JointState,
            '/motion_target/target_position_gripper_right',
            10
        )

        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        self.action_buffer = deque(maxlen=5) # 存储最近5帧的预测值
        self.target_q = None # 内部维护的目标位置
        self.is_initialized = False

        with open(stat_path, "rb") as f:
            stats = pickle.load(f)
            self.joint_mean = torch.tensor(stats['joint_mean'], device=self.device)
            self.joint_std = torch.tensor(stats['joint_std'], device=self.device)
            self.action_mean = torch.tensor(stats['action_mean'], device=self.device)
            self.action_std = torch.tensor(stats['action_std'], device=self.device)
        
        self.get_logger().info("BC Controller Node has been started.")

    def load_model(self):
        model = BCNet().to(self.device)
        model_path = "/home/nvidia/imitation_data_pipeline/imitate_ws/bc_model_v2.pth"
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"model pth not exist: {model_path}")
            raise FileNotFoundError(f"model pth not exist: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        
        self.get_logger().info(f"loaded model from {model_path}")
        return model

    def preprocess_image(self, cv_image):
        """
        必须与训练时的 val_transform 保持完全一致
        """
        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)            
            resized = cv2.resize(rgb_image, (224, 224))
            image_pt = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image_pt = normalize(image_pt)
            return image_pt.unsqueeze(0).to(self.device)
        except Exception as e:
            self.get_logger().error(f"Preprocess failed: {e}")
            return None

    def image_callback(self, msg):
        if self.image_lock:
            return
            
        self.image_lock = True
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            self.latest_image = cv_image
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_log_time > 2.0:
                fps = self.frame_count / 2.0
                self.frame_count = 0
                self.last_log_time = current_time
                
        except Exception as e:
            self.get_logger().error(f"image transform failed: {e}")
        finally:
            self.image_lock = False

    def joint_callback(self, msg):
        try:
            name_to_pos = dict(zip(msg.name, msg.position))
            arm_r = np.array([name_to_pos.get(f'right_arm_joint{i}', 0.0) for i in range(1, 8)], dtype=np.float32)
            torso = np.array([name_to_pos.get(f'torso_joint{i}', 0.0) for i in range(1, 5)], dtype=np.float32)
            arm_l = np.array([name_to_pos.get(f'left_arm_joint{i}', 0.0) for i in range(1, 8)], dtype=np.float32)

            obs_state = np.zeros(24, dtype=np.float32)
            obs_state[0:7] = arm_r
            obs_state[7:11] = torso
            obs_state[11:18] = arm_l
            self.current_joint_state = obs_state
        except Exception as e:
            self.get_logger().error(f"joint state callback failed: {e}")    

    def timer_callback(self):
        if self.latest_image is None or self.current_joint_state is None:
            return 
            
        try:
            img_tensor = self.preprocess_image(self.latest_image)
            joint_raw = torch.from_numpy(self.current_joint_state).float().to(self.device)
            joint_norm = (joint_raw - self.joint_mean) / self.joint_std
            joint_tensor = joint_norm.unsqueeze(0)

            with torch.no_grad():
                output = self.model(img_tensor, joint_tensor) #  Action [B, 8]
            
            action_norm = output.squeeze(0)
            action_denorm = action_norm * self.action_std + self.action_mean
            action_np = action_denorm.cpu().numpy()
            
            arm_delta = action_np[:7]
            gripper_target = action_np[7]

            # 时域滤波 (Smoothing)?
            self.action_buffer.append(arm_delta)
            smoothed_delta = np.mean(self.action_buffer, axis=0)

            if not self.is_initialized:
                self.target_q = self.current_joint_state[:7].copy()
                self.is_initialized = True
            
            # 只有增量大于死区才累加
            if np.linalg.norm(smoothed_delta) > 0.0005: 
                self.target_q += smoothed_delta

            arm_msg = JointState()
            arm_msg.header.stamp = self.get_clock().now().to_msg()
            arm_msg.position = self.target_q.tolist()
            self.arm_publisher.publish(arm_msg)

            gripper_msg = JointState()
            gripper_msg.position = [float(gripper_target)]
            self.gripper_publisher.publish(gripper_msg)

        except Exception as e:
            self.get_logger().error(f"Control loop failed: {e}")

    


def main(args=None):
    rclpy.init(args=args)
    node = BCControlloer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()