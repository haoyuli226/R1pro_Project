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
from bc_model import BCNet_Transformer
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

        self.action_fifo = deque() # 存储即将执行的动作序列
        self.chunk_size = 8 # 与训练时一致

        self.current_gripper_pos = None  # 初始化为 None
        self.latest_depth = None         # 顺便初始化深度图，防止之前提到的潜在报错

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

        self.depth_sub = self.create_subscription(
            Image,
            '/hdas/camera_head/depth/depth_registered',
            self.depth_callback,
            qos_profile
        )
        
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            qos_profile
        )

        self.gripper_sub = self.create_subscription(
            JointState,
            '/hdas/feedback_gripper_right',
            self.gripper_callback,
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
        

        self.window_size = 4 # 与训练时一致
        self.image_queue = deque(maxlen=self.window_size) 
        self.state_queue = deque(maxlen=self.window_size)
        self.depth_queue = deque(maxlen=self.window_size)

        self.get_logger().info("BC Controller Node has been started.")

    def load_model(self):
        model = BCNet_Transformer().to(self.device)
        model_path = "/home/nvidia/imitation_data_pipeline/imitate_ws/bc_model_v6.pth"
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"model pth not exist: {model_path}")
            raise FileNotFoundError(f"model pth not exist: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        
        self.get_logger().info(f"loaded model from {model_path}")
        return model

    def preprocess_image_single(self, cv_image):
        """
        必须与训练时的 val_transform 保持完全一致
        仅仅处理单张不带batch维度的图片
        """
        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)            
            resized = cv2.resize(rgb_image, (224, 224))
            image_pt = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            return normalize(image_pt)
        except Exception as e:
            self.get_logger().error(f"Preprocess failed: {e}")
            return None

    def image_callback(self, msg):
        self.get_logger().info("Received new image frame")
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

    def preprocess_depth_single(self, cv_depth):
        """
        深度图预处理：必须与训练时的 apply_sync_transform 保持一致
        """
        try:
            depth_np = np.nan_to_num(cv_depth, nan=0.0, posinf=3.0, neginf=0.0)
            depth_resized = cv2.resize(depth_np, (224, 224), interpolation=cv2.INTER_NEAREST)
            depth_pt = torch.from_numpy(depth_resized).float().unsqueeze(0) # [1, 224, 224]
            depth_pt = torch.clamp(depth_pt, 0.0, 3.0) / 3.0
            return depth_pt
        except Exception as e:
            self.get_logger().error(f"Depth preprocess failed: {e}")
            return None

    def depth_callback(self, msg):
        try:
            # 如果是 16UC1，需要除以 1000 转化成米
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if msg.encoding == '16UC1':
                cv_depth = cv_depth.astype(np.float32) / 1000.0
            self.latest_depth = cv_depth
        except Exception as e:
            self.get_logger().error(f"depth callback failed: {e}")

    def gripper_callback(self, msg):
        self.current_gripper_pos = msg.position[0]

    def get_full_state(self):
        if self.current_joint_state is None or self.current_gripper_pos is None:
            return None
        # 合并 24 维关节 + 1 维夹爪 = 25 维
        full_state = np.zeros(25, dtype=np.float32)
        full_state[0:24] = self.current_joint_state
        full_state[24] = self.current_gripper_pos
        return full_state

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
        # 1. 检查基础数据
        full_state = self.get_full_state()
        if self.latest_image is None or full_state is None:
            return

        # 更新观测队列
        self.image_queue.append(self.latest_image)
        self.state_queue.append(full_state)
        self.depth_queue.append(self.latest_depth)

        # 2. 推理逻辑：当缓冲区快空时，触发模型预测
        # 注意：为了实时性，我们通常在缓冲区还剩 1-2 个动作时就开始异步预测
        if len(self.image_queue) == self.window_size and len(self.action_fifo) <= 1:
            self.get_logger().info("Buffer low, running inference...")
            try:
                # 准备 Tensor (代码同你之前的逻辑)
                img_list = [self.preprocess_image_single(img) for img in self.image_queue]
                depth_list = [self.preprocess_depth_single(d) for d in self.depth_queue]
                state_list = [(torch.from_numpy(s).to(self.device) - self.joint_mean)/self.joint_std for s in self.state_queue]

                img_tensor = torch.stack(img_list).unsqueeze(0).to(self.device)
                joint_tensor = torch.stack(state_list).unsqueeze(0).to(self.device)
                depth_tensor = torch.stack(depth_list).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    # 输出维度: [1, chunk_size, action_dim]
                    output = self.model(img_tensor, joint_tensor, depth_tensor)

                # 反归一化
                action_chunk = output.squeeze(0) # [6, 8]
                action_denorm = action_chunk * self.action_std + self.action_mean
                action_np = action_denorm.cpu().numpy()

                # 将预测的 6 个动作放入执行队列
                self.action_fifo.clear() # 清空旧的，换成最新的预测
                for i in range(self.chunk_size):
                    self.action_fifo.append(action_np[i])

            except Exception as e:
                self.get_logger().error(f"Inference failed: {e}")

        # 3. 执行逻辑：从队列中取出一个动作发布
        if len(self.action_fifo) > 0:
            current_action = self.action_fifo.popleft()

            arm_delta = current_action[:7]
            gripper_target = current_action[7]

            # 初始化 target_q
            if not self.is_initialized:
                self.target_q = self.current_joint_state[:7].copy()
                self.is_initialized = True

            # 增量累加
            if np.linalg.norm(arm_delta) > 0.0005:
                self.target_q += arm_delta

            # 发布指令
            arm_msg = JointState()
            arm_msg.header.stamp = self.get_clock().now().to_msg()
            arm_msg.position = self.target_q.tolist()
            self.arm_publisher.publish(arm_msg)

            gripper_msg = JointState()
            gripper_msg.position = [float(gripper_target)]
            self.gripper_publisher.publish(gripper_msg)




    


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