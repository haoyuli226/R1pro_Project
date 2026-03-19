import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # 引入专用的传感器 QoS 配置
import torch
import numpy as np
import cv2
import pickle
from collections import deque
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ROS 2 消息类型
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

# 导入你搭建的模型
from bc_model_v7 import BCNet_Transformer

# ================= 话题配置 =================
TOPIC_IMAGE = '/hdas/camera_head/rgb/image_rect_color'
TOPIC_DEPTH = '/hdas/camera_head/depth/depth_registered'
TOPIC_ARM_FEEDBACK = '/motion_control/pose_ee_arm_right'
TOPIC_GRIPPER_FEEDBACK = '/hdas/feedback_gripper_right'

TOPIC_ARM_TARGET = '/motion_target/target_pose_arm_right'
TOPIC_GRIPPER_TARGET = '/motion_target/target_position_gripper_right'

class BCInferenceNode(Node):
    def __init__(self):
        super().__init__('bc_inference_node')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"[*] 推理节点启动，使用设备: {self.device}")

        # 1. 挂载数字大脑 (加载模型和权重)
        self.window_size = 4
        self.chunk_size = 8
        self.model = BCNet_Transformer(
            mode="resnet", state_dim=8, action_dim=8, 
            window_size=self.window_size, chunk_size=self.chunk_size
        ).to(self.device)
        
        model_path = "bc_model_v6.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.get_logger().info("[*] 成功加载模型权重: bc_model_v6.pth")

        # 2. 加载翻译字典 (stats.pkl)
        with open("stats.pkl", "rb") as f:
            self.stats = pickle.load(f)
        self.get_logger().info("[*] 成功加载归一化参数 stats.pkl")

        # 3. 图像预处理流水线 (必须与验证集完全一致)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.bridge = CvBridge()

        # 4. 记忆缓冲队列 (保存最近 4 帧的数据)
        self.history_rgb = deque(maxlen=self.window_size)
        self.history_depth = deque(maxlen=self.window_size)
        self.history_state = deque(maxlen=self.window_size)

        # 缓存最新收到的 ROS 消息
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_arm_pose = None
        self.latest_gripper = None

        # 5. 订阅者 (感知器)
        # 使用 qos_profile_sensor_data 适配相机的 BEST_EFFORT 协议
        self.create_subscription(Image, TOPIC_IMAGE, self.rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, TOPIC_DEPTH, self.depth_cb, qos_profile_sensor_data)
        
        # 状态话题数据量小，底层依然是 RELIABLE，保持 10 即可
        self.create_subscription(PoseStamped, TOPIC_ARM_FEEDBACK, self.arm_pose_cb, 10)
        self.create_subscription(JointState, TOPIC_GRIPPER_FEEDBACK, self.gripper_cb, 10)

        # 6. 发布者 (运动神经)
        self.arm_pub = self.create_publisher(PoseStamped, TOPIC_ARM_TARGET, 10)
        self.gripper_pub = self.create_publisher(JointState, TOPIC_GRIPPER_TARGET, 10)

        # [新增] 物理计数器，用于替代容易失效的系统时间 throttle
        self.loop_counter = 0

        # 7. 脑电波循环 (10Hz 推理频率)
        self.timer = self.create_timer(0.1, self.inference_loop)
        self.get_logger().info("[*] 视觉神经已连通，等待感知画面填满窗口...")

    # ================= 传感器回调函数 =================
    def rgb_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if 'rgb' in msg.encoding.lower():
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.latest_rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)

    def depth_cb(self, msg):
        depth_img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        self.latest_depth = depth_img

    def arm_pose_cb(self, msg):
        self.latest_arm_pose = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        ], dtype=np.float32)

    def gripper_cb(self, msg):
        if len(msg.position) > 0:
            self.latest_gripper = msg.position[0]

    # ================= 核心推理循环 =================
    def inference_loop(self):
        self.loop_counter += 1

        # ================= 听诊器：绝对防弹版 =================
        missing = []
        if self.latest_rgb is None: missing.append('RGB主相机')
        if self.latest_depth is None: missing.append('Depth深度图')
        if self.latest_arm_pose is None: missing.append('Arm右臂位姿')
        if self.latest_gripper is None: missing.append('Gripper右夹爪')
        
        if missing:
            # 推理频率是 10Hz，所以每 20 次循环就是 2 秒
            if self.loop_counter % 20 == 0:
                self.get_logger().info(f"[!] 大脑空转中... 正在等待底层躯体通电发数据。当前缺失: {missing}")
            return
        # ========================================================

        # 1. 组装当前状态 (8维: 3平移 + 4四元数 + 1夹爪)
        curr_state_np = np.zeros(8, dtype=np.float32)
        curr_state_np[0:7] = self.latest_arm_pose
        curr_state_np[7] = self.latest_gripper

        # 保存当前物理位姿，用于后续推算目标位姿
        curr_pos = curr_state_np[0:3].copy()
        curr_quat = curr_state_np[3:7].copy()

        # 2. 数据标准化 (借助 stats.pkl)
        norm_state = (curr_state_np - self.stats['state_mean']) / self.stats['state_std']

        # 3. 处理视觉输入
        img_tensor = self.transform(self.latest_rgb)
        
        depth_np = np.nan_to_num(self.latest_depth, nan=0.0, posinf=3.0, neginf=0.0)
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).float()
        depth_tensor = TF.resize(depth_tensor, (224, 224), interpolation=T.InterpolationMode.NEAREST)
        depth_tensor = torch.clamp(depth_tensor, 0.0, 3.0) / 3.0  # [0, 1] 归一化

        # 4. 填入滑动窗口
        self.history_rgb.append(img_tensor)
        self.history_depth.append(depth_tensor)
        self.history_state.append(torch.from_numpy(norm_state).float())

        # 如果还没攒够 4 帧，先不推理，保持静止
        if len(self.history_rgb) < self.window_size:
            if self.loop_counter % 5 == 0:  # 每 0.5 秒打印一次
                self.get_logger().info(f"⏳ 正在填充时序记忆窗口: {len(self.history_rgb)}/{self.window_size}")
            return

        # 5. 组装 Batch (B=1) 喂给模型
        seq_rgb = torch.stack(list(self.history_rgb)).unsqueeze(0).to(self.device)   # [1, 4, 3, 224, 224]
        seq_depth = torch.stack(list(self.history_depth)).unsqueeze(0).to(self.device) # [1, 4, 1, 224, 224]
        seq_state = torch.stack(list(self.history_state)).unsqueeze(0).to(self.device) # [1, 4, 8]

        with torch.no_grad():
            pred_actions = self.model(seq_rgb, seq_depth, seq_state) # shape: [1, chunk_size, 8]
        
        # 6. 提取未来的第 1 步动作，并反归一化
        pred_act_np = pred_actions[0, 0, :].cpu().numpy()
        pred_act_np = pred_act_np * self.stats['action_std'] + self.stats['action_mean']

        # 拆解预测的 8 维 Action
        pred_pos_delta = pred_act_np[0:3]
        pred_quat_delta = pred_act_np[3:7]
        pred_gripper_target = pred_act_np[7]

        # 7. 还原绝对位姿 (极其关键的几何计算)
        target_pos = curr_pos + pred_pos_delta
        
        rot_curr = R.from_quat(curr_quat)
        rot_delta = R.from_quat(pred_quat_delta)
        rot_target = rot_delta * rot_curr
        target_quat = rot_target.as_quat()

        # 8. 发布控制指令给底层
        self.publish_arm_command(target_pos, target_quat)
        self.publish_gripper_command(pred_gripper_target)
        
        # 成功下发指令的反馈 (每 0.5 秒打印一次防刷屏)
        if self.loop_counter % 5 == 0:
            self.get_logger().info("🧠 大脑正常运转中：已下发控制指令！")

    def publish_arm_command(self, pos, quat):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        # [注意] 如果机器人抽搐乱动，检查这里的 frame_id 是否需要改为你真机的基座名字
        msg.header.frame_id = "base_link" 
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self.arm_pub.publish(msg)

    def publish_gripper_command(self, gripper_val):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = [float(gripper_val)]
        self.gripper_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = BCInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[!] 推理节点收到中断信号，正在安全退出...")
    finally:
        # 安全关闭逻辑，防止底层上下文重复 shutdown 报错
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()