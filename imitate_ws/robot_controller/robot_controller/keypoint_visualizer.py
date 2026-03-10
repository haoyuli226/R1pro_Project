import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
import torchvision.transforms as T
from bc_model_v2 import BCNet

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class KeypointVisualizer(Node):
    def __init__(self, model_path):
        super().__init__('keypoint_visualizer')
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 加载模型
        self.get_logger().info(f"Loading model from {model_path}...")
        self.model = BCNet(joint_dim=24, action_dim=8).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT, # 关键：匹配相机的尽力而为模式
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1  # 只要最新的一帧，减小延迟
        )

        # 2. 修改订阅者，使用这个 qos_profile
        self.subscription = self.create_subscription(
            Image,
            '/hdas/camera_head/rgb/image_rect_color',
            self.image_callback,
            qos_profile  # 替换原来的数字 10
        )

        # 2. 预处理定义
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.latest_cv_img = None
        self.get_logger().info("Visualizer node started. Waiting for images...")

    def image_callback(self, msg):
        try:
            # ROS Image → OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    
            # BGR → RGB
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
            # 预处理
            img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
    
            with torch.no_grad():
                # 提取 backbone 特征
                features = self.model.backbone(img_tensor)      # (1,128,28,28)
                features = self.model.shared_se(features)
    
                coords = self.model.pos_extractor(features)[0]  # (256,)
                coords = coords.cpu().numpy()
    
            # channel 数量
            C = coords.shape[0] // 2
    
            xs = coords[:C]
            ys = coords[C:]
    
            # [-1,1] → image pixel
            h, w = img_rgb.shape[:2]
            xs = ((xs + 1) / 2 * w).astype(int)
            ys = ((ys + 1) / 2 * h).astype(int)
    
            # 可视化
            display_img = cv_img.copy()
    
            # 每隔几个通道画一个点
            step = 4
    
            for i in range(0, C, step):
                x = xs[i]
                y = ys[i]
    
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(display_img, (x, y), 3, (255, 0, 255), -1)
    
            # 显示
            cv2.imshow("SpatialSoftmax Keypoints", display_img)
            cv2.waitKey(1)
    
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main():
    rclpy.init()
    # 填入你训练好的模型路径
    model_path = "/home/nvidia/imitation_data_pipeline/imitate_ws/bc_model_v2.pth"
    node = KeypointVisualizer(model_path)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()