import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class ImageTest(Node):
    def __init__(self):
        super().__init__('image_test_node')
        self.sub = self.create_subscription(
            CompressedImage,
            '/hdas/camera_head/left_raw/image_raw_color/compressed', # 换成你那个绿屏的话题
            self.callback,
            10)

    def callback(self, msg):
        # 尝试解码
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow("Debug View", img)
            cv2.waitKey(1)
        else:
            print("解码失败！数据损坏或格式不支持")

def main():
    rclpy.init()
    node = ImageTest()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()