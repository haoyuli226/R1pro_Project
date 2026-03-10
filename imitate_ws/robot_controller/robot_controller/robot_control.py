from sensor_msgs.msg import JointState
import rclpy
from rclpy.node import Node

# def main(args=None):
#     msg = JointState()
#     # 1. 设置时间戳
#     msg.header.stamp = node.get_clock().now().to_msg()

#     # 2. 定义 4 个关节的目标位置 (假设 0.0 是直立，-0.2 是弯曲)
#     # 注意：具体数值需参考机器人当前的 position 反馈
#     msg.position = [-0.1, -0.1, -0.2, -0.2] 

#     # 3. 设置运行速度限制 (文档建议最大 1.5)
#     msg.velocity = [0.5, 0.5, 0.5, 0.5] 

#     # 4. 发布到 /motion_target/target_joint_state_torso
#     publisher.publish(msg)

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_control')
        self.get_logger().info('Robot Controller Node has been started.')
        self.publisher = self.create_publisher(JointState, '/motion_target/target_joint_state_torso', 10)
        self.timer = self.create_timer(1.0, self.publish_joint_state)

    def publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = [-0.1, -0.1, -0.2, -0.2] 
        msg.velocity = [0.5, 0.5, 0.5, 0.5] 
        self.publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()