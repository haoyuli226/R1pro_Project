import os
import cv2
import numpy as np
import torch
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, JointState
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

# =====================================
# 1️⃣ 基本配置
# =====================================

# 所有 rosbag 所在目录
bags_root = '/home/nvidia/imitation_data_pipeline/bags'

# 你录制的 10 个示教包
bag_names = [f'pick_cup_{i:03d}' for i in range(1, 11)]

# 输出目录
output_dir = './processed_data'
os.makedirs(output_dir, exist_ok=True)

# 需要读取的 topic
IMAGE_TOPIC = '/hdas/camera_head/rgb/image_rect_color'
ARM_TOPIC = '/motion_target/target_joint_state_arm_right'
GRIPPER_TOPIC = '/motion_target/target_position_gripper_right'

bridge = CvBridge()

# 最终数据容器
all_images = []   # 存放处理好的图像
all_actions = []  # 存放 7 关节 + 1 夹爪


# =====================================
# 2️⃣ 遍历所有 bag
# =====================================

for bag_name in bag_names:

    bag_path = os.path.join(bags_root, bag_name)
    print(f'\nProcessing {bag_path}')

    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # 存储原始记录 (timestamp, data)
    joint_records = []
    gripper_records = []
    image_records = []

    # =====================================
    # 3️⃣ 读取 bag 内所有消息
    # =====================================

    while reader.has_next():
        topic, data, _ = reader.read_next()

        # ---------------------------------
        # 3.1 读取右臂目标关节
        # ---------------------------------
        if topic == ARM_TOPIC:

            msg = deserialize_message(data, JointState)

            # 将 ROS 时间戳转换为 float 秒
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            # 将 name 和 position 对齐为字典
            joint_dict = dict(zip(msg.name, msg.position))

            # 只取右臂 7 个关节
            right_arm = [
                joint_dict.get(f'right_arm_joint{i+1}', 0.0)
                for i in range(7)
            ]

            joint_records.append(
                (stamp, np.array(right_arm, dtype=np.float32))
            )

        # ---------------------------------
        # 3.2 读取夹爪目标
        # ---------------------------------
        elif topic == GRIPPER_TOPIC:

            msg = deserialize_message(data, JointState)

            if len(msg.position) > 0:
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                # 这里除以100是假设单位是cm → m
                gripper = np.array(
                    [msg.position[0] / 100.0],
                    dtype=np.float32
                )

                gripper_records.append((stamp, gripper))

        # ---------------------------------
        # 3.3 读取图像
        # ---------------------------------
        elif topic == IMAGE_TOPIC:

            msg = deserialize_message(data, Image)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            # ROS Image → OpenCV BGR 格式
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 统一尺寸到 224x224（适配 CNN）
            cv_img = cv2.resize(cv_img, (224, 224))

            # OpenCV 默认 BGR → 转换为 RGB
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            # uint8 [0,255] → float32 [0,1]
            cv_img = cv_img.astype(np.float32) / 255.0

            # HWC → CHW（PyTorch 格式）
            cv_img = np.transpose(cv_img, (2, 0, 1))

            image_records.append((stamp, cv_img))

    # =====================================
    # 4️⃣ 时间排序（保证时间单调递增）
    # =====================================

    joint_records.sort(key=lambda x: x[0])
    gripper_records.sort(key=lambda x: x[0])
    image_records.sort(key=lambda x: x[0])

    if not joint_records or not gripper_records:
        print(f'Skip {bag_name}, missing data')
        continue

    # =====================================
    # 5️⃣ 时间对齐（以图像为基准）
    # 双指针寻找最近时间动作
    # 时间复杂度 O(N)
    # =====================================

    j_idx = 0
    g_idx = 0

    for img_stamp, img in image_records:

        # 找到时间最接近的关节目标
        while j_idx + 1 < len(joint_records) and \
              abs(joint_records[j_idx+1][0] - img_stamp) < \
              abs(joint_records[j_idx][0] - img_stamp):
            j_idx += 1

        # 找到时间最接近的夹爪目标
        while g_idx + 1 < len(gripper_records) and \
              abs(gripper_records[g_idx+1][0] - img_stamp) < \
              abs(gripper_records[g_idx][0] - img_stamp):
            g_idx += 1

        joint = joint_records[j_idx][1]
        gripper = gripper_records[g_idx][1]

        # 拼接为 8 维动作向量
        action = np.concatenate([joint, gripper], axis=0)

        all_images.append(img)
        all_actions.append(action)

    print(f'{bag_name} done. Image samples: {len(image_records)}')


# =====================================
# 6️⃣ 转为 PyTorch Tensor
# =====================================

images_tensor = torch.tensor(
    np.stack(all_images),
    dtype=torch.float32
)

actions_tensor = torch.tensor(
    np.stack(all_actions),
    dtype=torch.float32
)

print('\n---------------------------------')
print('Final dataset size:', images_tensor.shape[0])
print('Image tensor shape:', images_tensor.shape)
print('Action tensor shape:', actions_tensor.shape)
print('---------------------------------')

# =====================================
# 7️⃣ 保存为 .pt 文件
# =====================================

torch.save(
    {
        'images': images_tensor,   # [N, 3, 224, 224]
        'actions': actions_tensor  # [N, 8]
    },
    os.path.join(output_dir, 'bc_dataset.pt')
)

print(f'Saved to {output_dir}/bc_dataset.pt')