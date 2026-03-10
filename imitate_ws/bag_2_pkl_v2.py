import os
import pickle
import numpy as np
import psutil
import cv2

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TOPIC_IMAGE = '/hdas/camera_head/rgb/image_rect_color'

# ===== [ADDED] =====
# 新增 Depth Topic
# 对应 ros2 bag 中的深度图
TOPIC_DEPTH = '/hdas/camera_head/depth/depth_registered'

TOPIC_JOINT_STATES = '/joint_states'

JOINT_NAMES = {
    'right_arm': [f'right_arm_joint{i}' for i in range(1, 8)],
    'torso': [f'torso_joint{i}' for i in range(1, 5)],
    'left_arm': [f'left_arm_joint{i}' for i in range(1, 8)],
}

TOPIC_GRIPPER_TARGET = '/motion_target/target_position_gripper_right'
TOPIC_ARM_TARGET = '/motion_target/target_joint_state_arm_right'


def print_memory_usage(step_name=""):
    mem = psutil.virtual_memory()
    print(f"[MEMORY] {step_name} | Used: {mem.used / 1024**2:.1f} MB")


def open_bag(bag_path):

    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id='sqlite3'
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    reader.open(storage_options, converter_options)

    return reader


def get_topic_types(reader):

    topic_types = {}

    for info in reader.get_all_topics_and_types():
        topic_types[info.name] = info.type

    return topic_types


def parse_bag_for_bc(bag_path, delta_clip=0.1):

    reader = open_bag(bag_path)

    topic_types = get_topic_types(reader)

    curr_obs_joint_state = None
    curr_arm_r_feedback = None
    curr_arm_target = None
    curr_gripper_target = 0.0

    latest_image = None

    # ===== [ADDED] =====
    # 用于缓存最近一帧深度图
    # 因为 depth 与 RGB topic 是异步发布
    latest_depth = None

    dataset = []

    while reader.has_next():

        topic, data, t = reader.read_next()

        msg_type = get_message(topic_types[topic])

        msg = deserialize_message(data, msg_type)

        # ---------------- Joint State ----------------
        if topic == TOPIC_JOINT_STATES:

            name_to_pos = dict(zip(msg.name, msg.position))

            arm_r = np.array(
                [name_to_pos.get(n, 0.0) for n in JOINT_NAMES['right_arm']],
                dtype=np.float32
            )

            torso = np.array(
                [name_to_pos.get(n, 0.0) for n in JOINT_NAMES['torso']],
                dtype=np.float32
            )

            arm_l = np.array(
                [name_to_pos.get(n, 0.0) for n in JOINT_NAMES['left_arm']],
                dtype=np.float32
            )

            curr_arm_r_feedback = arm_r

            obs_state = np.zeros(24, dtype=np.float32)

            obs_state[0:7] = arm_r
            obs_state[7:11] = torso
            obs_state[11:18] = arm_l

            curr_obs_joint_state = obs_state

        # ---------------- Arm Target ----------------
        elif topic == TOPIC_ARM_TARGET:

            curr_arm_target = np.array(
                msg.position[:7],
                dtype=np.float32
            )

        # ---------------- Gripper Target ----------------
        elif topic == TOPIC_GRIPPER_TARGET:

            if len(msg.position) > 0:
                curr_gripper_target = msg.position[0]

        # ==================================================
        # ===== [ADDED] ===== 深度图解析
        # ==================================================
        elif topic == TOPIC_DEPTH:

            # 获取深度编码格式
            encoding = msg.encoding.lower()

            # 你的 bag 中 encoding = 32FC1
            # 表示 float32 depth，单位 meters
            if encoding == "32fc1":

                # ROS Image.data 是 byte buffer
                # np.frombuffer 将其解释为 float32 array
                depth = np.frombuffer(
                    msg.data,
                    dtype=np.float32
                ).reshape(msg.height, msg.width)

            else:
                # 如果出现其它编码，暂时跳过
                continue

            # ---------- 深度数据清理 ----------

            # 有些像素会出现 NaN 或 inf
            # 使用 nan_to_num 转为安全值
            depth = np.nan_to_num(
                depth,
                nan=2.0,
                posinf=2.0
            )

            # ---------- 深度范围裁剪 ----------

            # 抓取任务通常只关心桌面附近
            # 限制深度范围减少背景噪声
            depth = np.clip(depth, 0.2, 1.5)

            # ---------- 深度归一化 ----------

            # 将深度值映射到 [0,1]
            depth = (depth - 0.2) / (1.5 - 0.2)

            # ---------- resize ----------

            # CNN训练通常使用224x224
            depth = cv2.resize(depth, (224, 224))

            # 保存为最近一帧深度
            latest_depth = depth

        # ---------------- RGB Image ----------------
        elif topic == TOPIC_IMAGE:

            encoding = msg.encoding.lower()

            img = np.frombuffer(
                msg.data,
                dtype=np.uint8
            ).reshape(msg.height, msg.width, -1)

            if encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            elif encoding == 'rgba8':
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            elif encoding == 'bgra8':
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            img = img[:, :, :3]

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # resize RGB
            rgb = cv2.resize(rgb, (224, 224))

            latest_image = rgb

            # ==================================================
            # sample trigger
            # ==================================================

            # ===== [MODIFIED] =====
            # 现在必须保证 depth 也存在
            if all(v is not None for v in [
                latest_image,
                latest_depth,  # 新增 depth 条件
                curr_obs_joint_state,
                curr_arm_target,
                curr_arm_r_feedback
            ]):

                arm_action_delta = np.clip(
                    curr_arm_target - curr_arm_r_feedback,
                    -delta_clip,
                    delta_clip
                )

                action_vec = np.zeros(8, dtype=np.float32)

                action_vec[0:7] = arm_action_delta
                action_vec[7] = curr_gripper_target

                dataset.append({

                    # ===== [MODIFIED] =====
                    # observation 中加入 depth
                    'obs': {
                        'image': latest_image.copy(),
                        'depth': latest_depth.copy(),   # 新增
                        'joint_state': curr_obs_joint_state.copy()
                    },

                    'action': action_vec
                })

    return dataset


if __name__ == "__main__":

    bags_root = "/home/nvidia/imitation_data_pipeline/bags"

    output_root = "/home/nvidia/imitation_data_pipeline/imitate_ws/output"

    os.makedirs(output_root, exist_ok=True)

    bag_dirs = [
        d for d in os.listdir(bags_root)
        if os.path.isdir(os.path.join(bags_root, d))
    ]

    for idx, bag_dir in enumerate(sorted(bag_dirs)):

        bag_dir_path = os.path.join(bags_root, bag_dir)

        db_files = [
            f for f in os.listdir(bag_dir_path)
            if f.endswith(".db3")
        ]

        if not db_files:
            continue

        bag_path = os.path.join(bag_dir_path, db_files[0])

        print(f"\n[{idx+1}/{len(bag_dirs)}] Processing {bag_dir}")

        data_samples = parse_bag_for_bc(bag_path)

        if data_samples:

            out_path = os.path.join(
                output_root,
                f"{bag_dir}.pkl"
            )

            with open(out_path, "wb") as f:
                pickle.dump(data_samples, f)

            print(f"Saved {len(data_samples)} samples to {out_path}")

        print_memory_usage("Post-Folder")

    print("\nPre-processing complete.")