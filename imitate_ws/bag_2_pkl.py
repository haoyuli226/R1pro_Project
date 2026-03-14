import os
import pickle
import numpy as np
import psutil
import cv2

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TOPIC_IMAGE = '/hdas/camera_head/rgb/image_rect_color'
TOPIC_DEPTH = '/hdas/camera_head/depth/depth_registered'
TOPIC_JOINT_STATES = '/joint_states'    # 24维反馈: 右臂7 + 躯干4 + 左臂7 + 底盘6 (补0)
JOINT_NAMES = {
    'right_arm': [f'right_arm_joint{i}' for i in range(1, 8)],
    'torso': [f'torso_joint{i}' for i in range(1, 5)],
    'left_arm': [f'left_arm_joint{i}' for i in range(1, 8)],
    'wheel': [f'wheel_motor_joint{i}' for i in range(1, 4)] + [f'steer_motor_joint{i}' for i in range(1, 4)]
}
TOPIC_GRIPPER_FEEDBACK = '/hdas/feedback_gripper_right' # 夹爪反馈 [0, 100]
TOPIC_ARM_TARGET = '/motion_target/target_joint_state_arm_right' # 右臂目标 (7维)
TOPIC_GRIPPER_TARGET = '/motion_target/target_position_gripper_right' # 夹爪目标 [0, 100]



def print_memory_usage(step_name=""):
    mem = psutil.virtual_memory()
    print(f"[MEMORY] {step_name} | Used: {mem.used / 1024**2:.1f} MB")

def open_bag(bag_path):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
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

    curr_obs_joint_state = None  # 24维
    curr_arm_r_feedback = None     # 7维 (用于计算delta)
    curr_arm_target = None       # 7维
    curr_gripper_target = 0.0
    curr_gripper_feedback = 0.0
    latest_image = None
    latest_depth = None
    
    dataset = []

    while reader.has_next():
        topic, data, t = reader.read_next()
        msg_type = get_message(topic_types[topic])
        msg = deserialize_message(data, msg_type)

        if topic == TOPIC_JOINT_STATES:
            name_to_pos = dict(zip(msg.name, msg.position))
            
            # 右臂 (7维)
            arm_r = np.array([name_to_pos.get(n, 0.0) for n in JOINT_NAMES['right_arm']], dtype=np.float32)
            # 躯干 (4维)
            torso = np.array([name_to_pos.get(n, 0.0) for n in JOINT_NAMES['torso']], dtype=np.float32)
            # 左臂 (7维)
            arm_l = np.array([name_to_pos.get(n, 0.0) for n in JOINT_NAMES['left_arm']], dtype=np.float32)
            
            curr_arm_r_feedback = arm_r
            # 构造 25 维 Observation: [右7, 躯干4, 左7, 底盘6(补0), 夹爪反馈]
            # ========= 注意：单纯的抓取任务不需要底盘信息，暂时补0 =========
            obs_state = np.zeros(25, dtype=np.float32)
            obs_state[0:7] = arm_r
            obs_state[7:11] = torso
            obs_state[11:18] = arm_l
            obs_state[24] = curr_gripper_feedback
            curr_obs_joint_state = obs_state

        elif topic == TOPIC_ARM_TARGET:
            curr_arm_target = np.array(msg.position[:7], dtype=np.float32)

        elif topic == TOPIC_GRIPPER_TARGET:
            if len(msg.position) > 0:
                curr_gripper_target = msg.position[0]
        
        elif topic == TOPIC_GRIPPER_FEEDBACK:
            if len(msg.position) > 0:
                curr_gripper_feedback = msg.position[0]

        elif topic == TOPIC_DEPTH:
            depth_img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            latest_depth = depth_img

        elif topic == TOPIC_IMAGE:
            encoding = msg.encoding.lower()
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            # print(f"  [DEBUG] Raw image shape: {img.shape}, encoding: {encoding}")

            if encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif encoding == 'rgba8':
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif encoding == 'bgra8':
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # 如果本身就是 bgr8，则无需转换

            if img is not None:
                # 裁剪/缩放前确保是 3 通道 BGR
                img = img[:, :, :3] 
                latest_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if all(v is not None for v in [latest_image, curr_obs_joint_state, curr_arm_target, curr_gripper_target, curr_gripper_feedback, latest_depth]):
                arm_action_delta = np.clip(curr_arm_target - curr_arm_r_feedback, -delta_clip, delta_clip)
                
                # Action: 7关节增量 + 1夹爪绝对值
                action_vec = np.zeros(8, dtype=np.float32)
                action_vec[0:7] = arm_action_delta
                action_vec[7] = curr_gripper_target

                dataset.append({
                    'obs': {
                        'image': latest_image.copy(),
                        'depth': latest_depth.copy(),
                        'joint_state': curr_obs_joint_state.copy()
                    },
                    'action': action_vec
                })

    return dataset

if __name__ == "__main__":  
    # ------------- 注意路径设置 -------------

    bags_root = "/home/nvidia/imitation_data_pipeline/bags"
    output_root = "/home/nvidia/imitation_data_pipeline/imitate_ws/output"
    os.makedirs(output_root, exist_ok=True)

    bag_dirs = [d for d in os.listdir(bags_root) if os.path.isdir(os.path.join(bags_root, d))]

    for idx, bag_dir in enumerate(sorted(bag_dirs)):
        # 寻找 db3 文件
        bag_dir_path = os.path.join(bags_root, bag_dir)
        db_files = [f for f in os.listdir(bag_dir_path) if f.endswith(".db3")]
        if not db_files: continue
        
        bag_path = os.path.join(bag_dir_path, db_files[0])
        print(f"\n[{idx+1}/{len(bag_dirs)}] Processing {bag_dir}")
        
        data_samples = parse_bag_for_bc(bag_path)
        
        if data_samples:
            out_path = os.path.join(output_root, f"{bag_dir}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(data_samples, f)
            print(f"\nSaved {len(data_samples)} samples to {out_path}")
        
        print_memory_usage("Post-Folder")

    print("\nPre-processing complete.")