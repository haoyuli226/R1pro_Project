import os
import pickle
import numpy as np
import psutil
import cv2
from scipy.spatial.transform import Rotation as R

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# ---------- 话题配置 ----------
TOPIC_IMAGE = '/hdas/camera_head/rgb/image_rect_color'
TOPIC_DEPTH = '/hdas/camera_head/depth/depth_registered'

# 状态反馈话题
TOPIC_ARM_FEEDBACK_POSE = '/motion_control/pose_ee_arm_right'  # 右臂位姿反馈 (PoseStamped)
TOPIC_GRIPPER_FEEDBACK = '/hdas/feedback_gripper_right' # 夹爪位置反馈

# 控制目标话题 (用于计算 Action)
TOPIC_ARM_TARGET_POSE = '/motion_target/target_pose_arm_right' # 右臂目标位姿 (PoseStamped)
TOPIC_GRIPPER_TARGET = '/motion_target/target_position_gripper_right' # 夹爪目标位置

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

def pose_to_array(pose_msg):
    """将 Pose 消息转换为 7 维 numpy 数组 [x, y, z, qx, qy, qz, qw]"""
    # scipy 的 Rotation 默认接受的四元数顺序就是 [x, y, z, w]
    return np.array([
        pose_msg.position.x,
        pose_msg.position.y,
        pose_msg.position.z,
        pose_msg.orientation.x,
        pose_msg.orientation.y,
        pose_msg.orientation.z,
        pose_msg.orientation.w
    ], dtype=np.float32)

def clip_vector_norm(vec, max_norm):
    """等比例限制向量的模长，避免改变原始运动方向"""
    norm = np.linalg.norm(vec)
    if norm > max_norm:
        vec = vec * (max_norm / norm)
    return vec

def parse_bag_for_bc(bag_path, max_pos_step=0.05, max_rot_step=0.1):
    """
    max_pos_step: 单次平移最大步长 (单位: 米)
    max_rot_step: 单次旋转最大步长 (单位: 弧度)
    """
    reader = open_bag(bag_path)
    topic_types = get_topic_types(reader)

    # 缓存变量
    curr_arm_pose_feedback = None  
    curr_arm_pose_target = None    
    curr_gripper_feedback = None   
    curr_gripper_target = None     
    latest_image = None
    latest_depth = None
    
    dataset = []

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in topic_types: continue
        
        msg_type = get_message(topic_types[topic])
        msg = deserialize_message(data, msg_type)

        if topic == TOPIC_ARM_FEEDBACK_POSE:
            curr_arm_pose_feedback = pose_to_array(msg.pose)
        elif topic == TOPIC_GRIPPER_FEEDBACK:
            if len(msg.position) > 0:
                curr_gripper_feedback = msg.position[0]
        elif topic == TOPIC_ARM_TARGET_POSE:
            curr_arm_pose_target = pose_to_array(msg.pose)
        elif topic == TOPIC_GRIPPER_TARGET:
            if len(msg.position) > 0:
                curr_gripper_target = msg.position[0]
        elif topic == TOPIC_DEPTH:
            depth_img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            latest_depth = depth_img.copy()
        elif topic == TOPIC_IMAGE:
            encoding = msg.encoding.lower()
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

            if 'rgb' in encoding:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif 'rgba' in encoding:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            if img is not None:
                img = img[:, :, :3]
                latest_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # --- 数据同步与特征构造 ---
            ready_list = [
                latest_image, latest_depth, 
                curr_arm_pose_feedback, curr_arm_pose_target, 
                curr_gripper_feedback, curr_gripper_target
            ]

            if all(v is not None for v in ready_list):
                # 1. 构造 Observation (8维: 3位置 + 4四元数 + 1夹爪)
                obs_state = np.zeros(8, dtype=np.float32)
                obs_state[0:7] = curr_arm_pose_feedback
                obs_state[7] = curr_gripper_feedback

                # 2. 计算平移增量并限制模长
                pos_curr = curr_arm_pose_feedback[0:3]
                pos_target = curr_arm_pose_target[0:3]
                pos_delta = pos_target - pos_curr
                pos_delta = clip_vector_norm(pos_delta, max_pos_step)

                # 3. 计算真实的旋转增量四元数 (R_delta = R_target * R_curr_inv)
                quat_curr = curr_arm_pose_feedback[3:7]
                quat_target = curr_arm_pose_target[3:7]
                
                rot_curr = R.from_quat(quat_curr)
                rot_target = R.from_quat(quat_target)
                rot_delta = rot_target * rot_curr.inv()
                
                # 借助旋转向量进行角度限制 (因为四元数不能直接切)
                rotvec_delta = rot_delta.as_rotvec()
                rotvec_norm = np.linalg.norm(rotvec_delta)
                if rotvec_norm > max_rot_step:
                    rotvec_delta = rotvec_delta * (max_rot_step / rotvec_norm)
                    rot_delta = R.from_rotvec(rotvec_delta) # 重建限制后的旋转
                
                # 将限制角度后的旋转无损转回 4维四元数
                quat_delta = rot_delta.as_quat()

                # 4. 构造 Action (8维: 3平移增量 + 4四元数增量 + 1夹爪绝对目标)
                action_vec = np.zeros(8, dtype=np.float32)
                action_vec[0:3] = pos_delta
                action_vec[3:7] = quat_delta
                action_vec[7] = curr_gripper_target

                dataset.append({
                    'obs': {
                        'image': latest_image.copy(),
                        'depth': latest_depth.copy(),
                        'state': obs_state
                    },
                    'action': action_vec
                })

    return dataset

if __name__ == "__main__":  
    bags_root = "/home/nvidia/imitation_data_pipeline/test_bag"
    output_root = "/home/nvidia/imitation_data_pipeline/imitate_ws/output"
    os.makedirs(output_root, exist_ok=True)

    bag_dirs = [d for d in os.listdir(bags_root) if os.path.isdir(os.path.join(bags_root, d))]

    for idx, bag_dir in enumerate(sorted(bag_dirs)):
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
            print(f"Saved {len(data_samples)} samples to {out_path}")
        
        print_memory_usage("Post-Folder")

    print("\nPre-processing complete.")