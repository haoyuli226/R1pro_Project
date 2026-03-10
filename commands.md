一、录包流程:
1.启动机器人本体程序:

```bash
cd ~/galaxea215/install/startup_config/share/startup_config/script

# VR Node
./robot_startup.sh  boot ../sessions.d/ATCStandard/R1PROVRTeleop.d/

2.启动 ROS2 Bag 录制：
ros2 bag record -o /home/nvidia/imitation_data_pipeline/bags/pick_cup_010 \
  /joint_states \
  /motion_target/target_joint_state_arm_right \
  /motion_target/target_position_gripper_right \
  /hdas/camera_head/rgb/image_rect_color \
  /tf \
  /tf_static \
  /hdas/camera_wrist_right/color/image_raw/compressed \
  /hdas/camera_head/depth/depth_registered \
  /hdas/feedback_arm_right \
  /hdas/feedback_gripper_right 


3.执行抓取操作（VR遥控）

4.操作完成后停止录制（Ctrl+C）

5.查看/解析 bag：
ros2 bag info demo_pick_cup_001
ros2 bag play demo_pick_cup_001

二、数据处理流程
# 1. 进入工程目录
cd ~/imitation_data_pipeline


# 2. 先 source ROS2
source /opt/ros/humble/setup.bash


# 3. 再激活 Python 虚拟环境
cd ~/imitation_data_pipeline
source venv/bin/activate



~/imitation_data_pipeline/start_bc.sh