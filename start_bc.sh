#!/bin/bash
# 文件名: start_bc.sh

# 基本环境
source ~/imitation_data_pipeline/venv/bin/activate
source /opt/ros/humble/setup.bash
source ~/imitation_data_pipeline/ros2_ws/install/setup.bash

# 运行节点
python3 ~/imitation_data_pipeline/ros2_ws/src/my_inference_package/my_inference_package/inference_node.py