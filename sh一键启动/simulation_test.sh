#!/bin/bash

CONDA_PATH="/home/beswad/miniconda3"

# 设置标题
echo "====================================="
echo "       视觉-导航耦合测试环境启动       "
echo "====================================="

# 初始化ROS环境
echo "初始化ROS环境..."
source /opt/ros/noetic/setup.bash
source /home/beswad/gazebo/devel/setup.bash

# 清理环境
echo "清理ROS环境..."
pkill -f "roslaunch"
pkill -f "roscore"
pkill -f "python3"
sleep 3


# 启动Gazebo仿真环境
echo "启动Gazebo仿真环境..."
gnome-terminal -- bash -c "
    roslaunch gazebo_pkg race.launch
    exec bash" &

# 等待Gazebo启动
echo "等待Gazebo初始化..."
sleep 10

# 启动所有视觉节点
echo "启动视觉识别节点..."

gnome-terminal -- bash -c "
    source $CONDA_PATH/etc/profile.d/conda.sh;
    conda activate py38;
    cd /home/beswad/ros_competition/src/camera_pkg/scripts
    python3 yolo_final_node_debug.py
    exec bash" &

echo "等待视觉节点启动..."
sleep 8

# 启动导航
echo "启动导航系统..."
gnome-terminal -- bash -c "
    roslaunch gazebo_nav gazebo_nav.launch
    exec bash" &

sleep 3

gnome-terminal -- bash -c "
    cd /home/beswad/gazebo/src/gazebo_pkg/script
    python3 random_model_for_room.py
    exec bash" &

echo "等待随机节点"

echo "====================================="
echo "视觉-导航测试环境启动完成!"
echo "运行的视觉节点:"
echo "  - camera_input.py: 摄像头输入和转发"
echo "  - yolo_final_node.py: 物体识别"
echo "====================================="
echo "测试流程:"
echo "1. 机器人导航到三个房间"
echo "3. 前往正确板子识别物体"
echo "====================================="
