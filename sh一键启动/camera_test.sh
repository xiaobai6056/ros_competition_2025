#!/bin/bash

# 设置标题
echo "====================================="
echo "       视觉-导航耦合测试环境启动       "
echo "====================================="

# 初始化ROS环境
echo "初始化ROS环境..."
CONDA_PATH="/home/beswad/miniconda3"
source /opt/ros/noetic/setup.bash
source /home/beswad/gazebo_ws1/devel/setup.bash

# 清理环境
echo "清理ROS环境..."
pkill -f "roslaunch"
pkill -f "roscore"
pkill -f "python3"
sleep 2

# 启动roscore
echo "启动ROS Master..."
gnome-terminal -- bash -c "
    roscore
    exec bash" &
sleep 3

# 启动Gazebo仿真环境
echo "启动Gazebo仿真环境..."
gnome-terminal -- bash -c "
    source $CONDA_PATH/etc/profile.d/conda.sh;
    conda activate py38;
    export ROS_MASTER_URI=http://10.219.232.159:11311
    export ROS_IP=10.219.232.159
    roslaunch gazebo_pkg camera_test.launch
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
    python3 qrcode_detect_node_debug.py
    exec bash" &

sleep 3

gnome-terminal -- bash -c "
    source $CONDA_PATH/etc/profile.d/conda.sh;
    conda activate py38;
    cd /home/beswad/ros_competition/src/camera_pkg/scripts
    python3 yolo_final_node_debug.py
    exec bash" &
    
sleep 3

gnome-terminal -- bash -c "
    source $CONDA_PATH/etc/profile.d/conda.sh;
    conda activate py38;
    cd /home/beswad/ros_competition/src/camera_pkg/scripts
    python3 tl_detect_debug.py
    exec bash" &

sleep 3

echo "等待视觉节点启动..."
sleep 8

# 启动导航
echo "启动导航系统..."
gnome-terminal -- bash -c "
    roslaunch gazebo_nav navigation.launch
    exec bash" &

echo "====================================="
echo "视觉-导航测试环境启动完成!"
echo "运行的视觉节点:"
echo "  - camera_input.py: 摄像头输入和转发"
echo "  - qrcode_detect_node.py: 二维码识别" 
echo "  - yolo_final_node.py: 物体识别"
echo "====================================="
echo "测试流程:"
echo "1. 机器人导航到二维码区域"
echo "2. 扫描二维码获取任务指令"
echo "3. 前往拣货区识别物体"
echo "4. 完成导航任务"
echo "====================================="
