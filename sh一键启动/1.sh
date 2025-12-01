#!/bin/bash

# 设置标题
echo "====================================="
echo "       视觉-导航耦合测试环境启动       "
echo "====================================="

# 初始化ROS环境
echo "初始化ROS环境..."
source /opt/ros/noetic/setup.bash
source /home/beswad/gazebo_ws1/devel/setup.bash
source /home/beswad/ros_competition/devel/setup.bash  # 添加你的工作空间

# 清理环境
echo "清理ROS环境..."
pkill -f "roslaunch"
pkill -f "roscore"
pkill -f "python3"
pkill -f "nav_client"  # 添加清理旧的导航客户端
sleep 2

# 启动roscore
echo "启动ROS Master..."
gnome-terminal --title="ROS Master" -- bash -c "
    roscore
    exec bash" &
sleep 3

# 启动Gazebo仿真环境
echo "启动Gazebo仿真环境..."
gnome-terminal --title="Gazebo" -- bash -c "
    roslaunch gazebo_pkg camera_test.launch
    exec bash" &

# 等待Gazebo启动
echo "等待Gazebo初始化..."
sleep 10

# 启动导航
echo "启动导航系统..."
gnome-terminal --title="Navigation" -- bash -c "
    roslaunch gazebo_nav navigation.launch
    exec bash" &

# 等待导航系统启动
echo "等待导航系统初始化..."
sleep 8


# 启动新的智能导航客户端
echo "启动智能导航评分客户端..."
gnome-terminal --title="Nav Client" --geometry=100x30 -- bash -c "
    source /home/beswad/ros_competition/devel/setup.bash
    rosrun test_pkg new_nav_client  # 假设你的节点叫new_nav_client
    exec bash" &

echo "====================================="
echo "       所有系统启动完成！             "
echo "====================================="
echo "请在新打开的终端中操作导航客户端"
echo "等待几秒钟让所有系统完全初始化..."