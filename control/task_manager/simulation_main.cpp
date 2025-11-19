// simulation_main.cpp
#include "SimulationStateMachine.h"
#include <ros/ros.h>

int main(int argc, char** argv) {
    setlocale(LC_ALL,"");
    // 初始化ROS节点
    ros::init(argc, argv, "simulation_state_machine");
    ros::NodeHandle nh;
    
    // 创建仿真状态机实例
    SimulationStateMachine state_machine(nh);
    
    // 设置循环频率
    ros::Rate rate(10); // 10Hz
    
    ROS_INFO("仿真状态机节点启动");
    
    // 主循环
    while (ros::ok()) {
        // 执行状态机
        state_machine.execute();
        
        // 处理回调
        ros::spinOnce();
        
        // 如果任务完成，可以退出或等待
        if (state_machine.isTaskComplete()) {
            ROS_INFO("仿真任务完成，结果房间: %s", state_machine.getFoundRoom().c_str());
            // 可以选择退出或保持运行
            // break;
        }
        
        rate.sleep();
    }
    
    return 0;
}
