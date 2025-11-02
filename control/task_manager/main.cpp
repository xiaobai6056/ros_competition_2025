#include "NavigationStateMachine.h"
#include <ros/ros.h>

int main(int argc, char** argv) {
    setlocale(LC_ALL,"");
    ros::init(argc, argv, "navigation_state_machine");
    ros::NodeHandle nh("~");  // 私有命名空间
    
    try {
        NavigationStateMachine state_machine(nh);
        ros::Rate rate(10); // 10Hz
        
        ROS_INFO("Navigation State Machine started");
        
        while (ros::ok()) {
            state_machine.execute();
            ros::spinOnce();
            rate.sleep();
        }
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in main: %s", e.what());
        return 1;
    }
    
    return 0;
}