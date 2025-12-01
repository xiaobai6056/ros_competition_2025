#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <iostream>
#include <string>
#include <sstream>

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

int main(int argc, char** argv)
{   
    setlocale(LC_ALL,"");
    ros::init(argc, argv, "simple_goal_sender");
    
    MoveBaseClient ac("move_base", true);
    
    // 等待action server
    if(!ac.waitForServer(ros::Duration(5.0))) {
        ROS_ERROR("Could not connect to move_base action server");
        return 1;
    }
    ROS_INFO("Connected to move_base action server");
    
    while(ros::ok()) {
        double x, y, yaw;
        
        std::cout << "\n=== 输入目标位置 ===" << std::endl;
        std::cout << "X 坐标 (默认 0): ";
        std::string x_str;
        std::getline(std::cin, x_str);
        x = x_str.empty() ? 0.0 : std::stod(x_str);
        
        std::cout << "Y 坐标 (默认 0): ";
        std::string y_str;
        std::getline(std::cin, y_str);
        y = y_str.empty() ? 0.0 : std::stod(y_str);
        
        std::cout << "朝向角度 (度, 默认 0): ";
        std::string yaw_str;
        std::getline(std::cin, yaw_str);
        yaw = yaw_str.empty() ? 0.0 : std::stod(yaw_str);
        
        // 将角度转换为四元数
        double yaw_rad = yaw * M_PI / 180.0;
        
        move_base_msgs::MoveBaseGoal goal;
        goal.target_pose.header.frame_id = "map";
        goal.target_pose.header.stamp = ros::Time::now();
        
        goal.target_pose.pose.position.x = x;
        goal.target_pose.pose.position.y = y;
        goal.target_pose.pose.position.z = 0.0;
        
        // 将偏航角转换为四元数
        goal.target_pose.pose.orientation.x = 0.0;
        goal.target_pose.pose.orientation.y = 0.0;
        goal.target_pose.pose.orientation.z = sin(yaw_rad / 2);
        goal.target_pose.pose.orientation.w = cos(yaw_rad / 2);
        
        ROS_INFO("发送目标: x=%.2f, y=%.2f, yaw=%.1f°", x, y, yaw);
        
        ac.sendGoal(goal);
        
        std::cout << "等待结果中... (输入 'c' 取消，或按回车继续等待)" << std::endl;
        
        // 简单的非阻塞等待
        bool finished = false;
        while(!finished && ros::ok()) {
            finished = ac.waitForResult(ros::Duration(0.5));
            
            // 检查是否有用户输入
            if(std::cin.rdbuf()->in_avail() > 0) {
                std::string cmd;
                std::getline(std::cin, cmd);
                if(cmd == "c" || cmd == "C") {
                    ac.cancelGoal();
                    ROS_INFO("目标已取消");
                    break;
                }
            }
        }
        
        if(finished) {
            if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED) {
                ROS_INFO("任务完成!");
            } else {
                ROS_WARN("任务失败: %s", ac.getState().toString().c_str());
            }
        }
        
        std::cout << "是否继续发送新目标? (y/n, 默认 y): ";
        std::string continue_str;
        std::getline(std::cin, continue_str);
        if(continue_str == "n" || continue_str == "N") {
            break;
        }
    }
    
    return 0;
}