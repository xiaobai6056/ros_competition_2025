#include <ros/ros.h>
#include <std_msgs/String.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    setlocale(LC_ALL,"");
    ros::init(argc, argv, "interactive_vision_simulator");
    ros::NodeHandle nh;
    
    // 初始化发布器
    ros::Publisher qr_pub = nh.advertise<std_msgs::String>("/demo/qr_result", 1);
    ros::Publisher object_pub = nh.advertise<std_msgs::String>("/demo/object_result", 1);
    ros::Publisher simulation_pub = nh.advertise<std_msgs::String>("/demo/simulation_result", 1);
    ros::Publisher traffic_pub = nh.advertise<std_msgs::String>("/demo/traffic_result", 1);
    
    ROS_INFO("交互式视觉模拟器启动");
    ROS_INFO("命令列表:");
    ROS_INFO("  1 - 发送二维码识别结果 (水果/蔬菜/饮料)");
    ROS_INFO("  2 - 发送物体识别结果 (格式: 类别:物体名)"); 
    ROS_INFO("  3 - 发送仿真任务结果 (房间号)");
    ROS_INFO("  4 - 发送路牌识别结果 (A/B)");
    ROS_INFO("  q - 退出程序");
    
    std::string input;
    
    while (ros::ok()) {
        std::cout << "\n请输入命令 (1/2/3/4/q): ";
        std::getline(std::cin, input);
        
        if (input == "q" || input == "Q") {
            ROS_INFO("退出程序");
            break;
        }
        else if (input == "1") {
            std::cout << "选择任务类型 (1-水果, 2-蔬菜, 3-饮料): ";
            std::getline(std::cin, input);
            
            std_msgs::String msg;
            if (input == "1") msg.data = "水果";
            else if (input == "2") msg.data = "蔬菜";
            else if (input == "3") msg.data = "饮料";
            else {
                ROS_WARN("无效选择，使用默认: 水果");
                msg.data = "水果";
            }
            
            qr_pub.publish(msg);
            ROS_INFO("已发送二维码识别结果: %s", msg.data.c_str());
        }
        else if (input == "2") {
            std::cout << "选择物体识别结果 (1-苹果, 2-香蕉, 3-西红柿, 4-可乐): ";
            std::getline(std::cin, input);
            
            std_msgs::String msg;
            if (input == "1") msg.data = "水果:苹果";
            else if (input == "2") msg.data = "水果:香蕉";
            else if (input == "3") msg.data = "蔬菜:西红柿";
            else if (input == "4") msg.data = "饮料:可乐";
            else {
                ROS_WARN("无效选择，使用默认: 水果:苹果");
                msg.data = "水果:苹果";
            }
            
            object_pub.publish(msg);
            ROS_INFO("已发送物体识别结果: %s", msg.data.c_str());
        }
        else if (input == "3") {
            std::cout << "输入仿真结果房间号 (A/B/C): ";
            std::getline(std::cin, input);
            
            std_msgs::String msg;
            if (input == "A" || input == "a") msg.data = "A";
            else if (input == "B" || input == "b") msg.data = "B";
            else if (input == "C" || input == "c") msg.data = "C";
            else {
                ROS_WARN("无效房间号，使用默认: A");
                msg.data = "A";
            }
            
            simulation_pub.publish(msg);
            ROS_INFO("已发送仿真任务结果: %s房间", msg.data.c_str());
        }
        else if (input == "4") {
            std::cout << "选择可通过的路口 (1-A路口, 2-B路口): ";
            std::getline(std::cin, input);
            
            std_msgs::String msg;
            if (input == "1") msg.data = "A";
            else if (input == "2") msg.data = "B";
            else {
                ROS_WARN("无效选择，使用默认: A");
                msg.data = "A";
            }
            
            traffic_pub.publish(msg);
            ROS_INFO("已发送路牌识别结果: 路口%s可通过", msg.data.c_str());
        }
        else {
            ROS_WARN("无效命令，请重新输入");
        }
        
        ros::spinOnce();
    }
    
    return 0;
}