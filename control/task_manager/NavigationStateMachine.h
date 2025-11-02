#ifndef NAVIGATION_STATE_MACHINE_H
#define NAVIGATION_STATE_MACHINE_H

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/String.h>
#include <actionlib/client/simple_action_client.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <map>
#include <string>

enum class RobotState {
    INIT,                   // 初始化
    MOVE_TO_QR_ZONE,        // 移动到二维码区域
    WAIT_FOR_QR,            // 等待二维码识别
    MOVE_TO_PICK_ZONE,      // 移动到拣货区
    WAIT_FOR_OBJECT,        // 等待物体识别
    MOVE_TO_OBJECT,         // 移动到具体物体前
    OBJECT_ARRIVED,         // 已到达物体位置
    MOVE_TO_WAIT_ZONE,      // 移动到等待区
    WAIT_FOR_SIMULATION,    // 等待仿真完成
    MOVE_TO_TRAFFIC_ZONE,   //移动到路牌识别
    WAIT_FOR_TRAFFIC,       //等待路牌识别
    MOVE_TO_INTERSECTION,   //前往对应路口
    MOVE_TO_FINISH,         // 移动到对应终点
    TASK_COMPLETE,          // 任务完成
    ERROR                   // 错误状态
};

class NavigationStateMachine {
public:
    NavigationStateMachine(ros::NodeHandle& nh);
    void execute();
    
private:
    // 状态处理函数
    void handleInitState();
    void handleMoveToQRZone();
    void handleWaitForQR();
    void handleMoveToPickZone();
    void handleWaitForObject();
    void handleMoveToObject();
    void handleObjectArrived();
    void handleMoveToWaitZone();
    void handleWaitForSimulation();
    void handleMoveToTrafficZone();
    void handleWaitForTraffic();
    void handleMoveToIntersection();
    void handleMoveToFinish();
    void handleTaskComplete();
    void handleErrorState();

    // ActionLib回调函数
    void navDoneCallback(const actionlib::SimpleClientGoalState& state,
                        const move_base_msgs::MoveBaseResultConstPtr& result);
    void navActiveCallback();
    void navFeedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback);

    // 工具函数
    void speak(const std::string& text);
    void sendNavigationGoal(const std::string& point_name);
    void setState(RobotState new_state);
    geometry_msgs::PoseStamped createPose(double x, double y, double yaw);
    void loadNavigationPoints();
    std::string getTargetPosition(const std::string& task, const std::string& object);
    void updateCostCalculation(const std::string& object);
    
    // 回调函数
    void qrCallback(const std_msgs::String::ConstPtr& msg);
    void objectCallback(const std_msgs::String::ConstPtr& msg);
    void simulationCallback(const std_msgs::String::ConstPtr& msg);
    void trafficCallback(const std_msgs::String::ConstPtr& msg);
    
    // 成员变量
    RobotState current_state_;
    ros::NodeHandle nh_;
    
    // ActionLib客户端
    actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> action_client_;
    
    // ROS 通信
    ros::Publisher tts_publisher_;
    ros::Subscriber qr_sub_;
    ros::Subscriber object_sub_;
    ros::Subscriber simulation_sub_;
    ros::Subscriber traffic_sub_;
    
    // 任务数据
    std::string current_task_;
    std::string picked_object_;
    std::string simulation_result_;
    std::string traffic_result_;
    std::string current_goal_point_;  // 当前目标点名称
    double total_cost_;
    
    // 导航点配置
    std::map<std::string, geometry_msgs::PoseStamped> navigation_points_;

    // 状态标志
    struct TaskFlags{
        bool qr_received = false;
        bool object_received = false;
        bool object_picked = false; 
        bool simulation_received = false;
        bool traffic_received = false;
        bool intersection_goal_sent = false;
        bool navigation_in_progress = false;  // 导航是否进行中
        bool qr_goal_sent = false;           
        bool pick_goal_sent = false;        
        bool object_goal_sent = false;      
        bool wait_goal_sent = false;        
        bool traffic_goal_sent = false;   
        bool intersection_goal_sent_flag = false;  
        bool finish_goal_sent = false; 
    };
    TaskFlags task_flags_; 
};

#endif