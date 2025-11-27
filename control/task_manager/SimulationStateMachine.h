// SimulationStateMachine.h
#ifndef SIMULATION_STATE_MACHINE_H
#define SIMULATION_STATE_MACHINE_H

#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <map>
#include <string>

enum class SimulationState {
    INIT,
    MOVE_TO_ROOM_A,
    WAITING_VISUAL_A,
    MOVE_TO_ROOM_B, 
    WAITING_VISUAL_B,
    MOVE_TO_ROOM_C,
    WAITING_VISUAL_C,
    OBJECT_FOUND,
    RETURN_TO_ORIGIN,
    ALL_ROOMS_CHECKED,
    ERROR
};

class SimulationStateMachine {
public:
    SimulationStateMachine(ros::NodeHandle& nh);
    void execute();
    bool isTaskComplete() const { return current_state_ == SimulationState::OBJECT_FOUND || 
                                        current_state_ == SimulationState::ALL_ROOMS_CHECKED; }
    std::string getFoundRoom() const { return found_room_; }
    std::string getTargetTask() const { return target_task_; }
    std::string getFoundObject() const { return found_object_; }

private:
    // 状态处理函数
    void handleInitState();
    void handleMoveToRoomA();
    void handleWaitingVisualA();
    void handleMoveToRoomB();
    void handleWaitingVisualB(); 
    void handleMoveToRoomC();
    void handleWaitingVisualC();
    void handleObjectFound();
    void handleReturnToOrigin();
    void handleAllRoomsChecked();
    void handleErrorState();

    // 工具函数
    void setState(SimulationState new_state);
    void sendNavigationGoal(const std::string& point_name);
    void speak(const std::string& text);
    bool callVisualService();
    void moveToNextRoom();
    void stopMoving();
    
    // 回调函数
    void navDoneCallback(const actionlib::SimpleClientGoalState& state,
                        const move_base_msgs::MoveBaseResultConstPtr& result);
    void navActiveCallback();
    void navFeedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback);
    void visualCallback(const std_msgs::String::ConstPtr& msg);  // 修正：使用std_msgs
    
    // 开始指令回调
    void startCallback(const std_msgs::String::ConstPtr& msg);
    
    // 发布结果函数
    void publishResult(const std::string& result);
    void publishFinalResult();
    
    // 智能停止相关
    void handleFixedPointNavigationStop(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback);
    void triggerStateTransition(const std::string& goal_name);
    float getYawFromPose(const geometry_msgs::Pose& pose);
    
    // 导航相关
    geometry_msgs::PoseStamped createPose(double x, double y, double yaw);
    void loadNavigationPoints();
    
    // 保存和返回原点功能
    void saveOriginalPose();
    bool getRobotPose(float& x, float& y, float& yaw);
    void resetStateMachine();

    ros::NodeHandle& nh_;
    SimulationState current_state_;
    ros::Time state_start_time_;
    
    // Action client
    actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> action_client_;
    
    // 发布器和订阅器
    ros::Publisher tts_publisher_;
    ros::Publisher cmd_vel_pub_;
    ros::Subscriber visual_sub_;
    
    // 开始指令订阅器和结果发布器
    ros::Subscriber start_sub_;
    ros::Publisher result_pub_;
    
    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // 服务客户端
    ros::ServiceClient visual_service_client_;
    
    // 导航点
    std::map<std::string, geometry_msgs::PoseStamped> navigation_points_;
    std::string current_goal_point_;
    
    // 任务相关
    std::string target_task_;
    std::string found_room_;
    std::string found_object_;
    std::string current_room_;
    
    // 视觉识别相关
    bool visual_service_called_;
    static constexpr int VISUAL_TIMEOUT = 15;
    
    // 房间检查状态
    bool room_a_checked_;
    bool room_b_checked_; 
    bool room_c_checked_;
    
    // 智能停止相关
    bool navigation_in_progress_;
    
    // 原点相关
    geometry_msgs::PoseStamped original_pose_;
    bool original_pose_saved_;
    
    // 智能停止阈值
    static constexpr float DISTANCE_THRESHOLD = 0.18f;
    static constexpr float YAW_THRESHOLD = 0.2f;
};

#endif
