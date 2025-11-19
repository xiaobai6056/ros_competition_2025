#include "SimulationStateMachine.h"
#include <sstream>
#include <cmath>


SimulationStateMachine::SimulationStateMachine(ros::NodeHandle& nh) 
    : nh_(nh), 
      current_state_(SimulationState::INIT),
      action_client_("move_base", true),
      tf_listener_(tf_buffer_),
      visual_service_called_(false),
      room_a_checked_(false),
      room_b_checked_(false),
      room_c_checked_(false),
      navigation_in_progress_(false) {
    
    // 等待action server
    ROS_INFO("仿真状态机: 等待move_base action server...");
    if (action_client_.waitForServer(ros::Duration(5.0))) {
        ROS_INFO("仿真状态机: move_base action server连接成功");
    } else {
        ROS_WARN("仿真状态机: move_base action server连接超时");
    }
    
    // 初始化发布器和订阅器
    tts_publisher_ = nh_.advertise<std_msgs::String>("/tts", 1);
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    visual_sub_ = nh_.subscribe("/simulation_object_detected", 1, &SimulationStateMachine::visualCallback, this);
    
    // 初始化服务客户端
    visual_service_client_ = nh_.serviceClient<std_srvs::Trigger>("/simulation_object_recognition");
    
    // 加载导航点
    loadNavigationPoints();
    
    ROS_INFO("仿真状态机初始化完成，目标物品: %s", TARGET_OBJECT.c_str());
}

void SimulationStateMachine::execute() {
    switch(current_state_) {
        case SimulationState::INIT: handleInitState(); break;
        case SimulationState::MOVE_TO_ROOM_A: handleMoveToRoomA(); break;
        case SimulationState::WAITING_VISUAL_A: handleWaitingVisualA(); break;
        case SimulationState::MOVE_TO_ROOM_B: handleMoveToRoomB(); break;
        case SimulationState::WAITING_VISUAL_B: handleWaitingVisualB(); break;
        case SimulationState::MOVE_TO_ROOM_C: handleMoveToRoomC(); break;
        case SimulationState::WAITING_VISUAL_C: handleWaitingVisualC(); break;
        case SimulationState::OBJECT_FOUND: handleObjectFound(); break;
        case SimulationState::ALL_ROOMS_CHECKED: handleAllRoomsChecked(); break;
        case SimulationState::ERROR: handleErrorState(); break;
    }
}

// ========== 状态处理函数 ==========

void SimulationStateMachine::handleInitState() {
    ROS_INFO("[SIM_INIT] 仿真任务开始，目标物品: %s", TARGET_OBJECT.c_str());
    speak("仿真任务开始，寻找目标物品" + TARGET_OBJECT);
    
    // 重置检查状态
    room_a_checked_ = false;
    room_b_checked_ = false;
    room_c_checked_ = false;
    found_room_ = "";
    current_room_ = "";
    
    setState(SimulationState::MOVE_TO_ROOM_A);
}

void SimulationStateMachine::handleMoveToRoomA() {
    if (!room_a_checked_) {
        ROS_INFO("[SIM_MOVE_TO_A] 前往A房间");
        speak("正在前往A房间");
        sendNavigationGoal("room_A");
        room_a_checked_ = true;
        current_room_ = "A";
    }
}

void SimulationStateMachine::handleWaitingVisualA() {
    static bool first_entered = true;
    static ros::Time wait_start_time;
    static ros::Time enter_time;
    
    if (first_entered) {
        ROS_INFO("[SIM_WAITING_VISUAL_A] 到达A房间，开始视觉识别");
        enter_time = ros::Time::now();
        wait_start_time = enter_time + ros::Duration(1.5); // 等待1.5秒后再调用视觉服务
        visual_service_called_ = false;
        first_entered = false;
        return;
    }
    
    // 等待1.5秒后再调用视觉服务
    if (!visual_service_called_ && ros::Time::now() >= wait_start_time) {
        if (callVisualService()) {
            visual_service_called_ = true;
        }
        return;
    }
    
    // 检查超时（从进入状态开始计算）
    if ((ros::Time::now() - enter_time).toSec() > VISUAL_TIMEOUT) {
        ROS_WARN("[SIM_WAITING_VISUAL_A] A房间识别超时，前往B房间");
        speak("A房间未找到目标物品");
        first_entered = true;
        setState(SimulationState::MOVE_TO_ROOM_B);
        return;
    }
    
    if (!visual_service_called_) {
        ROS_INFO_THROTTLE(1, "[SIM_WAITING_VISUAL_A] 等待视觉稳定...");
    } else {
        ROS_INFO_THROTTLE(1, "[SIM_WAITING_VISUAL_A] 等待A房间视觉识别结果...");
    }
}


void SimulationStateMachine::handleMoveToRoomB() {
    if (!room_b_checked_) {
        ROS_INFO("[SIM_MOVE_TO_B] 前往B房间");
        speak("正在前往B房间");
        sendNavigationGoal("room_B");
        room_b_checked_ = true;
        current_room_ = "B";
    }
}

void SimulationStateMachine::handleWaitingVisualB() {
    static bool first_entered = true;
    static ros::Time wait_start_time;
    static ros::Time enter_time;
    
    if (first_entered) {
        ROS_INFO("[SIM_WAITING_VISUAL_B] 到达B房间，开始视觉识别");
        enter_time = ros::Time::now();
        wait_start_time = enter_time + ros::Duration(1.5); // 等待1.5秒后再调用视觉服务
        visual_service_called_ = false;
        first_entered = false;
        return;
    }
    
    if (!visual_service_called_ && ros::Time::now() >= wait_start_time) {
        if (callVisualService()) {
            visual_service_called_ = true;
        }
        return;
    }
    
    if ((ros::Time::now() - enter_time).toSec() > VISUAL_TIMEOUT) {
        ROS_WARN("[SIM_WAITING_VISUAL_B] B房间识别超时，前往C房间");
        speak("B房间未找到目标物品");
        first_entered = true;
        setState(SimulationState::MOVE_TO_ROOM_C);
        return;
    }
    
    if (!visual_service_called_) {
        ROS_INFO_THROTTLE(1, "[SIM_WAITING_VISUAL_B] 等待视觉稳定...");
    } else {
        ROS_INFO_THROTTLE(1, "[SIM_WAITING_VISUAL_B] 等待B房间视觉识别结果...");
    }
}


void SimulationStateMachine::handleMoveToRoomC() {
    if (!room_c_checked_) {
        ROS_INFO("[SIM_MOVE_TO_C] 前往C房间");
        speak("正在前往C房间");
        sendNavigationGoal("room_C");
        room_c_checked_ = true;
        current_room_ = "C";
    }
}

void SimulationStateMachine::handleWaitingVisualC() {
    static bool first_entered = true;
    static ros::Time wait_start_time;
    static ros::Time enter_time;
    
    if (first_entered) {
        ROS_INFO("[SIM_WAITING_VISUAL_C] 到达C房间，开始视觉识别");
        enter_time = ros::Time::now();
        wait_start_time = enter_time + ros::Duration(1.5); // 等待1.5秒后再调用视觉服务
        visual_service_called_ = false;
        first_entered = false;
        return;
    }
    
    if (!visual_service_called_ && ros::Time::now() >= wait_start_time) {
        if (callVisualService()) {
            visual_service_called_ = true;
        }
        return;
    }
    
    if ((ros::Time::now() - enter_time).toSec() > VISUAL_TIMEOUT) {
        ROS_WARN("[SIM_WAITING_VISUAL_C] C房间识别超时，所有房间检查完毕");
        speak("C房间未找到目标物品");
        first_entered = true;
        setState(SimulationState::ALL_ROOMS_CHECKED);
        return;
    }
    
    if (!visual_service_called_) {
        ROS_INFO_THROTTLE(1, "[SIM_WAITING_VISUAL_C] 等待视觉稳定...");
    } else {
        ROS_INFO_THROTTLE(1, "[SIM_WAITING_VISUAL_C] 等待C房间视觉识别结果...");
    }
}


void SimulationStateMachine::handleObjectFound() {
    static bool announced = false;
    
    if (!announced) {
        ROS_INFO("[SIM_OBJECT_FOUND] 在%s房间找到目标物品: %s", found_room_.c_str(), TARGET_OBJECT.c_str());
        speak("在" + found_room_ + "房间找到目标物品" + TARGET_OBJECT);
        announced = true;
    }
}

void SimulationStateMachine::handleAllRoomsChecked() {
    static bool announced = false;
    
    if (!announced) {
        ROS_WARN("[SIM_ALL_ROOMS_CHECKED] 所有房间检查完毕，未找到目标物品");
        speak("所有房间都未找到目标物品");
        announced = true;
    }
}

void SimulationStateMachine::handleErrorState() {
    ROS_ERROR("[SIM_ERROR] 仿真任务进入错误状态");
    speak("仿真任务出现错误");
}

// ========== 智能停止核心函数 ==========

void SimulationStateMachine::navFeedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback) {
    ROS_INFO_THROTTLE(5, "仿真导航反馈 - 当前位置: (%.2f, %.2f)", 
                     feedback->base_position.pose.position.x,
                     feedback->base_position.pose.position.y);
    
    // 智能停止：针对所有固定导航点
    switch(current_state_) {
        case SimulationState::MOVE_TO_ROOM_A:
        case SimulationState::MOVE_TO_ROOM_B:
        case SimulationState::MOVE_TO_ROOM_C:
            handleFixedPointNavigationStop(feedback);
            break;
        default:
            // 其他状态不需要智能停止
            break;
    }
}

void SimulationStateMachine::handleFixedPointNavigationStop(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback) {
    auto it = navigation_points_.find(current_goal_point_);
    if (it == navigation_points_.end()) {
        return;
    }
    
    geometry_msgs::Pose target_pose = it->second.pose;
    
    // 计算距离
    float dx = feedback->base_position.pose.position.x - target_pose.position.x;
    float dy = feedback->base_position.pose.position.y - target_pose.position.y;
    float distance = sqrt(dx*dx + dy*dy);
    
    // 计算角度差
    float target_yaw = getYawFromPose(target_pose);
    float current_yaw = getYawFromPose(feedback->base_position.pose);
    float yaw_diff = fabs(current_yaw - target_yaw);
    if (yaw_diff > M_PI) yaw_diff = 2 * M_PI - yaw_diff;
    
    // 检查是否满足容差
    if (distance <= DISTANCE_THRESHOLD && yaw_diff <= YAW_THRESHOLD) {
        ROS_INFO("仿真到达 %s 位置！主动停止导航 (距离: %.3fm, 角度差: %.1f°)", 
                 current_goal_point_.c_str(), distance, yaw_diff * 180 / M_PI);
        action_client_.cancelAllGoals();
        stopMoving();
        
        // 状态切换
        navigation_in_progress_ = false;
        triggerStateTransition(current_goal_point_);
    }
}

void SimulationStateMachine::triggerStateTransition(const std::string& goal_name) {
    if (goal_name == "room_A") {
        setState(SimulationState::WAITING_VISUAL_A);
    } else if (goal_name == "room_B") {
        setState(SimulationState::WAITING_VISUAL_B);
    } else if (goal_name == "room_C") {
        setState(SimulationState::WAITING_VISUAL_C);
    } else {
        ROS_WARN("仿真的未知目标点 %s，无法触发状态转换", goal_name.c_str());
    }
}

// ========== 工具函数 ==========

void SimulationStateMachine::setState(SimulationState new_state) {
    ROS_INFO("仿真状态转换: %d -> %d", 
             static_cast<int>(current_state_), 
             static_cast<int>(new_state));
    current_state_ = new_state;
}

void SimulationStateMachine::sendNavigationGoal(const std::string& point_name) {
    auto it = navigation_points_.find(point_name);
    if (it != navigation_points_.end()) {
        if (navigation_in_progress_) {
            action_client_.cancelAllGoals();
            ROS_INFO("仿真取消之前的导航目标");
        }
        
        move_base_msgs::MoveBaseGoal goal;
        goal.target_pose = it->second;
        current_goal_point_ = point_name;
        
        action_client_.sendGoal(goal,
            boost::bind(&SimulationStateMachine::navDoneCallback, this, _1, _2),
            boost::bind(&SimulationStateMachine::navActiveCallback, this),
            boost::bind(&SimulationStateMachine::navFeedbackCallback, this, _1));
        
        navigation_in_progress_ = true;
        ROS_INFO("仿真发送导航目标: %s", point_name.c_str());
        
    } else {
        ROS_ERROR("仿真的未知导航点: %s", point_name.c_str());
    }
}

void SimulationStateMachine::stopMoving() {
    geometry_msgs::Twist stop_twist;
    cmd_vel_pub_.publish(stop_twist);
    ROS_INFO("仿真停止移动");
}

void SimulationStateMachine::speak(const std::string& text) {
    std_msgs::String msg;
    msg.data = text;
    tts_publisher_.publish(msg);
    ROS_INFO("仿真语音播报: %s", text.c_str());
}

bool SimulationStateMachine::callVisualService() {
    std_srvs::Trigger srv;
    if (visual_service_client_.call(srv)) {
        if (srv.response.success) {
            std::string detected_object = srv.response.message;
            ROS_INFO("仿真视觉服务返回: %s", detected_object.c_str());
            
            if (detected_object == TARGET_OBJECT) {
                // 找到目标物品
                found_room_ = current_room_;
                setState(SimulationState::OBJECT_FOUND);
                return true;
            } else if (detected_object == "NO_OBJECT_DETECTED") {
                // 关键修复：没有检测到物体，等待超时后前往下一个房间
                ROS_WARN("仿真任务在%s房间未检测到任何物体，等待超时", current_room_.c_str());
                // 不立即跳转，让超时机制处理
                return false;
            } else {
                // 识别到明确的非目标物品，立即前往下一个房间
                ROS_WARN("仿真任务在%s房间识别到非目标物品: %s，立即前往下一个房间", 
                         current_room_.c_str(), detected_object.c_str());
                speak(current_room_ + "房间找到非目标物品" + detected_object);
                moveToNextRoom();
                return true;
            }
        } else {
            ROS_WARN("仿真视觉识别失败: %s", srv.response.message.c_str());
            return false;
        }
    } else {
        ROS_ERROR("仿真任务无法调用视觉服务");
        return false;
    }
}

void SimulationStateMachine::moveToNextRoom() {
    if (!room_b_checked_) {
        setState(SimulationState::MOVE_TO_ROOM_B);
    } else if (!room_c_checked_) {
        setState(SimulationState::MOVE_TO_ROOM_C);
    } else {
        setState(SimulationState::ALL_ROOMS_CHECKED);
    }
}

// ========== 回调函数 ==========

void SimulationStateMachine::navDoneCallback(const actionlib::SimpleClientGoalState& state,
                                            const move_base_msgs::MoveBaseResultConstPtr& result) {
    navigation_in_progress_ = false;

    ROS_INFO("仿真导航完成回调 - 状态: %s, 目标点: %s", 
             state.toString().c_str(), current_goal_point_.c_str());

    // 如果当前状态已经不是导航状态，忽略这个回调
    if (current_state_ != SimulationState::MOVE_TO_ROOM_A &&
        current_state_ != SimulationState::MOVE_TO_ROOM_B &&
        current_state_ != SimulationState::MOVE_TO_ROOM_C) {
        ROS_INFO("仿真忽略导航回调，当前状态 %d 不是导航状态", static_cast<int>(current_state_));
        return;
    }

    if (state == actionlib::SimpleClientGoalState::SUCCEEDED) {
        ROS_INFO("仿真导航目标成功到达: %s", current_goal_point_.c_str());
        
        // 正常情况下，智能停止应该已经处理了状态转换
        // 如果走到这里，说明智能停止没触发
        ROS_WARN("仿真导航完成，但智能停止未触发，手动触发状态转换");
        triggerStateTransition(current_goal_point_);
        
    } else {
        // 特别处理被取消的情况
        if (state == actionlib::SimpleClientGoalState::PREEMPTED) {
            ROS_INFO("仿真导航被取消: %s", state.getText().c_str());
            // 如果是被智能停止取消，这是正常行为
            return;
        }
        
        ROS_ERROR("仿真导航目标失败: %s - %s", 
                 state.toString().c_str(), state.getText().c_str());
        setState(SimulationState::ERROR);
    }
}

void SimulationStateMachine::navActiveCallback() {
    ROS_INFO("仿真任务导航目标已激活: %s", current_goal_point_.c_str());
}

void SimulationStateMachine::visualCallback(const std_msgs::String::ConstPtr& msg) {
    std::string detected_object = msg->data;
    
    if (detected_object.empty()) {
        return;
    }
    
    ROS_INFO("仿真任务收到视觉检测: %s", detected_object.c_str());
    
    // 如果检测到目标物品，直接处理
    if (detected_object == TARGET_OBJECT) {
        found_room_ = current_room_;
        setState(SimulationState::OBJECT_FOUND);
    }
    // 注意：非目标物品的处理在服务调用中已经处理
}

// ========== 导航点配置 ==========

void SimulationStateMachine::loadNavigationPoints() {
    // 使用与主任务相同的房间坐标
    navigation_points_["room_A"] = createPose(3.6, 0.96, 1.57);
    navigation_points_["room_B"] = createPose(2.05, 1.58, 1.57);
    navigation_points_["room_C"] = createPose(0.51, 1.02, 1.57);
    
    ROS_INFO("仿真任务加载了 %zu 个导航点", navigation_points_.size());
}

geometry_msgs::PoseStamped SimulationStateMachine::createPose(double x, double y, double yaw) {
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "map";
    pose.header.stamp = ros::Time::now();
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.position.z = 0.0;
    
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    pose.pose.orientation = tf2::toMsg(q);
    
    return pose;
}

float SimulationStateMachine::getYawFromPose(const geometry_msgs::Pose& pose) {
    tf2::Quaternion q(
        pose.orientation.x,
        pose.orientation.y, 
        pose.orientation.z,
        pose.orientation.w
    );
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    return yaw;
}
