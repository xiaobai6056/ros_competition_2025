#include "SimulationStateMachine.h"
#include <sstream>
#include <cmath>

// 常量定义
// 常量定义 - 确保只在头文件中声明，在cpp中定义
constexpr float SimulationStateMachine::DISTANCE_THRESHOLD;
constexpr float SimulationStateMachine::YAW_THRESHOLD;
constexpr int SimulationStateMachine::VISUAL_TIMEOUT;

SimulationStateMachine::SimulationStateMachine(ros::NodeHandle& nh) 
    : nh_(nh), 
      current_state_(SimulationState::INIT),
      action_client_("move_base", true),
      tf_listener_(tf_buffer_),
      visual_service_called_(false),
      room_a_checked_(false),
      room_b_checked_(false),
      room_c_checked_(false),
      navigation_in_progress_(false),
      target_task_(""),
      original_pose_saved_(false) {
    
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
    
    // 修正：使用正确的回调函数绑定
    visual_sub_ = nh_.subscribe("/simulation_object_detected", 1, 
                               &SimulationStateMachine::visualCallback, this);
    
    // 开始指令订阅器和结果发布器
    start_sub_ = nh_.subscribe("/simulation_start", 1, 
                              &SimulationStateMachine::startCallback, this);
    result_pub_ = nh_.advertise<std_msgs::String>("/simulation_result", 1);
    
    // 初始化服务客户端
    visual_service_client_ = nh_.serviceClient<std_srvs::Trigger>("/simulation_object_recognition");
    
    // 加载导航点
    loadNavigationPoints();
    
    ROS_INFO("仿真状态机初始化完成，等待开始指令...");
}

void SimulationStateMachine::startCallback(const std_msgs::String::ConstPtr& msg) {
    target_task_ = msg->data;
    ROS_INFO("收到仿真开始指令，任务类型: %s", target_task_.c_str());
    
    // 保存起始位置
    if (!original_pose_saved_) {
        saveOriginalPose();
    }
    
    // 如果状态机还没开始，就启动
    if (current_state_ == SimulationState::INIT) {
        setState(SimulationState::MOVE_TO_ROOM_A);
    }
}

void SimulationStateMachine::execute() {
    switch(current_state_) {
        case SimulationState::INIT: 
            handleInitState();
            break;
        case SimulationState::MOVE_TO_ROOM_A: handleMoveToRoomA(); break;
        case SimulationState::WAITING_VISUAL_A: handleWaitingVisualA(); break;
        case SimulationState::MOVE_TO_ROOM_B: handleMoveToRoomB(); break;
        case SimulationState::WAITING_VISUAL_B: handleWaitingVisualB(); break;
        case SimulationState::MOVE_TO_ROOM_C: handleMoveToRoomC(); break;
        case SimulationState::WAITING_VISUAL_C: handleWaitingVisualC(); break;
        case SimulationState::OBJECT_FOUND: handleObjectFound(); break;
        case SimulationState::RETURN_TO_ORIGIN: handleReturnToOrigin(); break;
        case SimulationState::ALL_ROOMS_CHECKED: handleAllRoomsChecked(); break;
        case SimulationState::ERROR: handleErrorState(); break;
    }
}

// ========== 状态处理函数 ==========

void SimulationStateMachine::handleInitState() {
    ROS_INFO_THROTTLE(5, "[SIM_INIT] 等待仿真开始指令...");
}

void SimulationStateMachine::handleMoveToRoomA() {
    if (!room_a_checked_) {
        ROS_INFO("[SIM_MOVE_TO_A] 前往A房间，任务类型: %s", target_task_.c_str());
        speak("正在前往A房间，寻找" + target_task_);
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
        wait_start_time = enter_time + ros::Duration(0.5);
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
        ROS_INFO("[SIM_MOVE_TO_B] 前往B房间，任务类型: %s", target_task_.c_str());
        speak("正在前往B房间，寻找" + target_task_);
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
        wait_start_time = enter_time + ros::Duration(0.5);
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
        ROS_INFO("[SIM_MOVE_TO_C] 前往C房间，任务类型: %s", target_task_.c_str());
        speak("正在前往C房间，寻找" + target_task_);
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
        wait_start_time = enter_time + ros::Duration(0.5);
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
        std::string result_message = "找到" + found_object_ + "在" + found_room_ + "房间";
        ROS_INFO("[SIM_OBJECT_FOUND] %s", result_message.c_str());
        speak("在" + found_room_ + "房间找到" + found_object_);
        
        announced = true;
        setState(SimulationState::RETURN_TO_ORIGIN);
    }
}

void SimulationStateMachine::handleReturnToOrigin() {
    static bool goal_sent = false;
    
    if (!goal_sent) {
        ROS_INFO("[SIM_RETURN_TO_ORIGIN] 返回起始位置");
        speak("正在返回起始位置");
        sendNavigationGoal("origin_point");
        goal_sent = true;
    }
    
    double time_in_state = (ros::Time::now() - state_start_time_).toSec();
    ROS_INFO_THROTTLE(2, "[SIM_RETURN_TO_ORIGIN] 返回原点中... 已耗时: %.1f 秒", time_in_state);
}

void SimulationStateMachine::handleAllRoomsChecked() {
    static bool announced = false;
    
    if (!announced) {
        std::string result_message = "未找到" + target_task_ + "类物品";
        ROS_WARN("[SIM_ALL_ROOMS_CHECKED] %s", result_message.c_str());
        speak("所有房间都未找到目标物品");
        
        announced = true;
        setState(SimulationState::RETURN_TO_ORIGIN);
    }
}

void SimulationStateMachine::handleErrorState() {
    static bool announced = false;
    
    if (!announced) {
        std::string result_message = "仿真任务执行失败";
        ROS_ERROR("[SIM_ERROR] %s", result_message.c_str());
        speak("仿真任务出现错误");
        
        publishResult(result_message);
        announced = true;
    }
}

// ========== 视觉服务调用 ==========

bool SimulationStateMachine::callVisualService() {
    std_srvs::Trigger srv;
    if (visual_service_client_.call(srv)) {
        if (srv.response.success) {
            std::string response_message = srv.response.message;
            ROS_INFO("仿真视觉服务返回: %s", response_message.c_str());
            
            if (response_message.find("WARN:") == 0) {
                std::string mismatched_object = response_message.substr(5);
                ROS_WARN("仿真任务在%s房间识别到不匹配物品: %s，继续检查", 
                         current_room_.c_str(), mismatched_object.c_str());
                return true;
                
            } else if (response_message == "NO_OBJECT_DETECTED") {
                ROS_WARN("仿真任务在%s房间未检测到任何物体，等待超时", current_room_.c_str());
                return false;
                
            } else if (response_message == "CONTINUE_DETECTING") {
                ROS_INFO("仿真任务在%s房间视觉系统正在检测中", current_room_.c_str());
                return false;
                
            } else {
                // 找到匹配的目标物品，存储纯净的物品名
                found_object_ = response_message;
                found_room_ = current_room_;
                setState(SimulationState::OBJECT_FOUND);
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

// ========== 智能停止核心函数 ==========

void SimulationStateMachine::navFeedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback) {
    ROS_INFO_THROTTLE(5, "仿真导航反馈 - 当前位置: (%.2f, %.2f)", 
                     feedback->base_position.pose.position.x,
                     feedback->base_position.pose.position.y);
    
    switch(current_state_) {
        case SimulationState::MOVE_TO_ROOM_A:
        case SimulationState::MOVE_TO_ROOM_B:
        case SimulationState::MOVE_TO_ROOM_C:
        case SimulationState::RETURN_TO_ORIGIN:
            handleFixedPointNavigationStop(feedback);
            break;
        default:
            break;
    }
}

void SimulationStateMachine::handleFixedPointNavigationStop(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback) {
    auto it = navigation_points_.find(current_goal_point_);
    if (it == navigation_points_.end()) {
        return;
    }
    
    geometry_msgs::Pose target_pose = it->second.pose;
    
    float dx = feedback->base_position.pose.position.x - target_pose.position.x;
    float dy = feedback->base_position.pose.position.y - target_pose.position.y;
    float distance = sqrt(dx*dx + dy*dy);
    
    float target_yaw = getYawFromPose(target_pose);
    float current_yaw = getYawFromPose(feedback->base_position.pose);
    float yaw_diff = fabs(current_yaw - target_yaw);
    if (yaw_diff > M_PI) yaw_diff = 2 * M_PI - yaw_diff;
    
    if (distance <= DISTANCE_THRESHOLD && yaw_diff <= YAW_THRESHOLD) {
        ROS_INFO("仿真到达 %s 位置！主动停止导航 (距离: %.3fm, 角度差: %.1f°)", 
                 current_goal_point_.c_str(), distance, yaw_diff * 180 / M_PI);
        action_client_.cancelAllGoals();
        stopMoving();
        
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
    } else if (goal_name == "origin_point") {
        publishFinalResult();
    } else {
        ROS_WARN("仿真的未知目标点 %s，无法触发状态转换", goal_name.c_str());
    }
}

// ========== 保存和返回原点功能 ==========

void SimulationStateMachine::saveOriginalPose() {
    float x, y, yaw;
    if (getRobotPose(x, y, yaw)) {
        original_pose_ = createPose(x, y, yaw);
        original_pose_saved_ = true;
        ROS_INFO("保存起始位置: (%.2f, %.2f, %.1f°)", x, y, yaw * 180 / M_PI);
        
        navigation_points_["origin_point"] = original_pose_;
    } else {
        ROS_WARN("无法获取机器人位姿，使用默认原点");
        original_pose_ = createPose(0.0, 0.0, 0.0);
        original_pose_saved_ = true;
        navigation_points_["origin_point"] = original_pose_;
    }
}

bool SimulationStateMachine::getRobotPose(float& x, float& y, float& yaw) {
    try {
        geometry_msgs::TransformStamped transform;
        transform = tf_buffer_.lookupTransform("map", "base_footprint", ros::Time(0), ros::Duration(0.1));
        
        x = transform.transform.translation.x;
        y = transform.transform.translation.y;
        
        tf2::Quaternion q(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        );
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw_temp;
        m.getRPY(roll, pitch, yaw_temp);
        yaw = yaw_temp;
        
        return true;
    }
    catch (tf2::TransformException &ex) {
        ROS_WARN_THROTTLE(5, "TF变换获取失败: %s", ex.what());
        return false;
    }
}

// ========== 结果发布逻辑 ==========

void SimulationStateMachine::publishFinalResult() {
    std::string result_message;
    
    if (current_state_ == SimulationState::OBJECT_FOUND) {
        // ✅ 返回纯净的物品名给B服务端
        result_message = found_object_;
        ROS_INFO("[SIM_FINAL_RESULT] 找到目标物品: %s，在%s房间", 
                 found_object_.c_str(), found_room_.c_str());
    } else if (current_state_ == SimulationState::ALL_ROOMS_CHECKED) {
        result_message = "未找到" + target_task_ + "类物品";
        ROS_WARN("[SIM_FINAL_RESULT] %s", result_message.c_str());
    } else {
        result_message = "仿真任务执行完成";
        ROS_INFO("[SIM_FINAL_RESULT] %s", result_message.c_str());
    }
    
    speak("仿真任务完成");
    
    // 发布最终结果给B服务端
    publishResult(result_message);
    
    // 重置状态机
    resetStateMachine();
}

void SimulationStateMachine::publishResult(const std::string& result) {
    std_msgs::String msg;
    msg.data = result;
    result_pub_.publish(msg);
    ROS_INFO("发布仿真结果给B服务端: %s", result.c_str());
}

void SimulationStateMachine::resetStateMachine() {
    room_a_checked_ = false;
    room_b_checked_ = false;
    room_c_checked_ = false;
    visual_service_called_ = false;
    navigation_in_progress_ = false;
    found_object_ = "";
    found_room_ = "";
    target_task_ = "";
    
    setState(SimulationState::INIT);
    ROS_INFO("仿真状态机已重置，等待下一次任务");
}

// ========== 导航回调函数 ==========

void SimulationStateMachine::navDoneCallback(const actionlib::SimpleClientGoalState& state,
                                            const move_base_msgs::MoveBaseResultConstPtr& result) {
    navigation_in_progress_ = false;

    ROS_INFO("仿真导航完成回调 - 状态: %s, 目标点: %s", 
             state.toString().c_str(), current_goal_point_.c_str());

    if (state == actionlib::SimpleClientGoalState::SUCCEEDED) {
        ROS_INFO("仿真导航目标成功到达: %s", current_goal_point_.c_str());
        
        if (current_goal_point_ == "origin_point") {
            triggerStateTransition("origin_point");
        } else {
            ROS_WARN("仿真导航完成，但智能停止未触发，手动触发状态转换");
            triggerStateTransition(current_goal_point_);
        }
        
    } else {
        if (state == actionlib::SimpleClientGoalState::PREEMPTED) {
            ROS_INFO("仿真导航被取消: %s", state.getText().c_str());
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

// ========== 视觉回调函数 ==========

void SimulationStateMachine::visualCallback(const std_msgs::String::ConstPtr& msg) {
    std::string detected_object = msg->data;
    
    if (detected_object.empty()) {
        return;
    }
    
    ROS_INFO("仿真任务收到视觉检测: %s", detected_object.c_str());
    
    // 如果检测到目标物品，直接处理
    // 注意：这里需要根据实际情况调整逻辑
    if (!target_task_.empty()) {
        // 可以添加任务匹配逻辑
        found_object_ = detected_object;
        found_room_ = current_room_;
        setState(SimulationState::OBJECT_FOUND);
    }
}
// ========== 工具函数 ==========

void SimulationStateMachine::setState(SimulationState new_state) {
    ROS_INFO("仿真状态转换: %d -> %d", 
             static_cast<int>(current_state_), 
             static_cast<int>(new_state));
    current_state_ = new_state;
    state_start_time_ = ros::Time::now();
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

// ========== 导航点配置 ==========

void SimulationStateMachine::loadNavigationPoints() {
    navigation_points_["room_A"] = createPose(3.85, 1.10, 1.57);
    navigation_points_["room_B"] = createPose(2.69, 0.868, 2.28);
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

