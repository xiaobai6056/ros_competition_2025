#include "NavigationStateMachine.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

NavigationStateMachine::NavigationStateMachine(ros::NodeHandle& nh) 
    : nh_(nh), 
      current_state_(RobotState::INIT),
      action_client_("move_base", true),  // true表示自动启动线程
      total_cost_(0.0),
      task_flags_{}
{
    // 等待action server
    ROS_INFO("等待move_base action server...");
    if (action_client_.waitForServer(ros::Duration(5.0))) {
        ROS_INFO("move_base action server连接成功");
    } else {
        ROS_WARN("move_base action server连接超时，请检查move_base是否启动");
    }
    
    // 初始化发布器和订阅器
    tts_publisher_ = nh_.advertise<std_msgs::String>("/tts", 1);
    
    // 初始化订阅器 - 用于接收模拟的识别结果
    qr_sub_ = nh_.subscribe("/demo/qr_result", 1, &NavigationStateMachine::qrCallback, this);
    object_sub_ = nh_.subscribe("/demo/object_result", 1, &NavigationStateMachine::objectCallback, this);
    simulation_sub_ = nh_.subscribe("/demo/simulation_result", 1, &NavigationStateMachine::simulationCallback, this);
    traffic_sub_ = nh_.subscribe("/demo/traffic_result", 1, &NavigationStateMachine::trafficCallback, this);
    
    
    // 加载导航点
    loadNavigationPoints();
    
    ROS_INFO("Navigation State Machine with ActionLib Initialized");
}

void NavigationStateMachine::execute() {
    switch(current_state_) {
        case RobotState::INIT: handleInitState(); break;
        case RobotState::MOVE_TO_QR_ZONE: handleMoveToQRZone(); break;
        case RobotState::WAIT_FOR_QR: handleWaitForQR(); break;
        case RobotState::MOVE_TO_PICK_ZONE: handleMoveToPickZone(); break;
        case RobotState::WAIT_FOR_OBJECT: handleWaitForObject(); break;
        case RobotState::MOVE_TO_OBJECT: handleMoveToObject(); break;       
        case RobotState::OBJECT_ARRIVED: handleObjectArrived(); break;       
        case RobotState::MOVE_TO_WAIT_ZONE: handleMoveToWaitZone(); break;
        case RobotState::WAIT_FOR_SIMULATION: handleWaitForSimulation(); break;
        case RobotState::MOVE_TO_TRAFFIC_ZONE: handleMoveToTrafficZone(); break;
        case RobotState::WAIT_FOR_TRAFFIC: handleWaitForTraffic(); break;
        case RobotState::MOVE_TO_INTERSECTION: handleMoveToIntersection(); break;
        case RobotState::MOVE_TO_FINISH: handleMoveToFinish(); break;
        case RobotState::TASK_COMPLETE: handleTaskComplete(); break;
        case RobotState::ERROR: handleErrorState(); break;
    }
}

// ========== 状态处理函数 ==========

void NavigationStateMachine::handleInitState() {
    ROS_INFO("[INIT] 机器人初始化");
    speak("机器人准备就绪，开始执行任务");
    setState(RobotState::MOVE_TO_QR_ZONE); 
}

void NavigationStateMachine::handleMoveToQRZone() {
    if (!task_flags_.qr_goal_sent) {
        ROS_INFO("[MOVE_TO_QR_ZONE] 前往二维码区域");
        speak("正在前往二维码区域");
        sendNavigationGoal("qr_zone");
        task_flags_.qr_goal_sent = true;
    }
    ROS_INFO_THROTTLE(2, "[MOVE_TO_QR_ZONE] 等待导航完成...");
}

void NavigationStateMachine::handleWaitForQR() {
    if (task_flags_.qr_received) {
        ROS_INFO("[WAIT_FOR_QR] 收到二维码识别结果: %s", current_task_.c_str());
        speak("本次采购任务为" + current_task_);
        task_flags_.qr_received = false; // 重置标志
        setState(RobotState::MOVE_TO_PICK_ZONE);
    } else {
        ROS_INFO_THROTTLE(5, "[WAIT_FOR_QR] 等待二维码识别结果...");
    }
}

void NavigationStateMachine::handleMoveToPickZone() {
    if (!task_flags_.pick_goal_sent) {
        ROS_INFO("[MOVE_TO_PICK_ZONE] 前往拣货区");
        speak("正在前往拣货区");
        sendNavigationGoal("pick_zone");
        task_flags_.pick_goal_sent = true;
    }
    ROS_INFO_THROTTLE(2, "[MOVE_TO_PICK_ZONE] 等待导航完成...");
}


void NavigationStateMachine::handleWaitForObject() {
    if (task_flags_.object_received) {
        ROS_INFO("[WAIT_FOR_OBJECT] 收到物体识别结果: %s", picked_object_.c_str());
        speak("我已取到" + picked_object_);
        task_flags_.object_received = false; // 重置标志
        setState(RobotState::MOVE_TO_OBJECT);
    } else {
        ROS_INFO_THROTTLE(5, "[WAIT_FOR_OBJECT] 等待物体识别结果...");
    }
}

void NavigationStateMachine::handleMoveToObject() {
    if (!task_flags_.object_goal_sent) {
        ROS_INFO("[MOVE_TO_OBJECT] 移动到目标物体前");
        
        std::string target_position = getTargetPosition(current_task_, picked_object_);
        
        if (!target_position.empty() && navigation_points_.count(target_position)) {
            speak("正在前往" + picked_object_ + "位置");
            sendNavigationGoal(target_position);
            task_flags_.object_goal_sent = true;
        } else {
            ROS_ERROR("无法确定目标位置，任务: %s, 物体: %s", 
                     current_task_.c_str(), picked_object_.c_str());
            setState(RobotState::ERROR);
        }
    }
    ROS_INFO_THROTTLE(2, "[MOVE_TO_OBJECT] 等待导航完成...");
}

void NavigationStateMachine::handleObjectArrived() {
    ROS_INFO("[OBJECT_ARRIVED] 已到达物体位置: %s", picked_object_.c_str());
    speak("我已取到" + picked_object_);
    task_flags_.object_picked = true;
    updateCostCalculation(picked_object_);
    setState(RobotState::MOVE_TO_WAIT_ZONE);
}

void NavigationStateMachine::handleMoveToWaitZone() {
    if (!task_flags_.wait_goal_sent) {
        ROS_INFO("[MOVE_TO_WAIT_ZONE] 前往等待区");
        speak("正在前往等待区，等待仿真任务完成");
        sendNavigationGoal("wait_zone");
        task_flags_.wait_goal_sent = true;
    }
    ROS_INFO_THROTTLE(2, "[MOVE_TO_WAIT_ZONE] 等待导航完成...");
}

void NavigationStateMachine::handleWaitForSimulation() {
    if (task_flags_.simulation_received) {
        ROS_INFO("[WAIT_FOR_SIMULATION] 收到仿真结果: %s", simulation_result_.c_str());
        speak("仿真任务已完成，目标货物位于" + simulation_result_ + "房间");
        task_flags_.simulation_received = false; // 重置标志
        setState(RobotState::MOVE_TO_TRAFFIC_ZONE);
    } else {
        ROS_INFO_THROTTLE(5, "[WAIT_FOR_SIMULATION] 等待仿真任务完成...");
    }
}

void NavigationStateMachine::handleMoveToTrafficZone() {
    if (!task_flags_.traffic_goal_sent) {
        ROS_INFO("[MOVE_TO_TRAFFIC_ZONE] 前往路牌识别区");
        speak("正在前往路牌识别区");
        sendNavigationGoal("traffic_zone");
        task_flags_.traffic_goal_sent = true;
    }
    ROS_INFO_THROTTLE(2, "[MOVE_TO_TRAFFIC_ZONE] 等待导航完成...");
}

void NavigationStateMachine::handleWaitForTraffic() {
    if (task_flags_.traffic_received) {
        ROS_INFO("[WAIT_FOR_TRAFFIC] 收到路牌识别结果: %s", traffic_result_.c_str());
        speak("路口" + traffic_result_ + "可通过");
        task_flags_.traffic_received = false; // 重置标志
        setState(RobotState::MOVE_TO_INTERSECTION);
    } else {
        ROS_INFO_THROTTLE(5, "[WAIT_FOR_TRAFFIC] 等待路牌识别结果...");
    }
}

void NavigationStateMachine::handleMoveToIntersection() {
    if (!task_flags_.intersection_goal_sent_flag) {
        ROS_INFO("[MOVE_TO_INTERSECTION] 前往可通过的路口");
        
        std::string intersection_point;
        if (traffic_result_ == "A") {
            intersection_point = "intersection_A";
            ROS_INFO("A路口可通过，前往A路口入口");
        } else if (traffic_result_ == "B") {
            intersection_point = "intersection_B";
            ROS_INFO("B路口可通过，前往B路口入口");
        } else {
            ROS_ERROR("未知的路口识别结果: %s", traffic_result_.c_str());
            setState(RobotState::ERROR);
            return;
        }
        
        speak("正在前往可通过的路口");
        sendNavigationGoal(intersection_point);
        task_flags_.intersection_goal_sent_flag = true;
    }
    ROS_INFO_THROTTLE(2, "[MOVE_TO_INTERSECTION] 等待导航完成...");
}

void NavigationStateMachine::handleMoveToFinish() {
    if (!task_flags_.finish_goal_sent) {
        ROS_INFO("[MOVE_TO_FINISH] 从路口巡线前往终点");
        
        std::string finish_point;
        if (traffic_result_ == "A") {
            finish_point = "finish_zone_B";
            ROS_INFO("从A路口巡线前往右下方终点B");
        } else if (traffic_result_ == "B") {
            finish_point = "finish_zone_A";
            ROS_INFO("从B路口巡线前往左下方终点A");
        }
        
        speak("正在巡线前往终点");
        sendNavigationGoal(finish_point);
        task_flags_.finish_goal_sent = true;
    }
    ROS_INFO_THROTTLE(2, "[MOVE_TO_FINISH] 等待导航完成...");
}

void NavigationStateMachine::handleTaskComplete() {
    static bool task_complete_announced = false;
    
    if (!task_complete_announced) {
        ROS_INFO("[TASK_COMPLETE] 任务完成");
        speak("我已完成货物采购任务，本次采购货物为" + picked_object_ + "，总计花费15元，需找零5元");
        ROS_INFO("=== 演示任务完成 ===");
        task_complete_announced = true;
    }
    
    ROS_INFO_THROTTLE(5, "[TASK_COMPLETE] 任务已完成，等待程序结束...");
}

void NavigationStateMachine::handleErrorState() {
    ROS_ERROR("[ERROR] 进入错误状态");
    speak("系统出现错误，请检查");
    ros::Duration(2.0).sleep();
    setState(RobotState::INIT);
}


// ========== 回调函数 ==========

void NavigationStateMachine::qrCallback(const std_msgs::String::ConstPtr& msg) {
    current_task_ = msg->data;
    task_flags_.qr_received = true;
    ROS_INFO("收到二维码识别结果: %s", current_task_.c_str());
}

void NavigationStateMachine::objectCallback(const std_msgs::String::ConstPtr& msg) {
    // 假设消息格式为 "类别:物体名"，如 "水果:苹果"
    std::string full_data = msg->data;
    size_t pos = full_data.find(':');
    
    if (pos != std::string::npos) {
        std::string category = full_data.substr(0, pos);
        std::string object_name = full_data.substr(pos + 1);
        
        // 检查识别到的物体类别是否与任务匹配
        if (category == current_task_) {
            picked_object_ = object_name;
            task_flags_.object_received = true;
            ROS_INFO("收到匹配的物体识别结果: %s (任务: %s)", 
                     picked_object_.c_str(), current_task_.c_str());
        } else {
            ROS_WARN("物体类别不匹配: 识别到%s, 需要%s", 
                     category.c_str(), current_task_.c_str());
        }
    } else {
        // 如果没有分类信息，直接使用
        picked_object_ = full_data;
        task_flags_.object_received = true;
        ROS_INFO("收到物体识别结果: %s", picked_object_.c_str());
    }
}

void NavigationStateMachine::simulationCallback(const std_msgs::String::ConstPtr& msg) {
    simulation_result_ = msg->data;
    task_flags_.simulation_received = true;
    ROS_INFO("收到仿真结果: %s", simulation_result_.c_str());
}

void NavigationStateMachine::trafficCallback(const std_msgs::String::ConstPtr& msg) {
    traffic_result_ = msg->data;
    task_flags_.traffic_received = true;
    ROS_INFO("收到路牌识别结果: %s", traffic_result_.c_str());
}

// ActionLib完成回调
void NavigationStateMachine::navDoneCallback(const actionlib::SimpleClientGoalState& state,
                                            const move_base_msgs::MoveBaseResultConstPtr& result) {
    task_flags_.navigation_in_progress = false;

    ROS_INFO("导航完成回调 - 状态: %s, 目标点: %s", 
             state.toString().c_str(), current_goal_point_.c_str());

    task_flags_.qr_goal_sent = false;
    task_flags_.pick_goal_sent = false;
    task_flags_.object_goal_sent = false;
    task_flags_.wait_goal_sent = false;
    task_flags_.traffic_goal_sent = false;
    task_flags_.intersection_goal_sent_flag = false;
    task_flags_.finish_goal_sent = false;


    if (state == actionlib::SimpleClientGoalState::SUCCEEDED) {
        ROS_INFO("导航目标成功到达: %s", current_goal_point_.c_str());
        
        // 根据当前状态处理导航完成
        switch(current_state_) {
            case RobotState::MOVE_TO_QR_ZONE:
                setState(RobotState::WAIT_FOR_QR);
                break;
            case RobotState::MOVE_TO_PICK_ZONE:
                setState(RobotState::WAIT_FOR_OBJECT);
                break;
            case RobotState::MOVE_TO_OBJECT:
                setState(RobotState::OBJECT_ARRIVED);
                break;
            case RobotState::MOVE_TO_WAIT_ZONE:
                setState(RobotState::WAIT_FOR_SIMULATION);
                break;
            case RobotState::MOVE_TO_TRAFFIC_ZONE:
                setState(RobotState::WAIT_FOR_TRAFFIC);
                break;
            case RobotState::MOVE_TO_INTERSECTION:
                setState(RobotState::MOVE_TO_FINISH);
                break;
            case RobotState::MOVE_TO_FINISH:
                setState(RobotState::TASK_COMPLETE);
                break;
            default:
                ROS_WARN("导航完成但当前状态 %d 不需要处理", static_cast<int>(current_state_));
                break;
        }
    } else {
        ROS_ERROR("导航目标失败: %s - %s", 
                 state.toString().c_str(), state.getText().c_str());
        setState(RobotState::ERROR);
    }
}

// ActionLib激活回调
void NavigationStateMachine::navActiveCallback() {
    ROS_INFO("导航目标已激活: %s", current_goal_point_.c_str());
}

// ActionLib反馈回调
void NavigationStateMachine::navFeedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback) {
    // 可以在这里获取实时导航反馈
    ROS_INFO_THROTTLE(5, "导航反馈 - 当前位置: (%.2f, %.2f)", 
                     feedback->base_position.pose.position.x,
                     feedback->base_position.pose.position.y);
}
// ========== 工具函数 ==========

void NavigationStateMachine::speak(const std::string& text) {
    std_msgs::String msg;
    msg.data = text;
    tts_publisher_.publish(msg);
    ROS_INFO("语音播报: %s", text.c_str());
}

void NavigationStateMachine::sendNavigationGoal(const std::string& point_name) {
    auto it = navigation_points_.find(point_name);
    if (it != navigation_points_.end()) {
        // 取消之前的导航目标（如果有）
        if (task_flags_.navigation_in_progress) {
            action_client_.cancelAllGoals();
            ROS_INFO("取消之前的导航目标");
        }
        
        move_base_msgs::MoveBaseGoal goal;
        goal.target_pose = it->second;
        current_goal_point_ = point_name;
        
        // 发送目标并设置回调
        action_client_.sendGoal(goal,
            boost::bind(&NavigationStateMachine::navDoneCallback, this, _1, _2),
            boost::bind(&NavigationStateMachine::navActiveCallback, this),
            boost::bind(&NavigationStateMachine::navFeedbackCallback, this, _1));
        
        task_flags_.navigation_in_progress = true;
        ROS_INFO("发送导航目标: %s", point_name.c_str());
    } else {
        ROS_ERROR("未知的导航点: %s", point_name.c_str());
    }
}

std::string NavigationStateMachine::getTargetPosition(const std::string& task, const std::string& object) {
    ROS_INFO("[获取物体位置] 任务: %s, 物体: %s", task.c_str(), object.c_str());
    
    // 测试用硬编码 - 根据物体名称返回固定位置
    std::map<std::string, std::string> test_position_map = {
        {"苹果", "room_A"},
        {"香蕉", "room_B"},
        {"西红柿", "room_C"},
        {"可乐", "room_A"}
    };
    
    auto it = test_position_map.find(object);
    if (it != test_position_map.end()) {
        ROS_INFO("测试模式: 物体 %s 位于 %s", object.c_str(), it->second.c_str());
        return it->second;
    }
    
    ROS_WARN("测试模式: 未知物体 %s，使用默认位置 room_A", object.c_str());
    return "room_A";
}

void NavigationStateMachine::updateCostCalculation(const std::string& object) {
    ROS_INFO("[占位] 更新价格信息: %s", object.c_str());
    
    std::map<std::string, double> price_map = {
        {"苹果", 5.0},
        {"香蕉", 3.0},
        {"西红柿", 4.0},
        {"可乐", 3.0}
    };
    
    auto it = price_map.find(object);
    if (it != price_map.end()) {
        total_cost_ += it->second;
        ROS_INFO("[占位] 物品 %s 价格 %.1f 元，当前总价: %.1f 元", 
                object.c_str(), it->second, total_cost_);
    } else {
        total_cost_ += 5.0;
        ROS_WARN("[占位] 使用默认价格 5.0 元");
    }
}

void NavigationStateMachine::setState(RobotState new_state) {
    ROS_INFO("状态转换: %d -> %d", 
             static_cast<int>(current_state_), 
             static_cast<int>(new_state));
    
    // 在进入路口和终点状态时重置标志
    if (new_state == RobotState::MOVE_TO_INTERSECTION || 
        new_state == RobotState::MOVE_TO_FINISH) {
        task_flags_.intersection_goal_sent = false;
    }
    
    current_state_ = new_state;
}

void NavigationStateMachine::loadNavigationPoints() {
    navigation_points_["qr_zone"] = createPose(1.4, 1.1, 3.14);
    navigation_points_["pick_zone"] = createPose(1.75, 5.57, 1.57);
    navigation_points_["wait_zone"] = createPose(1.75, 5.57, 0.0);
    navigation_points_["traffic_zone"] = createPose(4.9, 6.4, 1.57); 
    navigation_points_["intersection_A"] = createPose(4.2, 4.3, -1.57);
    navigation_points_["intersection_B"] = createPose(7.3, 4.6, -1.57);
    navigation_points_["finish_zone_A"] = createPose(4.9, 0.4, -1.57);
    navigation_points_["finish_zone_B"] = createPose(6.5, 0.4, -1.57);

    //测试用
    navigation_points_["room_A"] = createPose(0.7, 5.5, 3.14);
    navigation_points_["room_B"] = createPose(1.6, 6.3, 1.57);
    navigation_points_["room_C"] = createPose(2.6, 5.5, 0.0);

    ROS_INFO("加载了 %zu 个导航点", navigation_points_.size());
}

geometry_msgs::PoseStamped NavigationStateMachine::createPose(double x, double y, double yaw) {
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