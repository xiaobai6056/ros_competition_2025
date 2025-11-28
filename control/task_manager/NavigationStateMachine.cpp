#include "NavigationStateMachine.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <cmath>
#include <limits>
#include <sstream>
#include <std_srvs/Trigger.h>
#include <std_msgs/String.h>
#include <iomanip>

// 常量定义（与头文件保持一致）
constexpr float NavigationStateMachine::TARGET_OBSTACLE_DISTANCE;
constexpr float NavigationStateMachine::DEFAULT_SAFE_DISTANCE;
constexpr float NavigationStateMachine::EXTENDED_SAFE_DISTANCE;
constexpr int NavigationStateMachine::LASER_SCAN_TIMEOUT;
constexpr int NavigationStateMachine::VISUAL_RECOGNITION_TIMEOUT;
constexpr int NavigationStateMachine::SERVICE_RETRY_COUNT;

// PCA检测参数
struct PCAParams {
    // 聚类参数
    float max_distance_jump = 0.1f;
    float min_valid_range = 0.1f;
    float max_valid_range = 4.0f;
    int min_cluster_size = 10;
    int max_cluster_size = 100;
    
    // 过滤参数
    float min_board_length = 0.4f;
    float max_board_length = 0.6f;
    float max_angular_width = 0.4f;
    float duplicate_distance = 0.3f;
    
    // PCA参数
    float min_pca_confidence = 0.6f;
    float max_distance_std = 0.2f;
    
    // 地理约束参数
    float valid_min_x = -0.20f;
    float valid_max_x = 3.58f;
    float valid_min_y = 2.94f;
    float valid_max_y = 7.50f;
} pca_params_;


NavigationStateMachine::NavigationStateMachine(ros::NodeHandle& nh) 
    : nh_(nh), 
      current_state_(RobotState::INIT),
      tf_listener_(tf_buffer_),
      action_client_("move_base", true),
      total_cost_(0.0),
      obstacle_distance_(std::numeric_limits<float>::max()),
      obstacle_detected_(false),
      task_flags_{},
      scan_start_time_(ros::Time::now()),
      scan_robot_x_(0.0f),
      scan_robot_y_(0.0f),
      scan_robot_yaw_(0.0f),
      clusters_calculated_(false),
      object_detected_during_scan_(false),
      detected_object_name_(""),
      costmap_updated_(false),
      current_target_cluster_(-1),
      clusters_detected_(false),
      moving_to_cluster_(false),
      qr_service_called_(false),
      object_service_called_(false),
      current_waypoint_index_(0),
      following_waypoint_sequence_(false),
      waypoint_switch_distance_(0.8f),
      // 新增三状态相关变量
      rotation_scan_complete_(false),
      pca_calculation_complete_(false),
      cached_laser_scans_(),
      max_cached_scans_(5),
      rotation_target_angle_(M_PI), // 180度
      current_rotated_angle_(0.0f)
{
    // 初始化状态时间统计
    state_start_time_ = ros::Time::now();
    state_durations_.clear();
    
    // 等待action server
    ROS_INFO("等待move_base action server...");
    if (action_client_.waitForServer(ros::Duration(5.0))) {
        ROS_INFO("move_base action server连接成功");
    } else {
        ROS_WARN("move_base action server连接超时，请检查move_base是否启动");
    }
    
    // 初始化发布器
    tts_publisher_ = nh_.advertise<std_msgs::String>("/tts", 1);
    task_pub_ = nh_.advertise<std_msgs::String>("/current_task", 1);
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    
    // 初始化订阅器
    simulation_sub_ = nh_.subscribe("/demo/simulation_result", 1, &NavigationStateMachine::simulationCallback, this);
    traffic_sub_ = nh_.subscribe("/demo/traffic_result", 1, &NavigationStateMachine::trafficCallback, this);
    laser_sub_ = nh_.subscribe("/scan", 1, &NavigationStateMachine::laserCallback, this);
    object_detected_sub_ = nh_.subscribe("/object_detected", 1, &NavigationStateMachine::objectDetectedCallback, this);
    
    // 初始化服务客户端
    qr_service_client_ = nh_.serviceClient<std_srvs::Trigger>("/qr_recognition");
    object_service_client_ = nh_.serviceClient<std_srvs::Trigger>("/object_recognition");
    simulation_service_client_ = nh_.serviceClient<service::Service>("/task");
    
    // 加载导航点
    loadNavigationPoints();
    
    costmap_sub_ = nh_.subscribe("/move_base/global_costmap/costmap", 1, 
                                    &NavigationStateMachine::costmapCallback, this);
    costmap_updated_ = false;

    ROS_INFO("导航状态机初始化完成 - 三状态PCA识别板检测");
}

void NavigationStateMachine::execute() {
    // 状态时间统计
    static ros::Time last_execute_time = ros::Time::now();
    ros::Time current_time = ros::Time::now();
    double time_in_state = (current_time - state_start_time_).toSec();
    
    ROS_INFO_THROTTLE(10, "[状态时间统计] 当前状态 %d 已持续: %.1f 秒", 
                     static_cast<int>(current_state_), time_in_state);
    
    switch(current_state_) {
        case RobotState::INIT: handleInitState(); break;
        case RobotState::MOVE_TO_QR_ZONE: handleMoveToQRZone(); break;
        case RobotState::WAITING_QR_SERVICE: handleWaitingQRService(); break;
        case RobotState::MOVE_TO_PICK_ZONE: handleMoveToPickZone(); break;
        
        // 三状态PCA检测
        case RobotState::ROTATION_SCAN: handleRotationScan(); break;
        case RobotState::PCA_CALCULATION: handlePCACalculation(); break;
        case RobotState::CLUSTER_SELECTION: handleClusterSelection(); break;
        
        case RobotState::NAVIGATING_TO_BOARD: handleNavigatingToBoard(); break;
        case RobotState::WAITING_VISUAL: handleWaitingVisual(); break;
        case RobotState::OBJECT_CONFIRMED: handleObjectConfirmed(); break;
        case RobotState::MOVE_TO_WAIT_ZONE: handleMoveToWaitZone(); break;
        case RobotState::WAITING_SIMULATION: handleWaitingSimulation(); break;
        case RobotState::MOVE_TO_TRAFFIC_ZONE: handleMoveToTrafficZone(); break;
        case RobotState::WAITING_TRAFFIC: handleWaitingTraffic(); break;
        case RobotState::NAVIGATE_TO_FINISH: handleNavigateToFinish(); break;
        case RobotState::TASK_COMPLETE: handleTaskComplete(); break;
        case RobotState::ERROR: handleErrorState(); break;
    }
    
    last_execute_time = current_time;
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
    
    // 时间统计
    double time_in_state = (ros::Time::now() - state_start_time_).toSec();
    ROS_INFO_THROTTLE(5, "[MOVE_TO_QR_ZONE] 已耗时: %.1f 秒", time_in_state);
}

void NavigationStateMachine::handleWaitingQRService() {
    if (!qr_service_called_) {
        ROS_INFO("[WAITING_QR_SERVICE] 调用二维码识别服务");
        ros::Duration(0.5).sleep();
        if (callQRService()) {
            qr_service_called_ = true;
            service_call_time_ = ros::Time::now();
        } else {
            ROS_WARN("二维码服务调用失败，0.1秒后重试");
            ros::Duration(0.1).sleep();
        }
    } else {
        double time_in_state = (ros::Time::now() - state_start_time_).toSec();
        ROS_INFO_THROTTLE(2, "[WAITING_QR_SERVICE] 等待二维码识别结果... 已耗时: %.1f 秒", time_in_state);
    }
}

void NavigationStateMachine::handleMoveToPickZone() {
    if (!task_flags_.pick_goal_sent) {
        ROS_INFO("[MOVE_TO_PICK_ZONE] 前往拣货区");
        speak("正在前往拣货区");
        sendNavigationGoal("pick_zone");
        task_flags_.pick_goal_sent = true;
    }
    
    // 时间统计
    double time_in_state = (ros::Time::now() - state_start_time_).toSec();
    ROS_INFO_THROTTLE(5, "[MOVE_TO_PICK_ZONE] 已耗时: %.1f 秒", time_in_state);
}

// ========== 三状态PCA检测 ==========

void NavigationStateMachine::handleRotationScan() {
    static bool first_enter = true;
    static ros::Time rotation_start_time;
    static float rotation_start_yaw = 0.0f;
    static bool rotation_active = false;

    if (first_enter) {
        ROS_INFO("[ROTATION_SCAN] 开始旋转扫描寻找目标物体");
        speak("开始扫描寻找目标物体");
        
        // === 重置视觉状态 ===
        try {
            std_srvs::Trigger reset_srv;
            ros::ServiceClient reset_client = nh_.serviceClient<std_srvs::Trigger>("/reset_vision_state");
            
            if (reset_client.waitForExistence(ros::Duration(1.0))) {
                if (reset_client.call(reset_srv)) {
                    ROS_INFO("旋转前视觉状态重置: %s", reset_srv.response.message.c_str());
                }
            }
        } catch (const std::exception& e) {
            ROS_WARN("视觉重置服务异常: %s", e.what());
        }
        
         ros::Duration(1.0).sleep();

        // === 重置状态标志 ===
        rotation_scan_complete_ = false;
        pca_calculation_complete_ = false;
        cached_laser_scans_.clear();
        current_rotated_angle_ = 0.0f;
        object_detected_during_scan_ = false;
        detected_object_name_ = "";
        rotation_active = true;
        
        // === 获取初始位姿 ===
        if (!getRobotPose(scan_robot_x_, scan_robot_y_, scan_robot_yaw_)) {
            ROS_WARN("无法获取机器人位姿，延迟扫描");
            return;
        }
        
        rotation_start_yaw = scan_robot_yaw_;
        rotation_start_time = ros::Time::now();
        
        // === 开始旋转 ===
        geometry_msgs::Twist rotate_cmd;
        rotate_cmd.angular.z = 0.6f; 
        cmd_vel_pub_.publish(rotate_cmd);
        
        ROS_INFO("旋转扫描开始，等待视觉识别目标物体...");
        ROS_INFO("当前任务目标: %s", current_task_.c_str());
        
        first_enter = false;
        // 重要：不return，继续执行下面的视觉检查逻辑
    }
    
    // === 持续发布旋转命令，确保机器人持续旋转 ===
    if (rotation_active) {
        geometry_msgs::Twist rotate_cmd;
        rotate_cmd.angular.z = 0.6f;
        cmd_vel_pub_.publish(rotate_cmd);
    }
    
    // === 检查视觉识别结果（第一次进入就会检查）===
    if (object_detected_during_scan_ && !detected_object_name_.empty()) {
        ROS_INFO("🎯 检测到目标物体: %s，立即停止旋转", detected_object_name_.c_str());
        speak("发现目标" + detected_object_name_);
        
        // 停止旋转
        geometry_msgs::Twist stop_cmd;
        stop_cmd.angular.z = 0.0;
        cmd_vel_pub_.publish(stop_cmd);
        rotation_active = false;
        
        // 等待机器人稳定
        ros::Duration(1.0).sleep();
        
        // 使用当前收集的数据进行PCA计算
        ROS_INFO("使用已收集的 %zu 帧激光数据进行PCA定位", cached_laser_scans_.size());
        
        if (cached_laser_scans_.size() >= 2) {
            rotation_scan_complete_ = true;
            setState(RobotState::PCA_CALCULATION);
        } else {
            ROS_WARN("收集的激光数据不足，直接进入选择阶段");
            setState(RobotState::CLUSTER_SELECTION);
        }
        
        return;
    }
    
    // === 更新当前旋转角度（用于超时判断）===
    float current_x, current_y, current_yaw;
    if (getRobotPose(current_x, current_y, current_yaw)) {
        current_rotated_angle_ = fabs(current_yaw - rotation_start_yaw);
        if (current_rotated_angle_ > M_PI) {
            current_rotated_angle_ = 2 * M_PI - current_rotated_angle_;
        }
    }
    
    // === 时间统计和超时处理 ===
    double rotation_time = (ros::Time::now() - rotation_start_time).toSec();
    ROS_INFO_THROTTLE(1, "[ROTATION_SCAN] 旋转中... 进度: %.1f°, 耗时: %.1f秒, 等待目标: %s", 
                     current_rotated_angle_ * 180 / M_PI, rotation_time, current_task_.c_str());
    
    // 超时保护：如果旋转超过一定时间或角度仍未发现目标，停止旋转
    bool should_timeout = false;
    if (rotation_time > 15.0) { // 15秒超时
        should_timeout = true;
        ROS_WARN("旋转扫描超时，未发现目标物体");
    } else if (current_rotated_angle_ >= rotation_target_angle_) {
        should_timeout = true;
        ROS_WARN("旋转达到目标角度，未发现目标物体");
    }
    
    if (should_timeout && rotation_active) {
        ROS_WARN("旋转扫描完成，未发现目标物体");
        speak("未发现目标物体，继续执行");
        
        // 停止旋转
        geometry_msgs::Twist stop_cmd;
        stop_cmd.angular.z = 0.0;
        cmd_vel_pub_.publish(stop_cmd);
        rotation_active = false;
        
        // 进入下一状态
        if (cached_laser_scans_.size() >= 2) {
            rotation_scan_complete_ = true;
            setState(RobotState::PCA_CALCULATION);
        } else {
            setState(RobotState::CLUSTER_SELECTION);
        }
    }
}

void NavigationStateMachine::handlePCACalculation() {
    static bool first_enter = true;
    
    if (first_enter) {
        ROS_INFO("[PCA_CALCULATION] 开始PCA计算");
        
        // 重置检测结果
        detected_clusters_.clear();
        detected_cluster_infos_.clear();
        clusters_detected_ = false;
        
        // 使用缓存的激光数据进行PCA计算
        if (!cached_laser_scans_.empty()) {
            ROS_INFO("使用 %zu 帧缓存激光数据进行PCA计算", cached_laser_scans_.size());
            
            // 使用最后一帧数据进行计算（通常最稳定）
            const auto& latest_scan = cached_laser_scans_.back();
            // 创建共享指针来调用函数
            sensor_msgs::LaserScan::ConstPtr scan_ptr = boost::make_shared<sensor_msgs::LaserScan>(latest_scan);
            detectObjectClusters(scan_ptr);
            
            pca_calculation_complete_ = true;
            ROS_INFO("PCA计算完成，检测到 %zu 个识别板", detected_clusters_.size());
        } else {
            ROS_WARN("没有可用的激光数据，PCA计算跳过");
        }
        
        first_enter = false;
    }
    
    // PCA计算是瞬时操作，完成后立即进入下一状态
    if (pca_calculation_complete_) {
        setState(RobotState::CLUSTER_SELECTION);
    }
}

void NavigationStateMachine::handleClusterSelection() {
    static bool first_enter = true;
    
    if (first_enter) {
        ROS_INFO("[CLUSTER_SELECTION] 开始簇选择");
        
        if (clusters_detected_ && !detected_clusters_.empty()) {
            // 选择最佳识别板
            selectBestCluster();
            
            if (current_target_cluster_ >= 0 && current_target_cluster_ < detected_clusters_.size()) {
                ROS_INFO("成功选择第 %d 个识别板，切换到导航状态", current_target_cluster_ + 1);
                speak("找到识别板，开始导航");
                setState(RobotState::NAVIGATING_TO_BOARD);
            } else {
                ROS_WARN("簇选择失败，前往等待区");
                speak("未找到合适识别板，继续执行");
                setState(RobotState::MOVE_TO_WAIT_ZONE);
            }
        } else {
            ROS_WARN("没有检测到有效识别板，前往等待区");
            speak("未检测到识别板，继续执行");
            setState(RobotState::MOVE_TO_WAIT_ZONE);
        }
        
        first_enter = false;
    }
}

void NavigationStateMachine::handleNavigatingToBoard() {
    if (clusters_detected_ && current_target_cluster_ >= 0 && 
        current_target_cluster_ < detected_clusters_.size() && !moving_to_cluster_) {
        
        geometry_msgs::Point target_point = detected_clusters_[current_target_cluster_];
        float board_yaw = detected_cluster_infos_[current_target_cluster_].board_yaw;
        float robot_target_yaw = board_yaw;
        
        ROS_INFO("[NAVIGATING_TO_BOARD] 前往第 %d 个识别板", current_target_cluster_ + 1);
        ROS_INFO("  目标位置: (%.2f, %.2f)", target_point.x, target_point.y);
        ROS_INFO("  板子朝向: %.1f°", board_yaw * 180 / M_PI);
        ROS_INFO("  机器人目标朝向: %.1f°", robot_target_yaw * 180 / M_PI);
        
        move_base_msgs::MoveBaseGoal goal;
        goal.target_pose = createPose(target_point.x, target_point.y, robot_target_yaw);
        goal.target_pose.header.stamp = ros::Time::now();
        goal.target_pose.header.frame_id = "map";
        
        action_client_.sendGoal(goal,
            boost::bind(&NavigationStateMachine::navDoneCallback, this, _1, _2),
            boost::bind(&NavigationStateMachine::navActiveCallback, this),
            boost::bind(&NavigationStateMachine::navFeedbackCallback, this, _1));
        
        moving_to_cluster_ = true;
        task_flags_.navigation_in_progress = true;
        cluster_nav_start_time_ = ros::Time::now();
    }
    
    // 时间统计
    if (moving_to_cluster_) {
        double nav_time = (ros::Time::now() - cluster_nav_start_time_).toSec();
        ROS_INFO_THROTTLE(2, "[NAVIGATING_TO_BOARD] 导航耗时: %.1f 秒", nav_time);
    }
}

void NavigationStateMachine::handleWaitingVisual() {
    static bool first_entered = true;
    static ros::Time wait_start_time;
    static ros::Time detection_start_time;
    static bool initial_delay_passed = false;
    
    if (first_entered) {
        ROS_INFO("[WAITING_VISUAL] 到达识别板位置，等待视觉系统稳定");

        try {
            std_srvs::Trigger reset_srv;
            ros::ServiceClient reset_client = nh_.serviceClient<std_srvs::Trigger>("/reset_vision_state");
            
            if (reset_client.waitForExistence(ros::Duration(1.0))) {
                if (reset_client.call(reset_srv)) {
                    if (reset_srv.response.success) {
                        ROS_INFO("视觉状态重置成功: %s", reset_srv.response.message.c_str());
                    } else {
                        ROS_WARN("视觉状态重置失败: %s", reset_srv.response.message.c_str());
                    }
                } else {
                    ROS_WARN("视觉重置服务调用失败");
                }
            } else {
                ROS_WARN("视觉重置服务不可用，继续执行");
            }
        } catch (const std::exception& e) {
            ROS_WARN("视觉重置服务异常: %s", e.what());
        }

        wait_start_time = ros::Time::now();
        detection_start_time = ros::Time::now() + ros::Duration(1.5);
        first_entered = false;
        initial_delay_passed = false;
        return;
    }
    
    // 时间统计
    double time_in_state = (ros::Time::now() - state_start_time_).toSec();
    double wait_time = (ros::Time::now() - wait_start_time).toSec();
    ROS_INFO_THROTTLE(2, "[WAITING_VISUAL] 状态耗时: %.1f 秒, 视觉等待: %.1f 秒", time_in_state, wait_time);
    
    if (!initial_delay_passed) {
        if (ros::Time::now() < detection_start_time) {
            ROS_INFO_THROTTLE(1, "[WAITING_VISUAL] 等待视觉系统稳定...");
            return;
        } else {
            ROS_INFO("[WAITING_VISUAL] 视觉系统稳定，开始正式检测");
            initial_delay_passed = true;
        }
    }
    
    std_srvs::Trigger srv;
    if (object_service_client_.call(srv)) {
        std::string current_object = srv.response.message;
        
        if (current_object == "CONTINUE_DETECTING") {
            ROS_INFO_THROTTLE(2, "视觉系统正在检测中，继续等待...");
            ros::Duration(0.3).sleep();
            return;
        }
        
        if (current_object == "NO_OBJECT_DETECTED") {
            ROS_INFO_THROTTLE(2, "当前未检测到任何物体，继续等待...");
            ros::Duration(0.3).sleep();
            return;
        }
        
        if (current_object.find("WARN:") != 0) {
            // 视觉节点已经完成稳定性检查，直接接受识别结果
            ROS_INFO("视觉节点确认物体: %s", current_object.c_str());
            picked_object_ = current_object;
            speak("识别到" + current_object);
            setState(RobotState::OBJECT_CONFIRMED);
            first_entered = true;
        } else {
            std::string mismatched_object = current_object.substr(5);
            ROS_WARN("识别到不匹配物体: %s，立即切换到下一个识别板", mismatched_object.c_str());
            ros::Duration(0.2).sleep();
            moveToNextCluster();
            first_entered = true;
        }
    } else {
        ROS_ERROR_THROTTLE(2, "无法调用物体识别服务，等待重试...");
        ros::Duration(0.3).sleep();
        return;
    }
    
    if (wait_time > VISUAL_RECOGNITION_TIMEOUT) {
        ROS_WARN("视觉识别超时，耗时: %.1f 秒，前往下一个识别板", wait_time);
        moveToNextCluster();
        first_entered = true;
    }
    
    ros::Duration(0.3).sleep();
}

void NavigationStateMachine::handleObjectConfirmed() {
    ROS_INFO("[OBJECT_CONFIRMED] 物体确认: %s", picked_object_.c_str());
    speak("我已取到" + picked_object_);
    task_flags_.object_picked = true;
    
    moving_to_cluster_ = false;
    current_target_cluster_ = -1;
    object_service_called_ = false;
    
    setState(RobotState::MOVE_TO_WAIT_ZONE);
}

void NavigationStateMachine::handleMoveToWaitZone() {
    if (!task_flags_.wait_goal_sent) {
        ROS_INFO("[MOVE_TO_WAIT_ZONE] 前往等待区");
        speak("正在前往等待区，等待仿真任务完成");
        sendNavigationGoal("wait_zone");
        task_flags_.wait_goal_sent = true;
    }
    
    // 时间统计
    double time_in_state = (ros::Time::now() - state_start_time_).toSec();
    ROS_INFO_THROTTLE(2, "[MOVE_TO_WAIT_ZONE] 等待导航完成... 已耗时: %.1f 秒", time_in_state);
}

void NavigationStateMachine::handleWaitingSimulation() {
    static bool first_enter = true;
    static ros::Time wait_start_time;
    
    if (first_enter) {
        ROS_INFO("[WAITING_SIMULATION] 发布拾取物品，等待A客户端处理");
        speak("等待仿真任务完成");
        
        // ✅ 发布任务类型和拾取物品
        std_msgs::String task_msg;
        task_msg.data = current_task_;
        task_pub_.publish(task_msg);
        ROS_INFO("已发布任务类型到 /current_task: %s", current_task_.c_str());
        
        std_msgs::String object_msg;
        object_msg.data = picked_object_;
        ros::Publisher picked_object_pub = nh_.advertise<std_msgs::String>("/picked_object", 1, true);
        picked_object_pub.publish(object_msg);
        ROS_INFO("已发布拾取物品到 /picked_object: %s", picked_object_.c_str());
        
        wait_start_time = ros::Time::now();
        first_enter = false;
        return;
    }
    
    // ✅ 等待A客户端调用B服务返回结果
    double wait_time = (ros::Time::now() - wait_start_time).toSec();
    ROS_INFO_THROTTLE(2, "[WAITING_SIMULATION] 等待A客户端返回B服务器结果... 已等待: %.1f 秒", wait_time);
    
    if (task_flags_.simulation_received) {
        ROS_INFO("[WAITING_SIMULATION] 收到仿真结果: %s", simulation_result_.c_str());
        
        // 修改：代价计算已经在callSimulationService中完成，这里只需要语音播报
        speak("仿真任务已完成，目标货物位于" + simulation_result_ + "房间");
        
        first_enter = true;
        setState(RobotState::MOVE_TO_TRAFFIC_ZONE);
    }
    
    // 超时处理
    if (wait_time > 5.0) { // 增加到30秒超时
        ROS_WARN("[WAITING_SIMULATION] A客户端响应超时，使用模拟数据继续");
        simulation_result_ = "A101";
        
        // 修改：超时情况下也计算代价
        updateCostCalculation(picked_object_, simulation_result_);
        
        speak("仿真任务超时，使用默认路径继续");
        
        first_enter = true;
        task_flags_.simulation_received = false;
        setState(RobotState::MOVE_TO_TRAFFIC_ZONE);
    }
}

void NavigationStateMachine::handleMoveToTrafficZone() {
    if (!task_flags_.traffic_goal_sent) {
        ROS_INFO("[MOVE_TO_TRAFFIC_ZONE] 前往路牌识别区");
        speak("正在前往路牌识别区");
        sendNavigationGoal("traffic_zone");
        task_flags_.traffic_goal_sent = true;
    }
    
    // 时间统计
    double time_in_state = (ros::Time::now() - state_start_time_).toSec();
    ROS_INFO_THROTTLE(2, "[MOVE_TO_TRAFFIC_ZONE] 等待导航完成... 已耗时: %.1f 秒", time_in_state);
}


void NavigationStateMachine::handleWaitingTraffic() {
    static ros::Time traffic_wait_start_time;
    static bool first_enter = true;
    static bool initial_delay_passed = false;
    
    if (first_enter) {
        ROS_INFO("[WAITING_TRAFFIC] 开始等待交通灯识别，初始延迟0.5秒");
        traffic_wait_start_time = ros::Time::now();
        first_enter = false;
        initial_delay_passed = false;
        return;
    }
    
    // 时间统计
    double time_in_state = (ros::Time::now() - state_start_time_).toSec();
    double wait_time = (ros::Time::now() - traffic_wait_start_time).toSec();
    
    // 初始延迟：至少等待0.5秒才开始处理识别结果
    if (!initial_delay_passed) {
        if (wait_time < 0.5) {
            ROS_INFO_THROTTLE(1, "[WAITING_TRAFFIC] 初始延迟中... %.1f/0.5秒", wait_time);
            return;
        } else {
            ROS_INFO("[WAITING_TRAFFIC] 初始延迟结束，开始处理交通灯识别结果");
            initial_delay_passed = true;
        }
    }
    
    ROS_INFO_THROTTLE(2, "[WAITING_TRAFFIC] 等待路牌识别结果... 已等待: %.1f 秒", wait_time);
    
    if (task_flags_.traffic_received) {
        // 只处理有效的识别结果（A或B），忽略"unknown"
        if (traffic_result_ == "A" || traffic_result_ == "B") {
            ROS_INFO("[WAITING_TRAFFIC] 收到有效路牌识别结果: %s", traffic_result_.c_str());
            speak("路口" + traffic_result_ + "可通过");
            task_flags_.traffic_received = false;
            first_enter = true;  // 重置状态
            initial_delay_passed = false;
            setState(RobotState::NAVIGATE_TO_FINISH);
        } else {
            ROS_WARN_THROTTLE(1, "[WAITING_TRAFFIC] 忽略无效识别结果: %s，继续等待...", traffic_result_.c_str());
            // 不清除traffic_received标志，等待下一个有效结果
        }
    }
    
}

void NavigationStateMachine::handleNavigateToFinish() {
    if (!task_flags_.finish_goal_sent) {
        ROS_INFO("[NAVIGATE_TO_FINISH] 使用中继点序列前往终点");
        
        // 设置中继点序列
        waypoint_sequence_.clear();
        current_waypoint_index_ = 0;
        
        if (traffic_result_ == "A") {
            // A路口可通过 -> 使用A路口作为中继点前往终点B
            waypoint_sequence_ = {"intersection_A", "finish_zone_B"};
            ROS_INFO("A路口可通过，使用B路口作为中继点前往终点B");
        } else if (traffic_result_ == "B") {
            // B路口可通过 -> 使用B路口作为中继点前往终点A
            waypoint_sequence_ = {"intersection_B", "finish_zone_A"};
            ROS_INFO("B路口可通过，使用A路口作为中继点前往终点A");
        } else {
            ROS_ERROR("未知的路口识别结果: %s", traffic_result_.c_str());
            setState(RobotState::ERROR);
            return;
        }
        
        speak("正在使用中继点导航前往终点");
        
        // 开始第一个中继点
        sendNavigationGoal(waypoint_sequence_[0]);
        following_waypoint_sequence_ = true;
        task_flags_.finish_goal_sent = true;
        
        ROS_INFO("开始中继点序列，共 %zu 个点", waypoint_sequence_.size());
        for (size_t i = 0; i < waypoint_sequence_.size(); ++i) {
            ROS_INFO("  中继点[%zu]: %s", i, waypoint_sequence_[i].c_str());
        }
    }
    
    // 时间统计
    double time_in_state = (ros::Time::now() - state_start_time_).toSec();
    ROS_INFO_THROTTLE(2, "[NAVIGATE_TO_FINISH] 中继点导航中... 已耗时: %.1f 秒", time_in_state);
}

void NavigationStateMachine::handleTaskComplete() {
    static bool task_complete_announced = false;
    
    if (!task_complete_announced) {
        ROS_INFO("[TASK_COMPLETE] 任务完成");
        
        // 机器人默认携带20元
        double payment = 20.0;
        double change = payment - total_cost_;
        
        // 生成采购报告
        std::string purchase_report = generatePurchaseReport(payment, change);
        
        speak("我已完成货物采购任务，" + purchase_report);
        
        // 输出详细采购信息到日志
        printPurchaseDetails(payment, change);
        
        // 延迟一下再输出时间统计
        ros::Duration(1.0).sleep();
        
        // 输出总时间统计
        printTimeStatistics();
        
        ROS_INFO("=== 演示任务完成 ===");
        task_complete_announced = true;
    }
    
    ROS_INFO_THROTTLE(5, "[TASK_COMPLETE] 任务已完成，等待程序结束...");
}

void NavigationStateMachine::handleErrorState() {
    double time_in_state = (ros::Time::now() - state_start_time_).toSec();
    ROS_ERROR("[ERROR] 进入错误状态，已持续: %.1f 秒，尝试恢复...", time_in_state);
    speak("导航出现问题，尝试恢复");
    
    action_client_.cancelAllGoals();
    stopMoving();
    
    if (task_flags_.object_picked) {
        ROS_INFO("恢复：前往等待区");
        setState(RobotState::MOVE_TO_WAIT_ZONE);
    } else {
        ROS_INFO("恢复：重新寻找物体");
        setState(RobotState::ROTATION_SCAN); // 修改为新的起始状态
    }
    
    ros::Duration(1.0).sleep();
}

// ========== 时间统计函数 ==========

void NavigationStateMachine::recordStateDuration(RobotState state, double duration) {
    state_durations_[static_cast<int>(state)] = duration;
}

void NavigationStateMachine::printTimeStatistics() {
    ROS_INFO("========== 导航状态时间统计 ==========");
    double total_time = 0.0;
    
    // 添加当前状态的持续时间
    ros::Time current_time = ros::Time::now();
    double current_state_duration = (current_time - state_start_time_).toSec();
    recordStateDuration(current_state_, current_state_duration);
    
    for (const auto& entry : state_durations_) {
        RobotState state = static_cast<RobotState>(entry.first);
        double duration = entry.second;
        total_time += duration;
        
        const char* state_name = getStateName(state);
        ROS_INFO("状态 %d (%s): %.1f 秒", entry.first, state_name, duration);
    }
    
    ROS_INFO("总执行时间: %.1f 秒 (约 %.1f 分钟)", total_time, total_time / 60.0);
    ROS_INFO("======================================");
}

const char* NavigationStateMachine::getStateName(RobotState state) {
    switch(state) {
        case RobotState::INIT: return "INIT";
        case RobotState::MOVE_TO_QR_ZONE: return "MOVE_TO_QR_ZONE";
        case RobotState::WAITING_QR_SERVICE: return "WAITING_QR_SERVICE";
        case RobotState::MOVE_TO_PICK_ZONE: return "MOVE_TO_PICK_ZONE";
        // 新增三状态
        case RobotState::ROTATION_SCAN: return "ROTATION_SCAN";
        case RobotState::PCA_CALCULATION: return "PCA_CALCULATION";
        case RobotState::CLUSTER_SELECTION: return "CLUSTER_SELECTION";
        // 原有状态
        case RobotState::NAVIGATING_TO_BOARD: return "NAVIGATING_TO_BOARD";
        case RobotState::WAITING_VISUAL: return "WAITING_VISUAL";
        case RobotState::OBJECT_CONFIRMED: return "OBJECT_CONFIRMED";
        case RobotState::MOVE_TO_WAIT_ZONE: return "MOVE_TO_WAIT_ZONE";
        case RobotState::WAITING_SIMULATION: return "WAITING_SIMULATION";
        case RobotState::MOVE_TO_TRAFFIC_ZONE: return "MOVE_TO_TRAFFIC_ZONE";
        case RobotState::WAITING_TRAFFIC: return "WAITING_TRAFFIC";
        case RobotState::NAVIGATE_TO_FINISH: return "NAVIGATE_TO_FINISH";
        case RobotState::TASK_COMPLETE: return "TASK_COMPLETE";
        case RobotState::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

// ========== PCA核心算法 ==========

PCAResult NavigationStateMachine::computePCA(const std::vector<int>& cluster, 
                                            const sensor_msgs::LaserScan::ConstPtr& scan) {
    PCAResult result;
    
    if (cluster.size() < 5) {
        ROS_WARN("PCA计算需要至少5个点，当前只有%zu个点", cluster.size());
        return result;
    }
    
    // 步骤1: 收集所有点的全局坐标
    std::vector<float> points_x, points_y;
    for(int idx : cluster) {
        float dist = scan->ranges[idx];
        float angle = scan->angle_min + idx * scan->angle_increment;
        
        float local_x = dist * cos(angle);
        float local_y = dist * sin(angle);
        
        float global_x = scan_robot_x_ + local_x * cos(scan_robot_yaw_) - local_y * sin(scan_robot_yaw_);
        float global_y = scan_robot_y_ + local_x * sin(scan_robot_yaw_) + local_y * cos(scan_robot_yaw_);
        
        points_x.push_back(global_x);
        points_y.push_back(global_y);
    }
    
    // 步骤2: 计算均值（中心化）
    float mean_x = 0.0f, mean_y = 0.0f;
    for(size_t i = 0; i < points_x.size(); ++i) {
        mean_x += points_x[i];
        mean_y += points_y[i];
    }
    mean_x /= points_x.size();
    mean_y /= points_y.size();
    
    // 步骤3: 计算协方差矩阵
    float cov_xx = 0.0f, cov_yy = 0.0f, cov_xy = 0.0f;
    for(size_t i = 0; i < points_x.size(); ++i) {
        float dx = points_x[i] - mean_x;
        float dy = points_y[i] - mean_y;
        cov_xx += dx * dx;
        cov_yy += dy * dy;
        cov_xy += dx * dy;
    }
    cov_xx /= points_x.size();
    cov_yy /= points_y.size();
    cov_xy /= points_x.size();
    
    // 步骤4: 计算特征值和特征向量（2D PCA解析解）
    float trace = cov_xx + cov_yy;
    float determinant = cov_xx * cov_yy - cov_xy * cov_xy;
    float discriminant = trace * trace - 4 * determinant;
    
    if (discriminant < 0) {
        ROS_WARN("PCA计算异常: 判别式为负");
        return result;
    }
    
    // 特征值
    float lambda1 = (trace + sqrt(discriminant)) / 2;
    float lambda2 = (trace - sqrt(discriminant)) / 2;
    
    // 第一主成分方向（对应最大特征值）
    float principal_x, principal_y;
    if(fabs(cov_xy) > 1e-6) {
        principal_x = lambda1 - cov_yy;
        principal_y = cov_xy;
    } else {
        // 如果协方差为0，选择方差较大的方向
        principal_x = (cov_xx >= cov_yy) ? 1.0f : 0.0f;
        principal_y = (cov_xx >= cov_yy) ? 0.0f : 1.0f;
    }
    
    // 归一化方向向量
    float norm = sqrt(principal_x * principal_x + principal_y * principal_y);
    if (norm < 1e-6) {
        ROS_WARN("PCA方向向量模长为0");
        return result;
    }
    principal_x /= norm;
    principal_y /= norm;
    
    // 步骤5: 计算投影范围
    float min_proj = std::numeric_limits<float>::max();
    float max_proj = std::numeric_limits<float>::lowest();
    
    for(size_t i = 0; i < points_x.size(); ++i) {
        float proj = (points_x[i] - mean_x) * principal_x + 
                    (points_y[i] - mean_y) * principal_y;
        min_proj = std::min(min_proj, proj);
        max_proj = std::max(max_proj, proj);
    }
    
    // 步骤6: 设置结果
    result.length = max_proj - min_proj;
    result.orientation = atan2(principal_y, principal_x);
    result.confidence = (lambda1 + lambda2 > 1e-6) ? lambda1 / (lambda1 + lambda2) : 0.0f;
    
    // 计算投影起点和终点
    result.start_point.x = mean_x + min_proj * principal_x;
    result.start_point.y = mean_y + min_proj * principal_y;
    result.start_point.z = 0.0;
    result.end_point.x = mean_x + max_proj * principal_x;
    result.end_point.y = mean_y + max_proj * principal_y;
    result.end_point.z = 0.0;
    
    ROS_DEBUG("PCA结果: 长度=%.3fm, 朝向=%.1f°, 置信度=%.3f", 
             result.length, result.orientation * 180/M_PI, result.confidence);
    
    return result;
}

// ========== 激光雷达相关函数 ==========

void NavigationStateMachine::laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
    int center_index = msg->ranges.size() / 2;
    int range = 30 * M_PI / 180.0 / msg->angle_increment;
    
    float min_distance = std::numeric_limits<float>::max();
    for (int i = center_index - range; i <= center_index + range; ++i) {
        if (i >= 0 && i < msg->ranges.size()) {
            float dist = msg->ranges[i];
            if (std::isfinite(dist) && dist > msg->range_min && dist < msg->range_max) {
                min_distance = std::min(min_distance, dist);
            }
        }
    }
    obstacle_distance_ = min_distance;
    obstacle_detected_ = (min_distance <= TARGET_OBSTACLE_DISTANCE);
    
    // 在旋转扫描状态下缓存激光数据
    if (current_state_ == RobotState::ROTATION_SCAN) {
        // 限制缓存数量，避免内存过度增长
        if (cached_laser_scans_.size() < max_cached_scans_) {
            cached_laser_scans_.push_back(*msg);
            ROS_DEBUG_THROTTLE(2, "缓存激光数据，当前帧数: %zu", cached_laser_scans_.size());
        }
    }
}

void NavigationStateMachine::objectDetectedCallback(const std_msgs::String::ConstPtr& msg) {
    std::string detected_object = msg->data;
    
    if (detected_object.empty()) {
        return;
    }
    
    // 任务匹配检查
    if (!current_task_.empty()) {
        bool is_fruit = (detected_object == "香蕉" || detected_object == "西瓜" || detected_object == "苹果");
        bool is_food = (detected_object == "蛋糕" || detected_object == "牛奶" || detected_object == "可乐");
        bool is_vegetable = (detected_object == "土豆" || detected_object == "番茄" || detected_object == "辣椒");
        
        bool task_matched = false;
        if (current_task_ == "水果" && is_fruit) task_matched = true;
        else if (current_task_ == "食品" && is_food) task_matched = true;
        else if (current_task_ == "蔬菜" && is_vegetable) task_matched = true;
        
        if (!task_matched) {
            ROS_WARN("检测到物体但与任务不匹配: %s (需要: %s)", detected_object.c_str(), current_task_.c_str());
            return;
        }
    }
    
    ROS_INFO("✅ 视觉检测到目标物体: %s", detected_object.c_str());
    
    // 关键修改：在旋转扫描状态下立即记录检测结果
    if (current_state_ == RobotState::ROTATION_SCAN) {
        object_detected_during_scan_ = true;
        detected_object_name_ = detected_object;
        ROS_INFO("🎯 旋转扫描中检测到目标，准备停止旋转");
    }
}

geometry_msgs::Point NavigationStateMachine::calculateSafeTarget(const ClusterInfo& cluster_info) {
    geometry_msgs::Point safe_target;
    
    // 使用扫描时缓存的机器人位姿
    float robot_x = scan_robot_x_;
    float robot_y = scan_robot_y_;
    
    // 基础安全距离
    float safe_distance = DEFAULT_SAFE_DISTANCE;
    
    // 基于costmap的动态安全距离调整
    if (costmap_updated_) {
        // 检查原始目标点是否可达
        if (!isTargetReachable(cluster_info.center)) {
            safe_distance = EXTENDED_SAFE_DISTANCE;
            ROS_WARN("识别板原始位置不可达，延长安全距离到 %.1fm", safe_distance);
        }
    }
    
    // 沿板子朝向的反方向后退安全距离
    float back_dir_x = -cos(cluster_info.board_yaw);
    float back_dir_y = -sin(cluster_info.board_yaw);
    
    safe_target.x = cluster_info.center.x + back_dir_x * safe_distance;
    safe_target.y = cluster_info.center.y + back_dir_y * safe_distance;
    safe_target.z = 0.0;
    
    ROS_INFO("安全目标点: 板子中心(%.2f,%.2f) -> 安全点(%.2f,%.2f), 距离%.1fm, 朝向%.1f°",
            cluster_info.center.x, cluster_info.center.y,
            safe_target.x, safe_target.y, safe_distance, cluster_info.board_yaw * 180 / M_PI);
    
    return safe_target;
}

void NavigationStateMachine::detectObjectClusters(const sensor_msgs::LaserScan::ConstPtr& scan) {
    detected_clusters_.clear();
    detected_cluster_infos_.clear(); 

    ROS_INFO("=== PCA聚类识别板检测开始 ===");
    ROS_INFO("机器人位置: (%.2f, %.2f, %.1f°)", scan_robot_x_, scan_robot_y_, scan_robot_yaw_ * 180 / M_PI);

    std::vector<std::vector<int>> clusters;
    std::vector<int> current_cluster;
    
    // PCA动态聚类算法
    for(size_t i = 0; i < scan->ranges.size(); ++i) {
        float dist = scan->ranges[i];
        
        if(!std::isfinite(dist) || dist < pca_params_.min_valid_range || dist > pca_params_.max_valid_range) {
            if(!current_cluster.empty() && current_cluster.size() >= pca_params_.min_cluster_size) {
                clusters.push_back(current_cluster);
            }
            current_cluster.clear();
            continue;
        }
        
        if(current_cluster.empty()) {
            current_cluster.push_back(i);
            continue;
        }
        
        int prev_idx = current_cluster.back();
        float prev_dist = scan->ranges[prev_idx];

        // 距离梯度检查
        float distance_gradient = fabs(dist - prev_dist);
        const float GRADIENT_THRESHOLD = 0.3f;
        
        if(distance_gradient > GRADIENT_THRESHOLD) {
            // 立即分割当前聚类
            if(!current_cluster.empty() && current_cluster.size() >= pca_params_.min_cluster_size) {
                clusters.push_back(current_cluster);
            }
            current_cluster.clear();
            current_cluster.push_back(i);
            continue;
        }

        float prev_angle = scan->angle_min + prev_idx * scan->angle_increment;
        float curr_angle = scan->angle_min + i * scan->angle_increment;
        
        float x1 = prev_dist * cos(prev_angle);
        float y1 = prev_dist * sin(prev_angle);
        float x2 = dist * cos(curr_angle);
        float y2 = dist * sin(curr_angle);
        float physical_distance = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
        
        if(physical_distance < pca_params_.max_distance_jump) {
            current_cluster.push_back(i);
        } else {
            if(current_cluster.size() >= pca_params_.min_cluster_size) {
                clusters.push_back(current_cluster);
            }
            current_cluster.clear();
            current_cluster.push_back(i);
        }
    }
    
    if(!current_cluster.empty() && current_cluster.size() >= pca_params_.min_cluster_size) {
        clusters.push_back(current_cluster);
    }
    
    ROS_INFO("PCA初步聚类完成，共 %zu 个候选簇", clusters.size());
    
    // 处理每个簇
    std::vector<geometry_msgs::Point> temp_clusters;
    std::vector<ClusterInfo> temp_infos;
    
    int valid_clusters = 0;
    for(size_t i = 0; i < clusters.size(); ++i) {
        const auto& cluster = clusters[i];
        ClusterInfo cluster_info = calculateClusterInfo(cluster, scan);
        
        std::string debug_info;
        bool isValid = isValidObjectCluster(cluster_info, cluster, scan, debug_info);
        
        if(isValid) {
            geometry_msgs::Point safe_target = calculateSafeTarget(cluster_info);
            temp_clusters.push_back(safe_target);
            temp_infos.push_back(cluster_info);
            valid_clusters++;
            
            ROS_INFO("✅ PCA聚类 %zu 有效: 长度=%.3fm, 距离=%.2fm, 置信度=%.3f", 
                    i+1, cluster_info.length, cluster_info.average_distance, 
                    cluster_info.pca_confidence);
        } else {
            ROS_WARN("❌ PCA聚类 %zu 被过滤: %s", i+1, debug_info.c_str());
        }
    }
    
    // 去重处理
    std::vector<bool> keep_flag(temp_clusters.size(), true);
    int duplicates_removed = 0;
    for(size_t i = 0; i < temp_clusters.size(); ++i) {
        if(!keep_flag[i]) continue;
        
        for(size_t j = i + 1; j < temp_clusters.size(); ++j) {
            if(!keep_flag[j]) continue;
            
            float dx = temp_clusters[i].x - temp_clusters[j].x;
            float dy = temp_clusters[i].y - temp_clusters[j].y;
            float distance = sqrt(dx*dx + dy*dy);
            
            if(distance < pca_params_.duplicate_distance) {
                float dist_i = sqrt(pow(temp_clusters[i].x - scan_robot_x_, 2) + 
                                   pow(temp_clusters[i].y - scan_robot_y_, 2));
                float dist_j = sqrt(pow(temp_clusters[j].x - scan_robot_x_, 2) + 
                                   pow(temp_clusters[j].y - scan_robot_y_, 2));
                
                if(dist_i < dist_j) {
                    keep_flag[j] = false;
                } else {
                    keep_flag[i] = false;
                    break;
                }
                duplicates_removed++;
            }
        }
    }
    
    if(duplicates_removed > 0) {
        ROS_INFO("PCA去重处理完成，移除了 %d 个重复聚类", duplicates_removed);
    }
    
    // 更新检测结果
    for(size_t i = 0; i < temp_clusters.size(); ++i) {
        if(keep_flag[i]) {
            detected_clusters_.push_back(temp_clusters[i]);
            detected_cluster_infos_.push_back(temp_infos[i]);
        }
    }
    
    clusters_detected_ = !detected_clusters_.empty();
    
    if(clusters_detected_) {
        ROS_INFO("🎯 PCA最终检测到 %zu 个识别板", detected_clusters_.size());
        
        // 显示最终有效的板子列表
        for(size_t i = 0; i < detected_clusters_.size(); ++i) {
            const auto& cluster = detected_clusters_[i];
            const auto& info = detected_cluster_infos_[i];
            
            ROS_INFO("识别板 %zu:", i+1);
            ROS_INFO("  ├─ 板子中心: (%.2f, %.2f)", info.center.x, info.center.y);
            ROS_INFO("  ├─ 安全目标点: (%.2f, %.2f)", cluster.x, cluster.y);
            ROS_INFO("  ├─ PCA长度: %.3fm", info.length);
            ROS_INFO("  ├─ PCA朝向: %.1f°", info.board_yaw * 180 / M_PI);
            ROS_INFO("  ├─ PCA置信度: %.3f", info.pca_confidence);
            ROS_INFO("  ├─ 距离: %.1fm", info.average_distance);
            ROS_INFO("  └─ 点数: %zu", info.size);
        }
    } else {
        ROS_INFO("⚠️ PCA未检测到符合标准的识别板");
    }
}

NavigationStateMachine::ClusterInfo NavigationStateMachine::calculateClusterInfo(const std::vector<int>& cluster, 
                                                                                const sensor_msgs::LaserScan::ConstPtr& scan) {
    ClusterInfo info;
    float sum_x = 0.0f, sum_y = 0.0f;
    float sum_dist = 0.0f;
    
    std::vector<float> global_x_points, global_y_points;
    
    for(int idx : cluster) {
        float dist = scan->ranges[idx];
        float angle = scan->angle_min + idx * scan->angle_increment;
        
        float local_x = dist * cos(angle);
        float local_y = dist * sin(angle);
        
        float global_x = scan_robot_x_ + local_x * cos(scan_robot_yaw_) - local_y * sin(scan_robot_yaw_);
        float global_y = scan_robot_y_ + local_x * sin(scan_robot_yaw_) + local_y * cos(scan_robot_yaw_);
        
        sum_x += global_x;
        sum_y += global_y;
        sum_dist += dist;
        
        global_x_points.push_back(global_x);
        global_y_points.push_back(global_y);
    }
    
    info.center.x = sum_x / cluster.size();
    info.center.y = sum_y / cluster.size();
    info.center.z = 0.0;
    info.average_distance = sum_dist / cluster.size();
    info.size = cluster.size();
    info.angular_width = (cluster.back() - cluster.front()) * scan->angle_increment;
    
    // 使用PCA计算板子朝向和长度
    PCAResult pca_result = computePCA(cluster, scan);
    info.length = pca_result.length;
    info.pca_confidence = pca_result.confidence;
    
    // 计算法向量并选择面向机器人的一侧
    float principal_yaw = pca_result.orientation;
    float normal_yaw = principal_yaw + M_PI / 2;
    
    // 确保法向量面向机器人
    float dx = info.center.x - scan_robot_x_;
    float dy = info.center.y - scan_robot_y_;
    
    float normal_dx = cos(normal_yaw);
    float normal_dy = sin(normal_yaw);
    float dot_product = normal_dx * dx + normal_dy * dy;
    
    // 如果点积为负，说明法向量背对机器人，需要翻转180度
    if (dot_product < 0) {
        normal_yaw += M_PI;
        ROS_DEBUG("法向量翻转180度，从背对机器人调整为面向机器人");
    }
    
    // 归一化到 [-π, π]
    while(normal_yaw > M_PI) normal_yaw -= 2 * M_PI;
    while(normal_yaw < -M_PI) normal_yaw += 2 * M_PI;
    
    // 使用法向量作为最终朝向
    info.board_yaw = normal_yaw;
    
    ROS_INFO("PCA计算: 长度=%.3fm, 主方向=%.1f°, 法向量=%.1f°, 置信度=%.3f", 
             info.length, principal_yaw * 180 / M_PI, 
             info.board_yaw * 180 / M_PI, info.pca_confidence);
    
    return info;
}

bool NavigationStateMachine::isValidObjectCluster(const ClusterInfo& cluster_info, 
                                                const std::vector<int>& cluster,
                                                const sensor_msgs::LaserScan::ConstPtr& scan,
                                                std::string& debug_info) {
    std::ostringstream oss;
    
    // 地理约束检查
    float x = cluster_info.center.x;
    float y = cluster_info.center.y;
    
    if (x < pca_params_.valid_min_x || x > pca_params_.valid_max_x || 
        y < pca_params_.valid_min_y || y > pca_params_.valid_max_y) {
        oss << "超出有效区域: (" << x << "," << y << ") 不在 [" 
            << pca_params_.valid_min_x << "," << pca_params_.valid_max_x << "]x[" 
            << pca_params_.valid_min_y << "," << pca_params_.valid_max_y << "]";
        debug_info = oss.str();
        return false;
    }

    // 基本长度检查
    if(cluster_info.length < pca_params_.min_board_length) {
        oss << "PCA长度过小: " << cluster_info.length << "m < " << pca_params_.min_board_length << "m";
        debug_info = oss.str();
        return false;
    }
    
    if(cluster_info.length > pca_params_.max_board_length) {
        oss << "PCA长度过大: " << cluster_info.length << "m > " << pca_params_.max_board_length << "m";
        debug_info = oss.str();
        return false;
    }
    
    // 点数检查
    if(cluster.size() < pca_params_.min_cluster_size) {
        oss << "点数过少: " << cluster.size() << " < " << pca_params_.min_cluster_size;
        debug_info = oss.str();
        return false;
    }
    
    // PCA置信度检查
    if(cluster_info.pca_confidence < pca_params_.min_pca_confidence) {
        oss << "PCA置信度过低: " << cluster_info.pca_confidence << " < " << pca_params_.min_pca_confidence;
        debug_info = oss.str();
        return false;
    }
    
    // 距离连续性检查
    float distance_std = 0.0f;
    float mean_dist = cluster_info.average_distance;
    for(int idx : cluster) {
        float diff = scan->ranges[idx] - mean_dist;
        distance_std += diff * diff;
    }
    distance_std = sqrt(distance_std / cluster.size());
    
    if(distance_std > pca_params_.max_distance_std) {
        oss << "距离变化过大: 标准差=" << distance_std << "m > " << pca_params_.max_distance_std << "m";
        debug_info = oss.str();
        return false;
    }
    
    debug_info = "PCA符合识别板特征";
    return true;
}

float NavigationStateMachine::calculateBoardLength(const std::vector<int>& cluster, 
                                                 const sensor_msgs::LaserScan::ConstPtr& scan) {
    if (cluster.size() < 5) {
        ROS_WARN("PCA长度计算需要至少5个点，当前只有%zu个点", cluster.size());
        return 0.0f;
    }
    
    PCAResult pca_result = computePCA(cluster, scan);
    
    ROS_INFO("=== PCA长度计算 ===");
    ROS_INFO("输入点数: %zu", cluster.size());
    ROS_INFO("PCA长度: %.3fm", pca_result.length);
    ROS_INFO("PCA朝向: %.1f°", pca_result.orientation * 180 / M_PI);
    ROS_INFO("PCA置信度: %.3f", pca_result.confidence);
    ROS_INFO("=================");
    
    return pca_result.length;
}

void NavigationStateMachine::clusterArrivedCallback(const actionlib::SimpleClientGoalState& state,
                                                   const move_base_msgs::MoveBaseResultConstPtr& result) {
    task_flags_.navigation_in_progress = false;
    moving_to_cluster_ = false;
    
    if (state == actionlib::SimpleClientGoalState::SUCCEEDED) {
        ROS_INFO("成功到达第 %d 个识别板位置", current_target_cluster_ + 1);
        setState(RobotState::WAITING_VISUAL);
    } else {
        ROS_WARN("前往第 %d 个识别板失败，尝试下一个", current_target_cluster_ + 1);
        moveToNextCluster();
    }
}

void NavigationStateMachine::moveToNextCluster() {
    current_target_cluster_++;
    
    if (current_target_cluster_ < detected_clusters_.size()) {
        ROS_INFO("切换到第 %d 个识别板", current_target_cluster_ + 1);
        object_service_called_ = false;
        setState(RobotState::NAVIGATING_TO_BOARD);
    } else {
        ROS_WARN("所有识别板都检查完毕，未找到匹配的目标物体");
        speak("未找到目标物体，继续执行");
        setState(RobotState::MOVE_TO_WAIT_ZONE);
    }
}

// ========== 移动控制函数 ==========

void NavigationStateMachine::stopMoving() {
    geometry_msgs::Twist stop_twist;
    cmd_vel_pub_.publish(stop_twist);
    ROS_INFO("停止移动");
}

// ========== 回调函数 ==========

bool NavigationStateMachine::callQRService() {
    std_srvs::Trigger srv;
    if (qr_service_client_.call(srv)) {
        if (srv.response.success) {
            current_task_ = srv.response.message;
            ROS_INFO("二维码服务返回: %s", current_task_.c_str());
            speak("本次采购任务为" + current_task_);
            
            // 发布任务给其他节点
            std_msgs::String task_msg;
            task_msg.data = current_task_;
            task_pub_.publish(task_msg);
            
            qr_service_called_ = false;
            setState(RobotState::MOVE_TO_PICK_ZONE);
            return true;
        } else {
            ROS_ERROR("二维码识别失败: %s", srv.response.message.c_str());
            return false;
        }
    } else {
        ROS_ERROR("无法调用二维码服务");
        return false;
    }
}

bool NavigationStateMachine::callObjectRecognitionService() {
    std_srvs::Trigger srv;
    
    // 使用定义的常量
    int retry_count = 0;
    const int max_retries = SERVICE_RETRY_COUNT;
    
    while (retry_count < max_retries) {
        if (object_service_client_.call(srv)) {
            if (srv.response.success) {
                picked_object_ = srv.response.message;
                
                // 检查是否有警告标记（任务类型不匹配）
                if (picked_object_.find("WARN:") == 0) {
                    picked_object_ = picked_object_.substr(5);
                    ROS_WARN("识别到物体但与任务类型不匹配: %s，前往下一个识别板", picked_object_.c_str());
                    moveToNextCluster();
                    return false;
                } else {
                    ROS_INFO("物体识别成功且匹配: %s", picked_object_.c_str());
                    setState(RobotState::OBJECT_CONFIRMED);
                    return true;
                }
            } else {
                ROS_WARN("物体识别失败: %s (重试 %d/%d)", 
                         srv.response.message.c_str(), retry_count + 1, max_retries);
                retry_count++;
                ros::Duration(1.0).sleep();
            }
        } else {
            ROS_ERROR("无法调用物体识别服务 (重试 %d/%d)", retry_count + 1, max_retries);
            retry_count++;
            ros::Duration(1.0).sleep();
        }
    }
    
    ROS_ERROR("物体识别服务调用失败，达到最大重试次数，前往下一个识别板");
    moveToNextCluster();
    return false;
}

void NavigationStateMachine::simulationCallback(const std_msgs::String::ConstPtr& msg) {
    simulation_result_ = msg->data;
    task_flags_.simulation_received = true;
    ROS_INFO("收到仿真结果: %s", simulation_result_.c_str());
}

void NavigationStateMachine::trafficCallback(const std_msgs::String::ConstPtr& msg) {
    // 在等待交通灯状态时持续更新结果（不只是第一次）
    if (current_state_ == RobotState::WAITING_TRAFFIC) {
        traffic_result_ = msg->data;
        task_flags_.traffic_received = true;
        ROS_INFO("收到路牌识别结果: %s", traffic_result_.c_str());
    } else {
        ROS_DEBUG_THROTTLE(5, "忽略路牌识别结果[状态%d]: %s", 
                          static_cast<int>(current_state_), msg->data.c_str());
    }
}

// ========== ActionLib回调函数 ==========

void NavigationStateMachine::navDoneCallback(const actionlib::SimpleClientGoalState& state,
                                            const move_base_msgs::MoveBaseResultConstPtr& result) {
    task_flags_.navigation_in_progress = false;

    ROS_INFO("导航完成回调 - 状态: %s, 目标点: %s, 当前状态: %d", 
             state.toString().c_str(), current_goal_point_.c_str(), static_cast<int>(current_state_));

    // 如果当前状态已经不是导航状态，忽略这个回调
    if (current_state_ != RobotState::MOVE_TO_QR_ZONE &&
        current_state_ != RobotState::MOVE_TO_PICK_ZONE &&
        current_state_ != RobotState::MOVE_TO_WAIT_ZONE &&
        current_state_ != RobotState::MOVE_TO_TRAFFIC_ZONE &&
        current_state_ != RobotState::NAVIGATE_TO_FINISH &&
        current_state_ != RobotState::NAVIGATING_TO_BOARD) {
        ROS_INFO("忽略导航回调，当前状态 %d 不是导航状态", static_cast<int>(current_state_));
        return;
    }

    if (state == actionlib::SimpleClientGoalState::SUCCEEDED) {
        ROS_INFO("导航目标成功到达: %s", current_goal_point_.c_str());
        
        switch(current_state_) {
            case RobotState::MOVE_TO_QR_ZONE:
                setState(RobotState::WAITING_QR_SERVICE);
                break;
            case RobotState::MOVE_TO_PICK_ZONE:
                setState(RobotState::ROTATION_SCAN); // 修改为新的起始状态
                break;
            case RobotState::MOVE_TO_WAIT_ZONE:
                setState(RobotState::WAITING_SIMULATION);
                break;
            case RobotState::MOVE_TO_TRAFFIC_ZONE:
                setState(RobotState::WAITING_TRAFFIC);
                break;
            case RobotState::NAVIGATE_TO_FINISH:
                // 中继点导航：如果还在序列中，忽略回调（由智能切换处理）
                if (following_waypoint_sequence_) {
                    ROS_INFO("忽略中继点导航回调（由智能切换处理）");
                    return;  // 直接返回，不处理状态转换
                } else {
                    // 非中继点模式下的终点导航（备用逻辑）
                    ROS_INFO("成功到达终点: %s", current_goal_point_.c_str());
                    setState(RobotState::TASK_COMPLETE);
                }
                break;
            case RobotState::NAVIGATING_TO_BOARD:
                // 正常情况下，NAVIGATING_TO_BOARD 应该由智能停止处理
                // 如果走到这里，说明智能停止没触发，使用move_base的结果
                ROS_WARN("NAVIGATING_TO_BOARD 导航完成，但智能停止未触发，使用move_base结果");
                moving_to_cluster_ = false;
                setState(RobotState::WAITING_VISUAL);
                break;
            default:
                ROS_WARN("导航完成但当前状态 %d 不需要处理", static_cast<int>(current_state_));
                break;
        }
    } else {
        // 特别处理被取消的情况
        if (state == actionlib::SimpleClientGoalState::PREEMPTED) {
            ROS_INFO("导航被取消: %s", state.getText().c_str());
            
            // 如果是识别板导航被智能停止取消，这是正常行为
            if (current_state_ == RobotState::NAVIGATING_TO_BOARD) {
                ROS_INFO("识别板导航被智能停止取消，正常行为");
                moving_to_cluster_ = false;
                // 状态已经由智能停止设置了，不需要重复设置
                return;
            }
            
            // 如果是中继点切换取消，也是正常行为
            if (following_waypoint_sequence_) {
                ROS_INFO("中继点切换取消，正常行为");
                return;
            }
        }
        
        ROS_ERROR("导航目标失败: %s - %s", 
                 state.toString().c_str(), state.getText().c_str());
        
        // 特别处理识别板导航失败
        if (current_state_ == RobotState::NAVIGATING_TO_BOARD) {
            moving_to_cluster_ = false;
            ROS_WARN("识别板导航真正失败，尝试下一个");
            moveToNextCluster();
        } else {
            // 其他固定导航点失败进入错误状态
            setState(RobotState::ERROR);
        }
    }
}

void NavigationStateMachine::navActiveCallback() {
    ROS_INFO("导航目标已激活: %s", current_goal_point_.c_str());
}

// ========== 智能停止和中继点切换核心函数 ==========

void NavigationStateMachine::navFeedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback) {
    ROS_INFO_THROTTLE(5, "导航反馈 - 当前位置: (%.2f, %.2f)", 
                     feedback->base_position.pose.position.x,
                     feedback->base_position.pose.position.y);
    
    // === 智能停止：只在识别板导航时工作 ===
    if (current_state_ == RobotState::NAVIGATING_TO_BOARD) {
        handleBoardNavigationStop(feedback);
    }
    
    // === 中继点智能切换 ===
    if (following_waypoint_sequence_ && !waypoint_sequence_.empty()) {
        handleWaypointSwitching(feedback);
    }
}

void NavigationStateMachine::handleBoardNavigationStop(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback) {
    // 安全检查
    if (current_target_cluster_ < 0 || current_target_cluster_ >= detected_clusters_.size()) {
        return;
    }
    
    // 获取识别板目标点
    geometry_msgs::Point target_point = detected_clusters_[current_target_cluster_];
    float target_yaw = detected_cluster_infos_[current_target_cluster_].board_yaw;
    
    // 计算距离
    float dx = feedback->base_position.pose.position.x - target_point.x;
    float dy = feedback->base_position.pose.position.y - target_point.y;
    float distance = sqrt(dx*dx + dy*dy);
    
    // 计算角度差
    float current_yaw = getYawFromPose(feedback->base_position.pose);
    float yaw_diff = fabs(current_yaw - target_yaw);
    if (yaw_diff > M_PI) yaw_diff = 2 * M_PI - yaw_diff;
    
    // 检查是否满足容差
    if (distance <= 0.18f && yaw_diff <= 0.2f) {
        ROS_INFO("到达识别板位置！主动停止导航");
        action_client_.cancelAllGoals();
        stopMoving();
        
        // 状态切换
        moving_to_cluster_ = false;
        task_flags_.navigation_in_progress = false;  // 重要：重置导航标志
        setState(RobotState::WAITING_VISUAL);
    }
}

void NavigationStateMachine::handleWaypointSwitching(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback) {
    if (current_waypoint_index_ >= waypoint_sequence_.size()) {
        return;
    }
    
    std::string current_target = waypoint_sequence_[current_waypoint_index_];
    auto it = navigation_points_.find(current_target);
    if (it == navigation_points_.end()) {
        return;
    }
    
    geometry_msgs::Pose target_pose = it->second.pose;
    float dx = feedback->base_position.pose.position.x - target_pose.position.x;
    float dy = feedback->base_position.pose.position.y - target_pose.position.y;
    float distance = sqrt(dx*dx + dy*dy);
    
    bool should_switch = false;
    
    // 所有中继点都使用相同的宽松条件
    if (distance <= waypoint_switch_distance_) {
        should_switch = true;
        ROS_INFO("到达中继点 %s 附近: 距离=%.3fm", current_target.c_str(), distance);
    }
    
    if (should_switch) {
        ROS_INFO("到达中继点 %s，准备切换到下一个目标", current_target.c_str());
        
        // 如果是最后一个中继点，直接完成任务，不取消导航
        if (current_waypoint_index_ == waypoint_sequence_.size() - 1) {
            ROS_INFO("到达最终目标 %s，任务完成", current_target.c_str());
            following_waypoint_sequence_ = false;
            task_flags_.navigation_in_progress = false;
            setState(RobotState::TASK_COMPLETE);
        } else {
            // 中间点：取消导航并切换到下一个
            action_client_.cancelAllGoals();
            ros::Duration(0.1).sleep();
            
            current_waypoint_index_++;
            std::string next_target = waypoint_sequence_[current_waypoint_index_];
            ROS_INFO("切换到下一个中继点: %s", next_target.c_str());
            sendNavigationGoal(next_target);
        }
    }
}

float NavigationStateMachine::getYawFromPose(const geometry_msgs::Pose& pose) {
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
        if (task_flags_.navigation_in_progress) {
            action_client_.cancelAllGoals();
            ROS_INFO("取消之前的导航目标");
        }
        
        move_base_msgs::MoveBaseGoal goal;
        goal.target_pose = it->second;
        current_goal_point_ = point_name;
        
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

void NavigationStateMachine::setState(RobotState new_state) {
    // 记录当前状态的持续时间
    ros::Time current_time = ros::Time::now();
    double duration = (current_time - state_start_time_).toSec();
    recordStateDuration(current_state_, duration);
    
    // 在关键状态转换时验证TF数据
    if (new_state == RobotState::ROTATION_SCAN || 
        new_state == RobotState::NAVIGATING_TO_BOARD) {
        
        if (!validateTFData()) {
            ROS_WARN("TF数据不完整，延迟状态转换");
            return;
        }
    }
    
    // 状态转换时的清理工作
    if (current_state_ == RobotState::ROTATION_SCAN && 
        new_state != RobotState::ROTATION_SCAN) {
        // 离开旋转扫描状态时停止机器人
        geometry_msgs::Twist stop_cmd;
        stop_cmd.angular.z = 0.0;
        cmd_vel_pub_.publish(stop_cmd);
        
        // 新增：重置旋转扫描相关的检测标志
        object_detected_during_scan_ = false;
        detected_object_name_ = "";
        ROS_DEBUG("重置旋转扫描检测标志");
    }

    ROS_INFO("状态转换: %s (%.1f 秒) -> %s", 
             getStateName(current_state_), duration, getStateName(new_state));
    
    current_state_ = new_state;
    state_start_time_ = current_time;
}

void NavigationStateMachine::loadNavigationPoints() {
    navigation_points_["qr_zone"] = createPose(1.35, 0.92, 3.14);
    navigation_points_["pick_zone"] = createPose(1.7, 5.35, 0.0);
    navigation_points_["wait_zone"] = createPose(1.7, 6.34, 0.0);
    navigation_points_["traffic_zone"] = createPose(4.9, 6.4, 1.57); 
    navigation_points_["intersection_A"] = createPose(4.2, 4.3, -1.57);
    navigation_points_["intersection_B"] = createPose(7.3, 4.6, -1.57);
    navigation_points_["finish_zone_A"] = createPose(4.9, 0.4, -1.57);
    navigation_points_["finish_zone_B"] = createPose(6.5, 0.4, -1.57);

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

void NavigationStateMachine::updateCostCalculation(const std::string& object, const std::string& simulation_result) {
    ROS_INFO("=== 更新价格计算 ===");
    ROS_INFO("拾取物品: %s", object.c_str());
    ROS_INFO("仿真结果: %s", simulation_result.c_str());
    
    // 物品价格（根据您提供的信息）
    std::map<std::string, double> price_map = {
        {"苹果", 4.0},
        {"香蕉", 2.0},
        {"西瓜", 5.0},
        {"辣椒", 2.0},
        {"番茄", 5.0},  // 西红柿
        {"土豆", 2.0},
        {"牛奶", 5.0},
        {"蛋糕", 10.0},
        {"可乐", 3.0}
    };
    
    // 计算物品价格
    double item_price = 5.0; // 默认价格
    auto price_it = price_map.find(object);
    if (price_it != price_map.end()) {
        item_price = price_it->second;
        ROS_INFO("找到物品 %s 的价格: %.1f 元", object.c_str(), item_price);
    } else {
        ROS_WARN("物品 %s 不在价格表中，使用默认价格 5.0 元", object.c_str());
    }
    
    // 累计总价
    total_cost_ += item_price;
    
    ROS_INFO("价格明细:");
    ROS_INFO("  ├─ 物品价格: %.1f 元", item_price);
    ROS_INFO("  └─ 当前累计总价: %.1f 元", total_cost_);
    
    // 记录采购历史
    PurchaseRecord record;
    record.object = object;
    record.room = simulation_result;
    record.price = item_price;
    purchase_history_.push_back(record);
    
    ROS_INFO("=== 价格计算完成 ===");
}

bool NavigationStateMachine::getRobotPose(float& x, float& y, float& yaw) {
    try {
        geometry_msgs::TransformStamped transform;
        transform = tf_buffer_.lookupTransform("map", "base_footprint", ros::Time(0), ros::Duration(0.1));
        
        // 检查TF数据的时间戳是否合理
        ros::Time now = ros::Time::now();
        if ((now - transform.header.stamp).toSec() > 0.5) {
            ROS_WARN_THROTTLE(5, "TF数据可能过时: %.3f秒前", 
                             (now - transform.header.stamp).toSec());
        }
        
        // 获取位置
        x = transform.transform.translation.x;
        y = transform.transform.translation.y;
        
        // 获取朝向
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
        
        ROS_DEBUG_THROTTLE(5, "机器人位姿: (%.2f, %.2f, %.2f)", x, y, yaw);
        return true;
    }
    catch (tf2::TransformException &ex) {
        ROS_WARN_THROTTLE(5, "TF变换获取失败: %s", ex.what());
        
        // 备用方案：尝试其他可能的坐标系
        std::vector<std::string> base_frames = {"base_footprint", "base_link", "odom"};
        for (const auto& frame : base_frames) {
            try {
                geometry_msgs::TransformStamped transform;
                transform = tf_buffer_.lookupTransform("map", frame, ros::Time(0), ros::Duration(0.1));
                
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
                
                ROS_WARN_THROTTLE(2, "使用备用坐标系 %s 获取位姿: (%.2f, %.2f, %.2f)", 
                                 frame.c_str(), x, y, yaw);
                return true;
            }
            catch (tf2::TransformException &ex2) {
                continue;
            }
        }
        
        // 所有尝试都失败
        ROS_ERROR_THROTTLE(5, "无法获取机器人位姿，使用默认值(0,0,0)");
        x = 0.0f;
        y = 0.0f;
        yaw = 0.0f;
        return false;
    }
}

bool NavigationStateMachine::validateTFData() {
    try {
        std::vector<std::string> required_frames = {"map", "odom", "base_footprint", "laser_frame"};
        
        for (const auto& target_frame : required_frames) {
            if (!tf_buffer_.canTransform("map", target_frame, ros::Time(0), ros::Duration(0.1))) {
                ROS_WARN("缺少TF变换: map -> %s", target_frame.c_str());
                return false;
            }
        }
        
        ROS_DEBUG("TF数据验证通过");
        return true;
        
    } catch (tf2::TransformException &ex) {
        ROS_WARN("TF验证失败: %s", ex.what());
        return false;
    }
}

// ========== 代价地图功能 ==========

void NavigationStateMachine::costmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
    current_costmap_ = *msg;
    costmap_updated_ = true;
}

bool NavigationStateMachine::isTargetReachable(const geometry_msgs::Point& point) {
    if (!costmap_updated_) {
        return true;
    }
    
    // 快速坐标转换
    float map_x = current_costmap_.info.origin.position.x;
    float map_y = current_costmap_.info.origin.position.y;
    float resolution = current_costmap_.info.resolution;
    
    int mx = static_cast<int>((point.x - map_x) / resolution);
    int my = static_cast<int>((point.y - map_y) / resolution);
    
    // 快速边界检查
    if (mx < 0 || mx >= current_costmap_.info.width || 
        my < 0 || my >= current_costmap_.info.height) {
        return false;
    }
    
    int index = my * current_costmap_.info.width + mx;
    if (index < 0 || index >= current_costmap_.data.size()) {
        return false;
    }
    
    int cost = current_costmap_.data[index];
    
    // 简单阈值判断：cost < 50 认为可达
    return cost < 50;
}

// ========== 自适应安全距离计算 ==========

float NavigationStateMachine::calculateAdaptiveSafeDistance(const geometry_msgs::Point& target_point) {
    if (!costmap_updated_) return DEFAULT_SAFE_DISTANCE;
    
    float map_x = current_costmap_.info.origin.position.x;
    float map_y = current_costmap_.info.origin.position.y;
    float resolution = current_costmap_.info.resolution;
    
    // 检查目标点周围的代价
    int check_radius = 3;
    int mx = static_cast<int>((target_point.x - map_x) / resolution);
    int my = static_cast<int>((target_point.y - map_y) / resolution);
    
    int high_cost_count = 0;
    int total_points = 0;
    
    for (int dx = -check_radius; dx <= check_radius; ++dx) {
        for (int dy = -check_radius; dy <= check_radius; ++dy) {
            int check_mx = mx + dx;
            int check_my = my + dy;
            
            if (check_mx >= 0 && check_mx < current_costmap_.info.width &&
                check_my >= 0 && check_my < current_costmap_.info.height) {
                
                int index = check_my * current_costmap_.info.width + check_mx;
                if (index >= 0 && index < current_costmap_.data.size()) {
                    int cost = current_costmap_.data[index];
                    if (cost >= 50) {
                        high_cost_count++;
                    }
                    total_points++;
                }
            }
        }
    }
    
    // 根据周围障碍物密度调整安全距离
    float obstacle_ratio = (float)high_cost_count / total_points;
    if (obstacle_ratio > 0.3f) {
        ROS_WARN("目标点周围障碍物密度: %.1f%%，使用扩展安全距离", obstacle_ratio * 100);
        return EXTENDED_SAFE_DISTANCE;
    }
    
    return DEFAULT_SAFE_DISTANCE;
}

void NavigationStateMachine::selectBestCluster() {
    if (detected_clusters_.empty()) {
        current_target_cluster_ = -1;
        return;
    }
    
    // 获取当前机器人位姿
    float current_x, current_y, current_yaw;
    if (!getRobotPose(current_x, current_y, current_yaw)) {
        ROS_WARN("无法获取当前位姿，使用默认选择");
        current_target_cluster_ = 0;
        return;
    }
    
    // 基于当前朝向重新排序
    std::vector<std::pair<size_t, float>> point_scores;
    
    for (size_t i = 0; i < detected_clusters_.size(); ++i) {
        float dx_to_robot = detected_clusters_[i].x - current_x;
        float dy_to_robot = detected_clusters_[i].y - current_y;
        float to_point_yaw = atan2(dy_to_robot, dx_to_robot);
        
        // 计算与当前朝向的角度差
        float angle_diff = fabs(to_point_yaw - current_yaw);
        if (angle_diff > M_PI) {
            angle_diff = 2 * M_PI - angle_diff;
        }
        
        // 评分：角度差越小，评分越高
        float direction_score = 1.0f - (angle_diff / M_PI);
        point_scores.push_back({i, direction_score});
        
        ROS_INFO("识别板[%zu]方向评分: 角度差=%.1f°, 得分=%.3f", 
                 i, angle_diff * 180 / M_PI, direction_score);
    }
    
    // 按评分排序
    std::sort(point_scores.begin(), point_scores.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // 重新排序簇
    std::vector<geometry_msgs::Point> sorted_clusters;
    std::vector<ClusterInfo> sorted_infos;

    for (const auto& item : point_scores) {
        sorted_clusters.push_back(detected_clusters_[item.first]);
        sorted_infos.push_back(detected_cluster_infos_[item.first]);
    }

    detected_clusters_ = sorted_clusters;
    detected_cluster_infos_ = sorted_infos;
    
    current_target_cluster_ = 0;
    
    ROS_INFO("=== PCA最终选择结果 ===");
    for (size_t i = 0; i < detected_clusters_.size(); ++i) {
        ROS_INFO("排序[%zu]: 位置(%.2f, %.2f), 原索引=%zu", 
                 i, detected_clusters_[i].x, detected_clusters_[i].y, 
                 point_scores[i].first);
    }
    
    ROS_INFO("选择最优识别板[0]: 位置(%.2f, %.2f)", 
             detected_clusters_[0].x, detected_clusters_[0].y);
}

bool NavigationStateMachine::callSimulationService() {
    try {
        // 创建服务请求
        service::Service srv;
        
        // 修改：使用 task_type 而不是 target_object
        srv.request.task_type = current_task_;  // 使用任务类型而不是物品
        
        ROS_INFO("调用A客户端 /task 服务，任务类型: %s", current_task_.c_str());
        
        // 调用服务
        if (simulation_service_client_.call(srv)) {
            if (srv.response.success) {
                simulation_result_ = srv.response.result;
                task_flags_.simulation_received = true;
                ROS_INFO("A客户端返回B服务器结果: %s", simulation_result_.c_str());
                
                // 修改：在这里更新代价计算，使用实际的仿真结果
                updateCostCalculation(picked_object_, simulation_result_);
                
                return true;
            } else {
                ROS_WARN("A客户端返回失败: %s", srv.response.result.c_str());
                return false;
            }
        } else {
            ROS_WARN("调用A客户端服务失败");
            return false;
        }
    }
    catch (const std::exception& e) {
        ROS_ERROR("调用A客户端异常: %s", e.what());
        return false;
    }
}

std::string NavigationStateMachine::generatePurchaseReport(double payment, double change) {
    std::stringstream report;
    
    report << "本次采购货物为" << picked_object_;
    report << "，价格" << getItemPrice(picked_object_) << "元";
    report << "，总计花费" << total_cost_ << "元";
    report << "，支付20元";
    report << "，需找零" << change << "元";
    
    return report.str();
}

double NavigationStateMachine::getItemPrice(const std::string& object) {
    std::map<std::string, double> price_map = {
        {"苹果", 4.0}, {"香蕉", 2.0}, {"西瓜", 5.0}, {"辣椒", 2.0},
        {"番茄", 5.0}, {"土豆", 2.0}, {"牛奶", 5.0}, {"蛋糕", 10.0}, {"可乐", 3.0}
    };
    
    auto it = price_map.find(object);
    return (it != price_map.end()) ? it->second : 5.0;
}

void NavigationStateMachine::printPurchaseDetails(double payment, double change) {
    ROS_INFO("========== 采购详情 ==========");
    ROS_INFO("采购物品: %s", picked_object_.c_str());
    ROS_INFO("物品价格: %.1f 元", getItemPrice(picked_object_));
    ROS_INFO("总计花费: %.1f 元", total_cost_);
    ROS_INFO("支付金额: %.1f 元", payment);
    ROS_INFO("找零金额: %.1f 元", change);
    
    if (!purchase_history_.empty()) {
        ROS_INFO("采购记录:");
        for (const auto& record : purchase_history_) {
            ROS_INFO("  - %s: %.1f元 (位置:%s)",
                    record.object.c_str(), record.price, record.room.c_str());
        }
    }
    ROS_INFO("==============================");
}

