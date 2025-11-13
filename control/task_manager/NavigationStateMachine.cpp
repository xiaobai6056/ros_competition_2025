#include "NavigationStateMachine.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <cmath>
#include <limits>
#include <sstream>
#include <std_srvs/Trigger.h>
#include <std_msgs/String.h>

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
      scan_robot_yaw_(0.0f)
{
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
    
    // 初始化服务客户端
    qr_service_client_ = nh_.serviceClient<std_srvs::Trigger>("/qr_recognition");
    object_service_client_ = nh_.serviceClient<std_srvs::Trigger>("/object_recognition");
    
    // 加载导航点
    loadNavigationPoints();
    
    costmap_sub_ = nh_.subscribe("/move_base/global_costmap/costmap", 1, 
                                    &NavigationStateMachine::costmapCallback, this);
    costmap_updated_ = false;

    ROS_INFO("导航状态机初始化完成 - 集成costmap版本");

}

void NavigationStateMachine::execute() {
    switch(current_state_) {
        case RobotState::INIT: handleInitState(); break;
        case RobotState::MOVE_TO_QR_ZONE: handleMoveToQRZone(); break;
        case RobotState::WAITING_QR_SERVICE: handleWaitingQRService(); break;
        case RobotState::MOVE_TO_PICK_ZONE: handleMoveToPickZone(); break;
        case RobotState::SCANNING_BOARDS: handleScanningBoards(); break;
        case RobotState::NAVIGATING_TO_BOARD: handleNavigatingToBoard(); break;
        case RobotState::WAITING_VISUAL: handleWaitingVisual(); break;
        case RobotState::OBJECT_CONFIRMED: handleObjectConfirmed(); break;
        case RobotState::MOVE_TO_WAIT_ZONE: handleMoveToWaitZone(); break;
        case RobotState::WAITING_SIMULATION: handleWaitingSimulation(); break;
        case RobotState::MOVE_TO_TRAFFIC_ZONE: handleMoveToTrafficZone(); break;
        case RobotState::WAITING_TRAFFIC: handleWaitingTraffic(); break;
        case RobotState::MOVE_TO_INTERSECTION: handleMoveToIntersection(); break;
        case RobotState::NAVIGATE_TO_FINISH: handleNavigateToFinish(); break;
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
}

void NavigationStateMachine::handleWaitingQRService() {
    if (!qr_service_called_) {
        ROS_INFO("[WAITING_QR_SERVICE] 调用二维码识别服务");
        ros::Duration(0.5).sleep();
        if (callQRService()) {
            qr_service_called_ = true;
            service_call_time_ = ros::Time::now();
        } else {
            ROS_WARN("二维码服务调用失败，0.5秒后重试");
            ros::Duration(0.5).sleep();
        }
    } else {
        // 服务已调用，等待导航完成或其他处理
        ROS_INFO_THROTTLE(5, "[WAITING_QR_SERVICE] 等待二维码识别结果...");
    }
}

void NavigationStateMachine::handleMoveToPickZone() {
    if (!task_flags_.pick_goal_sent) {
        ROS_INFO("[MOVE_TO_PICK_ZONE] 前往拣货区");
        speak("正在前往拣货区");
        sendNavigationGoal("pick_zone");
        task_flags_.pick_goal_sent = true;
    }
}

void NavigationStateMachine::handleScanningBoards() {
    ROS_INFO("[SCANNING_BOARDS] 开始扫描识别板");
    speak("正在扫描识别板位置");
    
    // 在开始扫描前获取一次机器人位姿，避免在回调中频繁查询TF
    if (!getRobotPose(scan_robot_x_, scan_robot_y_, scan_robot_yaw_)) {
        ROS_WARN("无法获取机器人位姿，延迟扫描");
        return;
    }
    
    ROS_INFO("扫描起始位姿: (%.2f, %.2f, %.1f°)", 
             scan_robot_x_, scan_robot_y_, scan_robot_yaw_ * 180 / M_PI);
    
    // 触发激光检测
    stopMoving();
    current_target_cluster_ = -1;
    clusters_detected_ = false;
    detected_clusters_.clear();
    
    // 设置超时机制
    scan_start_time_ = ros::Time::now();
    
    ROS_INFO("开始激光扫描，等待检测结果...");
}

void NavigationStateMachine::handleNavigatingToBoard() {
    if (clusters_detected_ && current_target_cluster_ >= 0 && 
        current_target_cluster_ < detected_clusters_.size() && !moving_to_cluster_) {
        
        geometry_msgs::Point target_point = detected_clusters_[current_target_cluster_];
        float board_yaw = detected_cluster_infos_[current_target_cluster_].board_yaw;
        
        // 机器人朝向：正对板子
        float robot_target_yaw = board_yaw;
        
        ROS_INFO("[NAVIGATING_TO_BOARD] 前往第 %d 个识别板", current_target_cluster_ + 1);
        ROS_INFO("  目标位置: (%.2f, %.2f)", target_point.x, target_point.y);
        ROS_INFO("  板子朝向: %.1f°", board_yaw * 180 / M_PI);
        ROS_INFO("  机器人目标朝向: %.1f°", robot_target_yaw * 180 / M_PI);
        
        // 发送导航目标
        move_base_msgs::MoveBaseGoal goal;
        goal.target_pose = createPose(target_point.x, target_point.y, robot_target_yaw);
        goal.target_pose.header.stamp = ros::Time::now();
        goal.target_pose.header.frame_id = "map";
        
        action_client_.sendGoal(goal,
            boost::bind(&NavigationStateMachine::clusterArrivedCallback, this, _1, _2),
            boost::bind(&NavigationStateMachine::navActiveCallback, this),
            boost::bind(&NavigationStateMachine::navFeedbackCallback, this, _1));
        
        moving_to_cluster_ = true;
        task_flags_.navigation_in_progress = true;
    }
}

void NavigationStateMachine::handleWaitingVisual() {
    static bool first_entered = true;
    static ros::Time wait_start_time;
    static ros::Time detection_start_time;
    static bool initial_delay_passed = false;
    static std::string last_detected_object;
    static int same_object_count = 0;
    
    if (first_entered) {
        ROS_INFO("[WAITING_VISUAL] 到达识别板位置，等待视觉系统稳定");

        // ========== 添加视觉状态重置 ==========
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
        // ========== 重置结束 ==========

        wait_start_time = ros::Time::now();
        detection_start_time = ros::Time::now() + ros::Duration(1.5); // 1.5秒后开始正式检测
        first_entered = false;
        initial_delay_passed = false;
        same_object_count = 0;
        last_detected_object = "";
        return;
    }
    
    // 检查是否过了初始延迟期
    if (!initial_delay_passed) {
        if (ros::Time::now() < detection_start_time) {
            ROS_INFO_THROTTLE(1, "[WAITING_VISUAL] 等待视觉系统稳定...");
            return;
        } else {
            ROS_INFO("[WAITING_VISUAL] 视觉系统稳定，开始正式检测");
            initial_delay_passed = true;
        }
    }
    
    // 持续调用识别服务
    std_srvs::Trigger srv;
    if (object_service_client_.call(srv)) {
        // 移除对srv.response.success的检查，因为视觉节点总是返回success
        std::string current_object = srv.response.message;
        
        // 处理"继续检测"标记
        if (current_object == "CONTINUE_DETECTING") {
            ROS_INFO_THROTTLE(2, "视觉系统正在检测中，继续等待...");
            ros::Duration(0.3).sleep();
            return;
        }
        
        // 处理"未检测到物体"标记
        if (current_object == "NO_OBJECT_DETECTED") {
            ROS_INFO_THROTTLE(2, "当前未检测到任何物体，继续等待...");
            same_object_count = 0;
            last_detected_object = "";
            ros::Duration(0.3).sleep();
            return;
        }
        
        // 检查是否与任务匹配
        if (current_object.find("WARN:") != 0) {
            // 匹配的物体
            if (current_object == last_detected_object) {
                same_object_count++;
                ROS_INFO("持续检测到: %s, 连续计数: %d", current_object.c_str(), same_object_count);
            } else {
                same_object_count = 1;
                last_detected_object = current_object;
                ROS_INFO("检测到新物体: %s, 开始计数", current_object.c_str());
            }
            
            // 连续5次检测到同一物体确认识别成功
            if (same_object_count >= 5) {
                ROS_INFO("物体识别确认: %s", current_object.c_str());
                picked_object_ = current_object;
                speak("识别到" + current_object);
                setState(RobotState::OBJECT_CONFIRMED);
                first_entered = true;
            }
        } else {
            // 不匹配的物体 - 立即切换到下一个识别板
            std::string mismatched_object = current_object.substr(5);
            ROS_WARN("识别到不匹配物体: %s，立即切换到下一个识别板", mismatched_object.c_str());
            ros::Duration(0.2).sleep();
            moveToNextCluster();
            first_entered = true;
        }
    } else {
        // 服务调用失败时，只记录错误但不切换识别板，等待超时处理
        ROS_ERROR_THROTTLE(2, "无法调用物体识别服务，等待重试...");
        ros::Duration(0.3).sleep();
        return;  // 不切换识别板，继续等待
    }
    
    // 超时处理（15秒内没有确认）
    if ((ros::Time::now() - wait_start_time).toSec() > 15.0) {
        ROS_WARN("视觉识别超时，前往下一个识别板");
        moveToNextCluster();
        first_entered = true;
    }
    
    // 限制调用频率
    ros::Duration(0.3).sleep();
}

void NavigationStateMachine::handleObjectConfirmed() {
    ROS_INFO("[OBJECT_CONFIRMED] 物体确认: %s", picked_object_.c_str());
    speak("我已取到" + picked_object_);
    task_flags_.object_picked = true;
    updateCostCalculation(picked_object_);
    
    // 重置状态
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
    ROS_INFO_THROTTLE(2, "[MOVE_TO_WAIT_ZONE] 等待导航完成...");
}

void NavigationStateMachine::handleWaitingSimulation() {
    if (task_flags_.simulation_received) {
        ROS_INFO("[WAITING_SIMULATION] 收到仿真结果: %s", simulation_result_.c_str());
        speak("仿真任务已完成，目标货物位于" + simulation_result_ + "房间");
        task_flags_.simulation_received = false;
        setState(RobotState::MOVE_TO_TRAFFIC_ZONE);
    } else {
        ROS_INFO_THROTTLE(5, "[WAITING_SIMULATION] 等待仿真任务完成...");
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

void NavigationStateMachine::handleWaitingTraffic() {
    if (task_flags_.traffic_received) {
        ROS_INFO("[WAITING_TRAFFIC] 收到路牌识别结果: %s", traffic_result_.c_str());
        speak("路口" + traffic_result_ + "可通过");
        task_flags_.traffic_received = false;
        setState(RobotState::MOVE_TO_INTERSECTION);
    } else {
        ROS_INFO_THROTTLE(5, "[WAITING_TRAFFIC] 等待路牌识别结果...");
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

void NavigationStateMachine::handleNavigateToFinish() {
    if (!task_flags_.finish_goal_sent) {
        ROS_INFO("[NAVIGATE_TO_FINISH] 从路口巡线前往终点");
        
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
    ROS_INFO_THROTTLE(2, "[NAVIGATE_TO_FINISH] 等待导航完成...");
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
    ROS_ERROR("[ERROR] 进入错误状态，尝试恢复...");
    speak("导航出现问题，尝试恢复");
    
    // 取消所有目标
    action_client_.cancelAllGoals();
    stopMoving();
    
    // 根据当前任务状态决定恢复策略
    if (task_flags_.object_picked) {
        ROS_INFO("恢复：前往等待区");
        setState(RobotState::MOVE_TO_WAIT_ZONE);
    } else {
        ROS_INFO("恢复：重新寻找物体");
        setState(RobotState::SCANNING_BOARDS);
    }
    
    ros::Duration(1.0).sleep();
}

// ========== 激光雷达相关函数 ==========

void NavigationStateMachine::laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
    // 障碍物检测
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
    
    // 在SCANNING_BOARDS状态时检测物体簇
    if (current_state_ == RobotState::SCANNING_BOARDS && !clusters_detected_) {
        // 使用预先获取的机器人位姿，避免在回调中频繁查询TF
        detectObjectClusters(msg);
        
        // 检测完成后直接触发状态转换
        if (clusters_detected_ && !detected_clusters_.empty()) {
            ROS_INFO("检测完成，发现 %zu 个识别板，开始选择最佳目标", detected_clusters_.size());
            
            // 选择最佳簇
            selectBestCluster();
            
            if (current_target_cluster_ >= 0 && current_target_cluster_ < detected_clusters_.size()) {
                ROS_INFO("选择第 %d 个识别板，切换到导航状态", current_target_cluster_ + 1);
                setState(RobotState::NAVIGATING_TO_BOARD);
            }
        }
        
        // 超时处理
        if ((ros::Time::now() - scan_start_time_).toSec() > 5.0) {
            ROS_WARN("激光扫描超时，继续执行");
            if (!clusters_detected_) {
                speak("未找到识别板，继续执行");
                setState(RobotState::MOVE_TO_WAIT_ZONE);
            }
        }
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

    ROS_INFO("=== 动态聚类识别板检测 ===");
    
    std::vector<std::vector<int>> clusters;
    std::vector<int> current_cluster;
    
    const float MAX_DISTANCE_JUMP = 0.1f;
    const float MIN_VALID_RANGE = 0.4f;
    const float MAX_VALID_RANGE = 4.0f;
    
    // 动态聚类算法
    for (size_t i = 0; i < scan->ranges.size(); ++i) {
        float dist = scan->ranges[i];
        
        // 过滤无效点
        if (!std::isfinite(dist) || dist < MIN_VALID_RANGE || dist > MAX_VALID_RANGE) {
            if (!current_cluster.empty() && current_cluster.size() >= 5) {
                clusters.push_back(current_cluster);
            }
            current_cluster.clear();
            continue;
        }
        
        if (current_cluster.empty()) {
            current_cluster.push_back(i);
            continue;
        }
        
        // 计算与前一点的实际物理距离
        int prev_idx = current_cluster.back();
        float prev_dist = scan->ranges[prev_idx];
        float prev_angle = scan->angle_min + prev_idx * scan->angle_increment;
        float curr_angle = scan->angle_min + i * scan->angle_increment;
        
        float x1 = prev_dist * cos(prev_angle);
        float y1 = prev_dist * sin(prev_angle);
        float x2 = dist * cos(curr_angle);
        float y2 = dist * sin(curr_angle);
        float physical_distance = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
        
        // 根据物理距离判断是否属于同一物体
        if (physical_distance < MAX_DISTANCE_JUMP) {
            current_cluster.push_back(i);
        } else {
            // 结束当前簇
            if (current_cluster.size() >= 12) {
                clusters.push_back(current_cluster);
            }
            current_cluster.clear();
            current_cluster.push_back(i);
        }
    }
    
    // 处理最后一个簇
    if (!current_cluster.empty() && current_cluster.size() >= 8) {
        clusters.push_back(current_cluster);
    }
    
    ROS_INFO("动态聚类结果: %zu个候选簇", clusters.size());
    
    // 临时存储所有有效识别板
    std::vector<geometry_msgs::Point> temp_clusters;
    std::vector<ClusterInfo> temp_infos;
    
    for (const auto& cluster : clusters) {
        ClusterInfo cluster_info = calculateClusterInfo(cluster, scan);
        float length = calculateBoardLength(cluster, scan);
        
        ROS_INFO("候选簇: 大小=%zu, 长度=%.3fm, 距离=%.2fm, 中心点=(%.2f, %.2f), 朝向=%.1f°", 
                cluster.size(), length, cluster_info.average_distance, 
                cluster_info.center.x, cluster_info.center.y, cluster_info.board_yaw * 180 / M_PI);
        
        if (isValidObjectCluster(cluster_info, cluster, scan)) {
            geometry_msgs::Point safe_target = calculateSafeTarget(cluster_info);
            temp_clusters.push_back(safe_target);
            temp_infos.push_back(cluster_info);
            
            ROS_INFO("✅ 识别板: 原始位置(%.2f,%.2f) -> 安全位置(%.2f,%.2f), 朝向=%.1f°, 长度=%.3fm", 
                    cluster_info.center.x, cluster_info.center.y,
                    safe_target.x, safe_target.y, cluster_info.board_yaw * 180 / M_PI, length);
        }
    }
    
    // ========== 精简重复检测 ==========
    std::vector<bool> keep_flag(temp_clusters.size(), true);
    const float DUPLICATE_DISTANCE = 0.3f;  // 重复判断距离阈值
    
    for (size_t i = 0; i < temp_clusters.size(); ++i) {
        if (!keep_flag[i]) continue;
        
        for (size_t j = i + 1; j < temp_clusters.size(); ++j) {
            if (!keep_flag[j]) continue;
            
            float dx = temp_clusters[i].x - temp_clusters[j].x;
            float dy = temp_clusters[i].y - temp_clusters[j].y;
            float distance = sqrt(dx*dx + dy*dy);
            
            if (distance < DUPLICATE_DISTANCE) {
                // 保留距离机器人更近的点
                float dist_i = sqrt(pow(temp_clusters[i].x - scan_robot_x_, 2) + 
                                   pow(temp_clusters[i].y - scan_robot_y_, 2));
                float dist_j = sqrt(pow(temp_clusters[j].x - scan_robot_x_, 2) + 
                                   pow(temp_clusters[j].y - scan_robot_y_, 2));
                
                if (dist_i < dist_j) {
                    keep_flag[j] = false;
                    ROS_INFO("过滤重复识别板: 保留索引%zu(距离%.2fm), 过滤索引%zu(距离%.2fm)", 
                             i, dist_i, j, dist_j);
                } else {
                    keep_flag[i] = false;
                    ROS_INFO("过滤重复识别板: 保留索引%zu(距离%.2fm), 过滤索引%zu(距离%.2fm)", 
                             j, dist_j, i, dist_i);
                    break;
                }
            }
        }
    }
    
    // 收集非重复的识别板
    for (size_t i = 0; i < temp_clusters.size(); ++i) {
        if (keep_flag[i]) {
            detected_clusters_.push_back(temp_clusters[i]);
            detected_cluster_infos_.push_back(temp_infos[i]);
        }
    }
    
    // ========== 精简优先级排序 ==========
    if (!detected_clusters_.empty()) {
        // 计算每个安全点相对于机器人的方向向量
        std::vector<std::pair<size_t, float>> point_scores;
        
        for (size_t i = 0; i < detected_clusters_.size(); ++i) {
            float dx = detected_clusters_[i].x - scan_robot_x_;
            float dy = detected_clusters_[i].y - scan_robot_y_;
            float to_point_yaw = atan2(dy, dx);
            
            // 计算与机器人朝向的夹角（绝对值）
            float angle_diff = fabs(to_point_yaw - scan_robot_yaw_);
            if (angle_diff > M_PI) angle_diff = 2 * M_PI - angle_diff;
            
            // 评分：夹角越小得分越高（越在正前方）
            float score = 1.0f - (angle_diff / M_PI);
            point_scores.push_back({i, score});
            
            ROS_INFO("安全点[%zu]评分: 朝向差=%.1f°, 得分=%.2f", 
                     i, angle_diff * 180/M_PI, score);
        }
        
        // 按得分降序排序
        std::sort(point_scores.begin(), point_scores.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // 重新排列结果
        std::vector<geometry_msgs::Point> sorted_clusters;
        std::vector<ClusterInfo> sorted_infos;
        
        for (const auto& item : point_scores) {
            sorted_clusters.push_back(detected_clusters_[item.first]);
            sorted_infos.push_back(detected_cluster_infos_[item.first]);
        }
        
        detected_clusters_ = sorted_clusters;
        detected_cluster_infos_ = sorted_infos;
        
        ROS_INFO("安全点排序完成，优先检查正前方的识别板");
    }
    
    clusters_detected_ = !detected_clusters_.empty();
    ROS_INFO("检测完成: 发现 %zu 个识别板", detected_clusters_.size());
}

NavigationStateMachine::ClusterInfo NavigationStateMachine::calculateClusterInfo(const std::vector<int>& cluster, 
                                                                                const sensor_msgs::LaserScan::ConstPtr& scan) {
    ClusterInfo info;
    float sum_x = 0.0f, sum_y = 0.0f;
    float sum_dist = 0.0f;
    
    // 使用扫描时缓存的机器人位姿，避免频繁TF查询
    float robot_x = scan_robot_x_;
    float robot_y = scan_robot_y_;
    float robot_yaw = scan_robot_yaw_;
    
    // 存储所有点的全局坐标用于线性拟合
    std::vector<float> global_x_points, global_y_points;
    
    for (int idx : cluster) {
        float dist = scan->ranges[idx];
        float angle = scan->angle_min + idx * scan->angle_increment;
        
        // 极坐标转机器人坐标系
        float local_x = dist * cos(angle);
        float local_y = dist * sin(angle);
        
        // 转全局坐标系
        float global_x = robot_x + local_x * cos(robot_yaw) - local_y * sin(robot_yaw);
        float global_y = robot_y + local_x * sin(robot_yaw) + local_y * cos(robot_yaw);
        
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
    
    // ========== 线性拟合计算板子朝向 ==========
    if (global_x_points.size() >= 5) {
        float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
        int n = global_x_points.size();
        
        for (int i = 0; i < n; ++i) {
            sum_x += global_x_points[i];
            sum_y += global_y_points[i];
            sum_xy += global_x_points[i] * global_y_points[i];
            sum_x2 += global_x_points[i] * global_x_points[i];
        }
        
        float denominator = n * sum_x2 - sum_x * sum_x;
        if (fabs(denominator) > 1e-6) {
            float k = (n * sum_xy - sum_x * sum_y) / denominator;
            
            // 板子朝向是垂直于拟合直线的方向（法线方向）
            info.board_yaw = atan2(1.0f, -k);
            
            // 确保朝向指向机器人（板子正面对着机器人）
            float dx = info.center.x - robot_x;
            float dy = info.center.y - robot_y;
            float dot_product = cos(info.board_yaw) * dx + sin(info.board_yaw) * dy;
            if (dot_product < 0) {
                // 如果点积为负，说明朝向背对机器人，需要翻转180度
                info.board_yaw += M_PI;
                ROS_INFO("板子朝向翻转180度以面向机器人");
            }
            
            // 归一化到 [-π, π]
            while (info.board_yaw > M_PI) info.board_yaw -= 2 * M_PI;
            while (info.board_yaw < -M_PI) info.board_yaw += 2 * M_PI;
            
            ROS_INFO("板子朝向: 拟合斜率=%.3f, 最终朝向=%.1f°", k, info.board_yaw * 180 / M_PI);
        } else {
            info.board_yaw = M_PI / 2;
            ROS_WARN("板子朝向: 垂直线，使用默认朝向90°");
        }
    } else {
        float dx = info.center.x - robot_x;
        float dy = info.center.y - robot_y;
        info.board_yaw = atan2(dy, dx);
        ROS_WARN("点数不足(%zu)，使用朝向机器人方向: %.1f°", global_x_points.size(), info.board_yaw * 180 / M_PI);
    }
    
    return info;
}

bool NavigationStateMachine::isValidObjectCluster(const ClusterInfo& cluster_info, 
                                                const std::vector<int>& cluster,
                                                const sensor_msgs::LaserScan::ConstPtr& scan) {
    float estimated_length = calculateBoardLength(cluster, scan);
    float angular_width_deg = cluster_info.angular_width * 180/M_PI;
    
    ROS_INFO("物体特征: 长度=%.3fm, 宽度=%.1f°, 距离=%.2fm, 大小=%zu", 
             estimated_length, angular_width_deg, cluster_info.average_distance, cluster.size());
    
    // 验证条件
    if (estimated_length < 0.2f || estimated_length > 0.7f) {
        ROS_DEBUG("长度不合适: %.3fm", estimated_length);
        return false;
    }
    
    if (cluster_info.average_distance < 0.8f || cluster_info.average_distance > 3.0f) {
        ROS_DEBUG("距离不合适: %.2fm", cluster_info.average_distance);
        return false;
    }
    
    if (cluster.size() < 8) {
        ROS_DEBUG("簇太小: %zu点", cluster.size());
        return false;
    }
    
    // 角度跨度过滤（避免墙壁）
    if (cluster_info.angular_width > 0.4f) {
        ROS_DEBUG("角度跨度太大: %.1f°", angular_width_deg);
        return false;
    }
    
    ROS_INFO("✅ 接受物体簇");
    return true;
}

void NavigationStateMachine::selectBestCluster() {
    if (detected_clusters_.empty()) {
        current_target_cluster_ = -1;
        return;
    }
    
    // 强制从第一个识别板开始按顺序检查
    current_target_cluster_ = 0;
    
    // 记录所有识别板位置用于调试
    for (size_t i = 0; i < detected_clusters_.size(); ++i) {
        ROS_INFO("识别板[%zu]: 位置(%.2f, %.2f)", 
                 i, detected_clusters_[i].x, detected_clusters_[i].y);
    }
    
    ROS_INFO("选择第一个识别板开始顺序检查: 位置(%.2f, %.2f)", 
             detected_clusters_[0].x, detected_clusters_[0].y);
}

float NavigationStateMachine::calculateBoardLength(const std::vector<int>& cluster, 
                                                 const sensor_msgs::LaserScan::ConstPtr& scan) {
    if (cluster.size() < 2) return 0.0f;
    
    // 使用扫描时缓存的机器人位姿
    float robot_x = scan_robot_x_;
    float robot_y = scan_robot_y_;
    float robot_yaw = scan_robot_yaw_;
    
    // 计算簇的起点和终点在全局坐标系中的位置
    int first_idx = cluster.front();
    int last_idx = cluster.back();
    
    // 起点坐标
    float dist1 = scan->ranges[first_idx];
    float angle1 = scan->angle_min + first_idx * scan->angle_increment;
    float global_x1 = robot_x + dist1 * cos(robot_yaw + angle1);
    float global_y1 = robot_y + dist1 * sin(robot_yaw + angle1);
    
    // 终点坐标  
    float dist2 = scan->ranges[last_idx];
    float angle2 = scan->angle_min + last_idx * scan->angle_increment;
    float global_x2 = robot_x + dist2 * cos(robot_yaw + angle2);
    float global_y2 = robot_y + dist2 * sin(robot_yaw + angle2);
    
    // 计算直线距离
    float dx = global_x2 - global_x1;
    float dy = global_y2 - global_y1;
    float straight_line_distance = sqrt(dx*dx + dy*dy);
    
    ROS_INFO("端点距离: 起点(%.2f,%.2f) 终点(%.2f,%.2f) 直线长度=%.3fm", 
             global_x1, global_y1, global_x2, global_y2, straight_line_distance);
    
    return straight_line_distance;
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
    
    // 添加重试机制
    int retry_count = 0;
    const int max_retries = 3;
    
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
    traffic_result_ = msg->data;
    task_flags_.traffic_received = true;
    ROS_INFO("收到路牌识别结果: %s", traffic_result_.c_str());
}

// ========== ActionLib回调函数 ==========

void NavigationStateMachine::navDoneCallback(const actionlib::SimpleClientGoalState& state,
                                            const move_base_msgs::MoveBaseResultConstPtr& result) {
    task_flags_.navigation_in_progress = false;

    ROS_INFO("导航完成回调 - 状态: %s, 目标点: %s", 
             state.toString().c_str(), current_goal_point_.c_str());

    if (state == actionlib::SimpleClientGoalState::SUCCEEDED) {
        ROS_INFO("导航目标成功到达: %s", current_goal_point_.c_str());
        
        switch(current_state_) {
            case RobotState::MOVE_TO_QR_ZONE:
                setState(RobotState::WAITING_QR_SERVICE);
                break;
            case RobotState::MOVE_TO_PICK_ZONE:
                setState(RobotState::SCANNING_BOARDS);
                break;
            case RobotState::MOVE_TO_WAIT_ZONE:
                setState(RobotState::WAITING_SIMULATION);
                break;
            case RobotState::MOVE_TO_TRAFFIC_ZONE:
                setState(RobotState::WAITING_TRAFFIC);
                break;
            case RobotState::MOVE_TO_INTERSECTION:
                setState(RobotState::NAVIGATE_TO_FINISH);
                break;
            case RobotState::NAVIGATE_TO_FINISH:
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

void NavigationStateMachine::navActiveCallback() {
    ROS_INFO("导航目标已激活: %s", current_goal_point_.c_str());
}

void NavigationStateMachine::navFeedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback) {
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
    // 在关键状态转换时验证TF数据
    if (new_state == RobotState::SCANNING_BOARDS || 
        new_state == RobotState::NAVIGATING_TO_BOARD) {
        
        if (!validateTFData()) {
            ROS_WARN("TF数据不完整，延迟状态转换");
            return;
        }
    }
    
    ROS_INFO("状态转换: %d -> %d", 
             static_cast<int>(current_state_), 
             static_cast<int>(new_state));
    current_state_ = new_state;
}

void NavigationStateMachine::loadNavigationPoints() {
    navigation_points_["qr_zone"] = createPose(1.35, 0.92, 3.14);
    navigation_points_["pick_zone"] = createPose(1.7, 5.45, 0.0);
    navigation_points_["wait_zone"] = createPose(1.63, 6.34, 0.0);
    navigation_points_["traffic_zone"] = createPose(4.9, 6.4, 1.57); 
    navigation_points_["intersection_A"] = createPose(4.2, 4.3, -1.57);
    navigation_points_["intersection_B"] = createPose(7.3, 4.6, -1.57);
    navigation_points_["finish_zone_A"] = createPose(4.9, 0.4, -1.57);
    navigation_points_["finish_zone_B"] = createPose(6.5, 0.4, -1.57);

    // 测试用
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
