#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Point.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <nav_msgs/OccupancyGrid.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <limits>
#include <iomanip>

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

class BoardDetectionNavigator {
private:
    ros::NodeHandle nh_;
    MoveBaseClient ac_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // 订阅器
    ros::Subscriber laser_sub_;
    ros::Subscriber costmap_sub_;

    sensor_msgs::LaserScan::ConstPtr latest_scan_;
    
    // PCA结果结构
    struct PCAResult {
        float length;           // 板子长度（第一主成分方向）
        float orientation;      // 板子朝向（弧度）
        float confidence;       // 置信度（第一特征值占比）
        geometry_msgs::Point start_point;  // 投影起点
        geometry_msgs::Point end_point;    // 投影终点
        
        PCAResult() : length(0.0f), orientation(0.0f), confidence(0.0f) {}
    };
    
    // 参数
    struct DetectionParams {
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
        
        // 安全距离参数
        float default_safe_distance = 0.5f;
        float extended_safe_distance = 0.5f;
        float waypoint_switch_distance = 0.8f;
    } params_;
    
    // 检测状态
    struct ClusterInfo {
        geometry_msgs::Point center;
        float average_distance;
        float board_yaw;
        float angular_width;
        size_t size;
        float length;
        float pca_confidence;
        std::string debug_info;
        
        ClusterInfo() : average_distance(0.0f), board_yaw(0.0f), angular_width(0.0f), 
                       size(0), length(0.0f), pca_confidence(0.0f) {}
    };
    
    std::vector<geometry_msgs::Point> detected_clusters_;
    std::vector<ClusterInfo> detected_cluster_infos_;
    nav_msgs::OccupancyGrid current_costmap_;
    bool costmap_updated_ = false;
    bool clusters_detected_ = false;
    
    // 机器人位姿
    float robot_x_ = 0.0f, robot_y_ = 0.0f, robot_yaw_ = 0.0f;

public:
    BoardDetectionNavigator() : 
        ac_("move_base", true),
        tf_listener_(tf_buffer_)
    {
        // 等待action server
        if(!ac_.waitForServer(ros::Duration(5.0))) {
            ROS_ERROR("无法连接到move_base action server");
        } else {
            ROS_INFO("成功连接到move_base action server");
        }
        
        // 初始化订阅器
        laser_sub_ = nh_.subscribe("/scan", 1, &BoardDetectionNavigator::laserCallback, this);
        costmap_sub_ = nh_.subscribe("/move_base/global_costmap/costmap", 1, 
                                   &BoardDetectionNavigator::costmapCallback, this);
        
        // 加载参数
        loadParameters();
        
        ROS_INFO("PCA板子检测导航节点初始化完成");
    }
    
    void loadParameters() {
        // 直接使用代码内设置的默认参数，不从ROS参数服务器加载
        ROS_INFO("使用PCA检测默认参数:");
        ROS_INFO("  聚类: 距离跳跃=%.2f, 有效范围=%.1f-%.1f, 簇大小=%d-%d", 
                params_.max_distance_jump, params_.min_valid_range, params_.max_valid_range,
                params_.min_cluster_size, params_.max_cluster_size);
        ROS_INFO("  过滤: 板长=%.2f-%.2f, 角度跨度<%.1f, 去重距离=%.2f",
                params_.min_board_length, params_.max_board_length, params_.max_angular_width,
                params_.duplicate_distance);
        ROS_INFO("  PCA: 最小置信度=%.2f, 最大距离标准差=%.2f",
                params_.min_pca_confidence, params_.max_distance_std);
        ROS_INFO("  安全: 默认=%.2f, 扩展=%.2f, 航点切换=%.2f",
                params_.default_safe_distance, params_.extended_safe_distance, 
                params_.waypoint_switch_distance);
        
        // 显示当前使用的参数值
        std::cout << "\n当前使用的PCA检测参数:" << std::endl;
        std::cout << "  min_cluster_size: " << params_.min_cluster_size << std::endl;
        std::cout << "  min_board_length: " << params_.min_board_length << std::endl;
        std::cout << "  max_board_length: " << params_.max_board_length << std::endl;
        std::cout << "  min_pca_confidence: " << params_.min_pca_confidence << std::endl;
        std::cout << "  max_distance_std: " << params_.max_distance_std << std::endl;
        std::cout << "  其他参数使用类定义中的默认值" << std::endl;
    }
    
    void run() {
        while(ros::ok()) {
            std::cout << "\n=== PCA板子检测导航系统 ===" << std::endl;
            std::cout << "1. 手动输入坐标导航" << std::endl;
            std::cout << "2. 扫描并检测识别板" << std::endl;
            std::cout << "3. 显示检测到的识别板" << std::endl;
            std::cout << "4. 导航到检测到的识别板" << std::endl;
            std::cout << "5. 重新加载参数" << std::endl;
            std::cout << "6. 退出" << std::endl;
            std::cout << "选择模式 (1-6): ";
            
            std::string choice;
            std::getline(std::cin, choice);
            
            if(choice == "1") {
                manualNavigation();
            } else if(choice == "2") {
                scanAndDetectBoards();
            } else if(choice == "3") {
                displayDetectedBoards();
            } else if(choice == "4") {
                navigateToDetectedBoard();
            } else if(choice == "5") {
                loadParameters();
            } else if(choice == "6") {
                break;
            } else {
                std::cout << "无效选择!" << std::endl;
            }
            
            ros::spinOnce();
        }
    }

private:
    void manualNavigation() {
        double x, y, yaw;
        
        std::cout << "\n=== 手动导航 ===" << std::endl;
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
        
        sendNavigationGoal(x, y, yaw);
    }
    
  void scanAndDetectBoards() {
    std::cout << "\n=== 开始PCA扫描识别板 ===" << std::endl;
    
    // 获取当前机器人位姿
    if(!getRobotPose()) {
        std::cout << "无法获取机器人位姿，请确保TF数据可用" << std::endl;
        return;
    }
    
    std::cout << "机器人当前位置: (" << std::fixed << std::setprecision(2) 
              << robot_x_ << ", " << robot_y_ << ", " 
              << std::setprecision(1) << (robot_yaw_ * 180 / M_PI) << "°)" << std::endl;
    
    // 清空之前的检测结果
    detected_clusters_.clear();
    detected_cluster_infos_.clear();
    clusters_detected_ = false;
    
    // ========== 修改：加入等待机制 ==========
    if (!latest_scan_) {
        std::cout << "等待激光数据..." << std::endl;
        
        // 等待0.1秒，给激光数据回调一些时间
        ros::Time start_time = ros::Time::now();
        while (ros::Time::now() - start_time < ros::Duration(0.1)) {
            ros::spinOnce();  // 处理回调
            if (latest_scan_) {
                break;  // 如果收到数据就退出等待
            }
            ros::Duration(0.01).sleep();  // 10ms间隔检查
        }
    }
    
    if (!latest_scan_) {
        std::cout << "没有可用的激光数据，请确保激光雷达正常工作" << std::endl;
        return;
    }
    
    std::cout << "使用最新激光数据，点数: " << latest_scan_->ranges.size() << std::endl;
    detectObjectClusters(latest_scan_);
    // ========== 修改结束 ==========
    
    if(detected_clusters_.empty()) {
        std::cout << "未检测到任何识别板，请调整参数或机器人位置" << std::endl;
    } else {
        std::cout << "PCA检测完成! 共发现 " << detected_clusters_.size() << " 个识别板" << std::endl;
        displayDetectedBoards();
    }
}
    
    void displayDetectedBoards() {
        if(detected_clusters_.empty()) {
            std::cout << "没有检测到的识别板，请先执行扫描" << std::endl;
            return;
        }
        
        std::cout << "\n=== PCA最终有效的识别板列表 ===" << std::endl;
        for(size_t i = 0; i < detected_clusters_.size(); ++i) {
            const auto& cluster = detected_clusters_[i];
            const auto& info = detected_cluster_infos_[i];
            
            std::cout << "识别板 " << (i+1) << ":" << std::endl;
            std::cout << "  ├─ 板子中心: (" << std::fixed << std::setprecision(2) 
                     << info.center.x << ", " << info.center.y << ")" << std::endl;
            std::cout << "  ├─ 安全目标点: (" << cluster.x << ", " << cluster.y << ")" << std::endl;
            std::cout << "  ├─ PCA长度: " << std::setprecision(3) << info.length << "m" << std::endl;
            std::cout << "  ├─ PCA朝向: " << std::setprecision(1) << (info.board_yaw * 180 / M_PI) << "°" << std::endl;
            std::cout << "  ├─ PCA置信度: " << std::setprecision(3) << info.pca_confidence << std::endl;
            std::cout << "  ├─ 距离: " << std::setprecision(1) << info.average_distance << "m" << std::endl;
            std::cout << "  ├─ 点数: " << info.size << std::endl;
            std::cout << "  └─ 角度跨度: " << std::setprecision(3) << info.angular_width << "rad" << std::endl;
            std::cout << std::endl;
        }
    }
    
    void navigateToDetectedBoard() {
        if(detected_clusters_.empty()) {
            std::cout << "没有检测到的识别板，请先执行扫描" << std::endl;
            return;
        }
        
        displayDetectedBoards();
        
        std::cout << "选择要导航的识别板 (1-" << detected_clusters_.size() << "): ";
        std::string choice_str;
        std::getline(std::cin, choice_str);
        
        try {
            int choice = std::stoi(choice_str) - 1;
            if(choice >= 0 && choice < detected_clusters_.size()) {
                const auto& safe_point = detected_clusters_[choice];
                const auto& info = detected_cluster_infos_[choice];
                
                std::cout << "导航到识别板 " << (choice+1) << ":" << std::endl;
                std::cout << "安全目标点: 板子中心(" << std::fixed << std::setprecision(2) 
                         << info.center.x << "," << info.center.y << ") -> "
                         << "安全点(" << safe_point.x << "," << safe_point.y << "), "
                         << "距离" << std::setprecision(1) << params_.default_safe_distance 
                         << "m, 朝向" << std::setprecision(1) 
                         << (info.board_yaw * 180 / M_PI) << "°" << std::endl;
                
                sendNavigationGoal(safe_point.x, safe_point.y, info.board_yaw * 180 / M_PI);
            } else {
                std::cout << "无效选择!" << std::endl;
            }
        } catch(const std::exception& e) {
            std::cout << "输入错误: " << e.what() << std::endl;
        }
    }
    
    void sendNavigationGoal(double x, double y, double yaw_deg) {
        double yaw_rad = yaw_deg * M_PI / 180.0;
        
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
        
        ROS_INFO("发送目标: x=%.2f, y=%.2f, yaw=%.1f°", x, y, yaw_deg);
        
        ac_.sendGoal(goal);
        
        std::cout << "等待结果中... (输入 'c' 取消，或按回车继续等待)" << std::endl;
        
        // 简单的非阻塞等待
        bool finished = false;
        while(!finished && ros::ok()) {
            finished = ac_.waitForResult(ros::Duration(0.5));
            
            // 检查是否有用户输入
            if(std::cin.rdbuf()->in_avail() > 0) {
                std::string cmd;
                std::getline(std::cin, cmd);
                if(cmd == "c" || cmd == "C") {
                    ac_.cancelGoal();
                    ROS_INFO("目标已取消");
                    break;
                }
            }
        }
        
        if(finished) {
            if(ac_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED) {
                ROS_INFO("任务完成!");
            } else {
                ROS_WARN("任务失败: %s", ac_.getState().toString().c_str());
            }
        }
    }
    
    // ========== PCA核心算法 ==========
    
    PCAResult computePCA(const std::vector<int>& cluster, 
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
            
            float global_x = robot_x_ + local_x * cos(robot_yaw_) - local_y * sin(robot_yaw_);
            float global_y = robot_y_ + local_x * sin(robot_yaw_) + local_y * cos(robot_yaw_);
            
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
    
    // ========== 激光雷达处理函数 ==========
    
 void laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
    if(!getRobotPose()) {
        ROS_WARN_THROTTLE(5, "无法获取机器人位姿，跳过数据保存");
        return;
    }
    
    // 保存最新激光数据（不立即处理）
    latest_scan_ = msg;
    ROS_DEBUG_THROTTLE(5, "更新最新激光数据，点数: %zu", latest_scan_->ranges.size());
}
    
   void detectObjectClusters(const sensor_msgs::LaserScan::ConstPtr& scan) {
    std::vector<std::vector<int>> clusters;
    std::vector<int> current_cluster;
    
    ROS_DEBUG("开始PCA聚类处理，激光数据点数: %zu", scan->ranges.size());
    
    // 动态聚类算法
    for(size_t i = 0; i < scan->ranges.size(); ++i) {
        float dist = scan->ranges[i];
        
        if(!std::isfinite(dist) || dist < params_.min_valid_range || dist > params_.max_valid_range) {
            if(!current_cluster.empty() && current_cluster.size() >= params_.min_cluster_size) {
                clusters.push_back(current_cluster);
                ROS_DEBUG("完成一个聚类，点数: %zu", current_cluster.size());
            } else if(!current_cluster.empty()) {
                ROS_DEBUG("丢弃小聚类，点数: %zu (小于最小要求 %d)", 
                         current_cluster.size(), params_.min_cluster_size);
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

            // ========== 新增：距离梯度检查 ==========
        float distance_gradient = fabs(dist - prev_dist);
        const float GRADIENT_THRESHOLD = 0.3f; // 宽松阈值
        
        if(distance_gradient > GRADIENT_THRESHOLD) {
            // 立即分割当前聚类
            if(!current_cluster.empty()) {
                clusters.push_back(current_cluster);
                ROS_DEBUG("梯度分割: 梯度=%.3f > %.3f, 点数=%zu", 
                        distance_gradient, GRADIENT_THRESHOLD, current_cluster.size());
            }
            current_cluster.clear();
            current_cluster.push_back(i); // 新聚类从当前点开始
            continue; // 跳过后续的物理距离检查
        }
        // ========== 距离梯度检查结束 ==========

        float prev_angle = scan->angle_min + prev_idx * scan->angle_increment;
        float curr_angle = scan->angle_min + i * scan->angle_increment;
        
        float x1 = prev_dist * cos(prev_angle);
        float y1 = prev_dist * sin(prev_angle);
        float x2 = dist * cos(curr_angle);
        float y2 = dist * sin(curr_angle);
        float physical_distance = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
        
        if(physical_distance < params_.max_distance_jump) {
            current_cluster.push_back(i);
        } else {
            if(current_cluster.size() >= params_.min_cluster_size) {
                clusters.push_back(current_cluster);
                ROS_DEBUG("完成一个聚类，点数: %zu", current_cluster.size());
            } else {
                ROS_DEBUG("丢弃小聚类，点数: %zu (小于最小要求 %d)", 
                         current_cluster.size(), params_.min_cluster_size);
            }
            current_cluster.clear();
            current_cluster.push_back(i);
        }
    }
    
    if(!current_cluster.empty() && current_cluster.size() >= params_.min_cluster_size) {
        clusters.push_back(current_cluster);
        ROS_DEBUG("完成最后一个聚类，点数: %zu", current_cluster.size());
    }
    
    ROS_INFO("PCA初步聚类完成，共 %zu 个聚类", clusters.size());
    
    // 处理每个簇 - 保存所有聚类信息（包括被过滤的）
    std::vector<geometry_msgs::Point> temp_clusters;
    std::vector<ClusterInfo> temp_infos;
    std::vector<ClusterInfo> all_cluster_infos;
    std::vector<bool> validity_flags;
    std::vector<std::string> filter_reasons;
    
    int valid_clusters = 0;
    for(size_t i = 0; i < clusters.size(); ++i) {
        const auto& cluster = clusters[i];
        ClusterInfo cluster_info = calculateClusterInfo(cluster, scan);
        
        std::string debug_info;
        bool isValid = isValidObjectCluster(cluster_info, cluster, scan, debug_info);
        
        // 保存所有聚类信息
        all_cluster_infos.push_back(cluster_info);
        validity_flags.push_back(isValid);
        filter_reasons.push_back(debug_info);
        
        if(isValid) {
            geometry_msgs::Point safe_target = calculateSafeTarget(cluster_info);
            temp_clusters.push_back(safe_target);
            cluster_info.debug_info = "PCA有效聚类";
            temp_infos.push_back(cluster_info);
            valid_clusters++;
            
            ROS_INFO("PCA聚类 %zu 有效: 长度=%.3fm, 距离=%.2fm, 置信度=%.3f", 
                    i+1, cluster_info.length, cluster_info.average_distance, 
                    cluster_info.pca_confidence);
        } else {
            ROS_WARN("PCA聚类 %zu 被过滤: %s", i+1, debug_info.c_str());
        }
    }
    
    // 显示所有检测到的板子信息（包括被过滤的）
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "            PCA所有检测到的物体（共 " << all_cluster_infos.size() << " 个）" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    for(size_t i = 0; i < all_cluster_infos.size(); ++i) {
        const auto& info = all_cluster_infos[i];
        geometry_msgs::Point safe_point = calculateSafeTarget(info);
        
        // 获取当前聚类的首尾点索引
        const auto& cluster_indices = clusters[i];
        int first_idx = cluster_indices.front();
        int last_idx = cluster_indices.back();
        
        // 计算首尾点的坐标
        float first_angle = scan->angle_min + first_idx * scan->angle_increment;
        float first_dist = scan->ranges[first_idx];
        float first_x = first_dist * cos(first_angle);
        float first_y = first_dist * sin(first_angle);
        
        float last_angle = scan->angle_min + last_idx * scan->angle_increment;
        float last_dist = scan->ranges[last_idx];
        float last_x = last_dist * cos(last_angle);
        float last_y = last_dist * sin(last_angle);
        
        std::cout << "物体 " << std::setw(2) << (i+1) << " [" 
                 << (validity_flags[i] ? "✓ 有效" : "✗ 被过滤") << "]" << std::endl;
        std::cout << "  ├─ 板子中心: (" << std::fixed << std::setprecision(2) 
                 << info.center.x << ", " << info.center.y << ")" << std::endl;
        std::cout << "  ├─ 安全目标点: (" << safe_point.x << ", " << safe_point.y << ")" << std::endl;
        std::cout << "  ├─ PCA长度: " << std::setprecision(3) << info.length << "m" << std::endl;
        std::cout << "  ├─ PCA朝向: " << std::setprecision(1) << (info.board_yaw * 180 / M_PI) << "°" << std::endl;
        std::cout << "  ├─ PCA置信度: " << std::setprecision(3) << info.pca_confidence << std::endl;
        std::cout << "  ├─ 距离: " << std::setprecision(1) << info.average_distance << "m" << std::endl;
        std::cout << "  ├─ 点数: " << info.size << std::endl;
        std::cout << "  ├─ 角度跨度: " << std::setprecision(3) << info.angular_width << "rad" << std::endl;
        std::cout << "  ├─ 首点坐标: (" << std::fixed << std::setprecision(2) 
                 << first_x << ", " << first_y << ") [索引:" << first_idx << "]" << std::endl;
        std::cout << "  ├─ 尾点坐标: (" << std::fixed << std::setprecision(2) 
                 << last_x << ", " << last_y << ") [索引:" << last_idx << "]" << std::endl;
        if(!validity_flags[i]) {
            std::cout << "  └─ 过滤原因: " << filter_reasons[i] << std::endl;
        } else {
            std::cout << "  └─ 状态: PCA符合识别板特征" << std::endl;
        }
        std::cout << std::endl;
    }
    
    std::cout << "PCA统计信息:" << std::endl;
    std::cout << "  - 总聚类数: " << clusters.size() << std::endl;
    std::cout << "  - PCA有效识别板: " << valid_clusters << std::endl;
    std::cout << "  - 被过滤物体: " << (clusters.size() - valid_clusters) << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
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
            
            if(distance < params_.duplicate_distance) {
                float dist_i = sqrt(pow(temp_clusters[i].x - robot_x_, 2) + 
                                   pow(temp_clusters[i].y - robot_y_, 2));
                float dist_j = sqrt(pow(temp_clusters[j].x - robot_x_, 2) + 
                                   pow(temp_clusters[j].y - robot_y_, 2));
                
                if(dist_i < dist_j) {
                    keep_flag[j] = false;
                    ROS_INFO("PCA去重: 移除聚类 %zu (距离 %.2fm)，保留聚类 %zu (距离 %.2fm)", 
                            j+1, dist_j, i+1, dist_i);
                } else {
                    keep_flag[i] = false;
                    ROS_INFO("PCA去重: 移除聚类 %zu (距离 %.2fm)，保留聚类 %zu (距离 %.2fm)", 
                            i+1, dist_i, j+1, dist_j);
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
    detected_clusters_.clear();
    detected_cluster_infos_.clear();
    
    for(size_t i = 0; i < temp_clusters.size(); ++i) {
        if(keep_flag[i]) {
            detected_clusters_.push_back(temp_clusters[i]);
            detected_cluster_infos_.push_back(temp_infos[i]);
        }
    }
    
    clusters_detected_ = !detected_clusters_.empty();
    
    if(clusters_detected_) {
        ROS_INFO("PCA最终检测到 %zu 个识别板", detected_clusters_.size());
        
        // 显示最终有效的板子列表
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "     PCA最终有效的识别板（共 " << detected_clusters_.size() << " 个）" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        for(size_t i = 0; i < detected_clusters_.size(); ++i) {
            const auto& cluster = detected_clusters_[i];
            const auto& info = detected_cluster_infos_[i];
            
            std::cout << "识别板 " << (i+1) << ":" << std::endl;
            std::cout << "  ├─ 板子中心: (" << std::fixed << std::setprecision(2) 
                     << info.center.x << ", " << info.center.y << ")" << std::endl;
            std::cout << "  ├─ 安全目标点: (" << cluster.x << ", " << cluster.y << ")" << std::endl;
            std::cout << "  ├─ PCA长度: " << std::setprecision(3) << info.length << "m" << std::endl;
            std::cout << "  ├─ PCA朝向: " << std::setprecision(1) << (info.board_yaw * 180 / M_PI) << "°" << std::endl;
            std::cout << "  ├─ PCA置信度: " << std::setprecision(3) << info.pca_confidence << std::endl;
            std::cout << "  ├─ 距离: " << std::setprecision(1) << info.average_distance << "m" << std::endl;
            std::cout << "  └─ 点数: " << info.size << std::endl;
            std::cout << std::endl;
        }
    } else {
        ROS_INFO("PCA未检测到有效识别板");
        std::cout << "\n⚠️  PCA未检测到符合标准的识别板，请调整参数或机器人位置" << std::endl;
    }
}
    
   ClusterInfo calculateClusterInfo(const std::vector<int>& cluster, 
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
        
        float global_x = robot_x_ + local_x * cos(robot_yaw_) - local_y * sin(robot_yaw_);
        float global_y = robot_y_ + local_x * sin(robot_yaw_) + local_y * cos(robot_yaw_);
        
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
    
       // ========== 修正：计算法向量并选择面向机器人的一侧 ==========
    // PCA给出的是板子主方向，我们需要法向量（垂直方向）
    float principal_yaw = pca_result.orientation;  // 板子主方向
    float normal_yaw = principal_yaw + M_PI / 2;   // 旋转90度得到法向量
    
    // 确保法向量面向机器人（机器人面向板子）
    float dx = info.center.x - robot_x_;
    float dy = info.center.y - robot_y_;
    
    // 计算法向量方向与机器人方向的点积
    float normal_dx = cos(normal_yaw);
    float normal_dy = sin(normal_yaw);
    float dot_product = normal_dx * dx + normal_dy * dy;
    
    // 如果点积为负，说明法向量背对机器人，需要翻转180度使其面向
    if (dot_product < 0) {  // 修改条件：< 0 而不是 > 0
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
        
    
  bool isValidObjectCluster(const ClusterInfo& cluster_info, 
                        const std::vector<int>& cluster,
                        const sensor_msgs::LaserScan::ConstPtr& scan,
                        std::string& debug_info) {
    std::ostringstream oss;
    
    // ========== 新增：地理约束检查 ==========
    float x = cluster_info.center.x;
    float y = cluster_info.center.y;
    
    // 房间边界（内缩0.15m）
    const float valid_min_x = -0.20f;
    const float valid_max_x = 3.58f;
    const float valid_min_y = 2.94f;
    const float valid_max_y = 7.50f;
    
    if (x < valid_min_x || x > valid_max_x || y < valid_min_y || y > valid_max_y) {
        oss << "超出有效区域: (" << x << "," << y << ") 不在 [" 
            << valid_min_x << "," << valid_max_x << "]x[" 
            << valid_min_y << "," << valid_max_y << "]";
        debug_info = oss.str();
        return false;
    }
    // ========== 地理约束检查结束 ==========

        // 1. 基本长度检查
        if(cluster_info.length < params_.min_board_length) {
            oss << "PCA长度过小: " << cluster_info.length << "m < " << params_.min_board_length << "m";
            debug_info = oss.str();
            return false;
        }
        
        if(cluster_info.length > params_.max_board_length) {
            oss << "PCA长度过大: " << cluster_info.length << "m > " << params_.max_board_length << "m";
            debug_info = oss.str();
            return false;
        }
        
        // 2. 点数检查
        if(cluster.size() < params_.min_cluster_size) {
            oss << "点数过少: " << cluster.size() << " < " << params_.min_cluster_size;
            debug_info = oss.str();
            return false;
        }
        
        // 3. PCA置信度检查
        if(cluster_info.pca_confidence < params_.min_pca_confidence) {
            oss << "PCA置信度过低: " << cluster_info.pca_confidence << " < " << params_.min_pca_confidence;
            debug_info = oss.str();
            return false;
        }
        
        // 4. 距离连续性检查
        float distance_std = 0.0f;
        float mean_dist = cluster_info.average_distance;
        for(int idx : cluster) {
            float diff = scan->ranges[idx] - mean_dist;
            distance_std += diff * diff;
        }
        distance_std = sqrt(distance_std / cluster.size());
        
        if(distance_std > params_.max_distance_std) {
            oss << "距离变化过大: 标准差=" << distance_std << "m > " << params_.max_distance_std << "m";
            debug_info = oss.str();
            return false;
        }
        
        debug_info = "PCA符合识别板特征";
        return true;
    }
    
    float calculateBoardLength(const std::vector<int>& cluster, 
                             const sensor_msgs::LaserScan::ConstPtr& scan) {
        if (cluster.size() < 5) {
            ROS_WARN("PCA长度计算需要至少5个点，当前只有%zu个点", cluster.size());
            return 0.0f;
        }
        
        PCAResult pca_result = computePCA(cluster, scan);
        
        // 添加调试信息
        ROS_INFO("=== PCA长度计算 ===");
        ROS_INFO("输入点数: %zu", cluster.size());
        ROS_INFO("PCA长度: %.3fm", pca_result.length);
        ROS_INFO("PCA朝向: %.1f°", pca_result.orientation * 180 / M_PI);
        ROS_INFO("PCA置信度: %.3f", pca_result.confidence);
        ROS_INFO("=================");
        
        return pca_result.length;
    }
    
geometry_msgs::Point calculateSafeTarget(const ClusterInfo& cluster_info) {
    geometry_msgs::Point safe_target;
    
    float safe_distance = params_.default_safe_distance;
    
    // 基于代价地图调整安全距离
    if(costmap_updated_ && !isTargetReachable(cluster_info.center)) {
        safe_distance = params_.extended_safe_distance;
        ROS_DEBUG("使用扩展安全距离 %.2fm", safe_distance);
    }
    
    // ========== 修改：沿法向量方向后退安全距离（不需要反方向） ==========
    // cluster_info.board_yaw 现在是法向量（垂直于板子，背对机器人）
    // 安全距离应该直接沿法向量方向后退
    
    float back_dir_x = -cos(cluster_info.board_yaw);  // 加回负号！
    float back_dir_y = -sin(cluster_info.board_yaw);  // 加回负号！
    
    safe_target.x = cluster_info.center.x + back_dir_x * safe_distance;
    safe_target.y = cluster_info.center.y + back_dir_y * safe_distance;
    safe_target.z = 0.0;
    
    ROS_DEBUG("安全目标: 从板子中心(%.2f,%.2f)沿法向量反方向%.1f°后退%.2fm到(%.2f,%.2f)",
             cluster_info.center.x, cluster_info.center.y,
             cluster_info.board_yaw * 180 / M_PI, safe_distance,
             safe_target.x, safe_target.y);
    
    return safe_target;
}
    
    bool getRobotPose() {
        try {
            geometry_msgs::TransformStamped transform;
            transform = tf_buffer_.lookupTransform("map", "base_footprint", ros::Time(0), ros::Duration(0.1));
            
            robot_x_ = transform.transform.translation.x;
            robot_y_ = transform.transform.translation.y;
            
            tf2::Quaternion q(
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            );
            tf2::Matrix3x3 m(q);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);
            robot_yaw_ = yaw;
            
            return true;
        }
        catch(tf2::TransformException &ex) {
            ROS_WARN_THROTTLE(5, "TF变换获取失败: %s", ex.what());
            return false;
        }
    }
    
    void costmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        current_costmap_ = *msg;
        costmap_updated_ = true;
        ROS_DEBUG_THROTTLE(5, "代价地图已更新");
    }
    
    bool isTargetReachable(const geometry_msgs::Point& point) {
        if(!costmap_updated_) return true;
        
        float map_x = current_costmap_.info.origin.position.x;
        float map_y = current_costmap_.info.origin.position.y;
        float resolution = current_costmap_.info.resolution;
        
        int mx = static_cast<int>((point.x - map_x) / resolution);
        int my = static_cast<int>((point.y - map_y) / resolution);
        
        if(mx < 0 || mx >= current_costmap_.info.width || 
           my < 0 || my >= current_costmap_.info.height) {
            ROS_DEBUG("目标点超出代价地图范围");
            return false;
        }
        
        int index = my * current_costmap_.info.width + mx;
        if(index < 0 || index >= current_costmap_.data.size()) {
            ROS_DEBUG("目标点索引超出范围");
            return false;
        }
        
        int cost = current_costmap_.data[index];
        bool reachable = cost < 50;
        
        if(!reachable) {
            ROS_DEBUG("目标点不可达，代价: %d", cost);
        }
        
        return reachable;
    }
};

int main(int argc, char** argv) {   
    setlocale(LC_ALL,"");
    ros::init(argc, argv, "pca_board_detection_navigator");
    
    BoardDetectionNavigator navigator;
    navigator.run();
    
    return 0;
}