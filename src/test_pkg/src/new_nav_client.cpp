#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <map>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <fstream>

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

class NavigationAnalyzer {
private:
    ros::NodeHandle nh_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // 导航数据
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point end_time_;
    geometry_msgs::Point target_pose_;
    double target_yaw_;
    std::string current_target_name_;
    bool navigation_active_;
    
    // 速度数据记录（优化频率）
    std::vector<double> linear_vel_history_;
    std::vector<double> angular_vel_history_;
    std::vector<geometry_msgs::Point> path_points_;
    std::vector<ros::Time> velocity_timestamps_;
    
    // 停稳相关
    std::chrono::steady_clock::time_point reach_goal_time_;
    std::chrono::steady_clock::time_point full_stop_time_;
    bool goal_reached_;
    bool full_stop_;
    
    // 缓存当前速度和位姿
    double current_linear_vel_;
    double current_angular_vel_;
    geometry_msgs::Point current_pose_;
    double current_yaw_;
    bool pose_updated_;
    
    // 性能优化：减少TF查询频率
    ros::Time last_tf_query_time_;
    const double tf_query_interval_ = 0.1; // 100ms查询一次TF
    
    // 定时器
    ros::Timer data_collection_timer_;
    
    // 参数配置
    const double max_vel_x_ = 6.5;        
    const double xy_goal_tolerance_ = 0.18;  // 匹配TEB配置
    const double yaw_goal_tolerance_ = 0.2;  // 匹配TEB配置
    const double stop_velocity_threshold_ = 0.05;
    
    // ==================== 转弯检测相关 ====================
    std::vector<double> path_yaws_;
    std::vector<ros::Time> yaw_timestamps_;
    double last_yaw_;
    bool is_turning_;
    double turn_start_yaw_;
    ros::Time turn_start_time_;
    std::vector<double> linear_accel_history_;
    std::vector<double> angular_accel_history_;
    double last_linear_vel_;
    double last_angular_vel_;
    ros::Time last_vel_time_;
    bool turn_logging_active_;
    std::ofstream turn_log_file_;
    
    // 转弯参数
    const double turn_threshold_ = 0.2;           // 角速度阈值，检测转弯开始
    const double turn_angle_threshold_ = 0.5;     // 转弯角度阈值
    const double max_linear_accel_ = 3.5;         // 匹配TEB的acc_lim_x
    const double max_angular_accel_ = 4.5;        // 匹配TEB的acc_lim_theta
    int turn_event_count_;                        // 转弯事件计数

public:
    NavigationAnalyzer() : 
        tf_listener_(tf_buffer_),
        goal_reached_(false),
        full_stop_(false),
        navigation_active_(false),
        current_linear_vel_(0.0),
        current_angular_vel_(0.0),
        pose_updated_(false),
        last_tf_query_time_(ros::Time::now()),
        last_yaw_(0.0),
        is_turning_(false),
        turn_start_yaw_(0.0),
        last_linear_vel_(0.0),
        last_angular_vel_(0.0),
        turn_logging_active_(false),
        turn_event_count_(0) {
        
        // 降低数据采集频率到2Hz
        data_collection_timer_ = nh_.createTimer(ros::Duration(0.5), 
                                               &NavigationAnalyzer::collectData, this);
        
        // 初始化转弯日志文件
        initializeTurnLogFile();
    }
    
    ~NavigationAnalyzer() {
        if (turn_log_file_.is_open()) {
            turn_log_file_ << "\n" << std::string(50, '=') << std::endl;
            turn_log_file_ << "转弯分析日志 - 结束时间: " << getCurrentTimeString() << std::endl;
            turn_log_file_ << "总转弯事件: " << turn_event_count_ << std::endl;
            turn_log_file_ << std::string(50, '=') << std::endl;
            turn_log_file_.close();
            ROS_INFO("转弯日志文件已关闭");
        }
    }
    
    void initializeTurnLogFile() {
        // 确保转弯日志文件可以创建和写入
        turn_log_file_.open("turn_analysis_log.txt", std::ios::app);
        if (turn_log_file_.is_open()) {
            ROS_INFO("转弯日志文件已打开: turn_analysis_log.txt");
            turn_log_file_ << "\n" << std::string(50, '=') << std::endl;
            turn_log_file_ << "转弯分析日志 - 开始时间: " << getCurrentTimeString() << std::endl;
            turn_log_file_ << std::string(50, '=') << std::endl;
            turn_log_file_.flush(); // 确保立即写入
        } else {
            // 如果打开失败，尝试使用绝对路径
            ROS_ERROR("无法打开转弯日志文件，尝试创建文件...");
            
            std::string home_dir = getenv("HOME") ? getenv("HOME") : ".";
            std::string file_path = home_dir + "/turn_analysis_log.txt";
            turn_log_file_.open(file_path.c_str(), std::ios::app);
            
            if (turn_log_file_.is_open()) {
                ROS_INFO("转弯日志文件已创建在: %s", file_path.c_str());
                turn_log_file_ << "\n" << std::string(50, '=') << std::endl;
                turn_log_file_ << "转弯分析日志 - 开始时间: " << getCurrentTimeString() << std::endl;
                turn_log_file_ << std::string(50, '=') << std::endl;
                turn_log_file_.flush();
            } else {
                ROS_ERROR("仍然无法创建转弯日志文件，转弯分析功能将禁用");
            }
        }
    }
    
    bool isTurnLogAvailable() const {
        return turn_log_file_.is_open();
    }
    
    void startNavigation(const geometry_msgs::Point& target, double yaw, const std::string& target_name = "") {
        // 重置数据
        linear_vel_history_.clear();
        angular_vel_history_.clear();
        path_points_.clear();
        velocity_timestamps_.clear();
        path_yaws_.clear();
        yaw_timestamps_.clear();
        linear_accel_history_.clear();
        angular_accel_history_.clear();
        goal_reached_ = false;
        full_stop_ = false;
        pose_updated_ = false;
        is_turning_ = false;
        turn_logging_active_ = false;
        last_linear_vel_ = 0.0;
        last_angular_vel_ = 0.0;
        last_vel_time_ = ros::Time::now();
        
        target_pose_ = target;
        target_yaw_ = yaw;
        current_target_name_ = target_name;
        navigation_active_ = true;
        start_time_ = std::chrono::steady_clock::now();
        last_tf_query_time_ = ros::Time::now();
        
        ROS_INFO("开始导航分析 - 目标位置: (%.2f, %.2f, %.1f°)", 
                 target.x, target.y, yaw * 180.0 / M_PI);
    }
    
    // 优化速度记录：添加转弯检测和加速度计算
    void recordVelocityData(double linear_x, double angular_z) {
        current_linear_vel_ = linear_x;
        current_angular_vel_ = angular_z;
        
        // 计算加速度
        ros::Time current_time = ros::Time::now();
        double dt = (current_time - last_vel_time_).toSec();
        if (dt > 0.01 && dt < 1.0) { // 避免除零和过大时间间隔
            double linear_accel = (linear_x - last_linear_vel_) / dt;
            double angular_accel = (angular_z - last_angular_vel_) / dt;
            
            linear_accel_history_.push_back(linear_accel);
            angular_accel_history_.push_back(angular_accel);
            
            // 检测转弯开始
            if (!is_turning_ && fabs(angular_z) > turn_threshold_) {
                is_turning_ = true;
                turn_start_yaw_ = current_yaw_;
                turn_start_time_ = current_time;
                turn_logging_active_ = true;
                turn_event_count_++;
                
                ROS_INFO("🚗 检测到转弯开始 - 角速度: %.3f rad/s, 当前偏航: %.2f°", 
                         angular_z, current_yaw_ * 180.0 / M_PI);
                if (isTurnLogAvailable()) {
                    turn_log_file_ << "\n[转弯开始] 时间: " << getCurrentTimeString() 
                                 << ", 角速度: " << angular_z << " rad/s"
                                 << ", 当前偏航: " << current_yaw_ * 180.0 / M_PI << "°"
                                 << std::endl;
                    turn_log_file_.flush();
                }
            }
            
            // 检测转弯结束
            if (is_turning_ && fabs(angular_z) < turn_threshold_ * 0.3) {
                is_turning_ = false;
                double turn_duration = (current_time - turn_start_time_).toSec();
                double turn_angle = fabs(normalizeAngle(current_yaw_ - turn_start_yaw_));
                
                ROS_INFO("🛑 转弯结束 - 持续时间: %.2fs, 转弯角度: %.2f°, 最终偏航: %.2f°", 
                         turn_duration, turn_angle * 180.0 / M_PI, current_yaw_ * 180.0 / M_PI);
                
                if (isTurnLogAvailable()) {
                    turn_log_file_ << "[转弯结束] 时间: " << getCurrentTimeString()
                                 << ", 持续时间: " << turn_duration << "s"
                                 << ", 转弯角度: " << turn_angle * 180.0 / M_PI << "°"
                                 << ", 最终偏航: " << current_yaw_ * 180.0 / M_PI << "°"
                                 << std::endl;
                    turn_log_file_.flush();
                }
                
                turn_logging_active_ = false;
            }
            
            // 转弯过程中的详细日志记录
            if (turn_logging_active_) {
                logTurningData(linear_x, angular_z, linear_accel, angular_accel);
            }
            
            // 检查加速度限制
            if (fabs(linear_accel) > max_linear_accel_) {
                ROS_WARN_THROTTLE(2.0, "⚠️  线加速度超过限制: %.2f m/s² > %.2f m/s²", 
                                 fabs(linear_accel), max_linear_accel_);
            }
            
            if (fabs(angular_accel) > max_angular_accel_) {
                ROS_WARN_THROTTLE(2.0, "⚠️  角加速度超过限制: %.2f rad/s² > %.2f rad/s²", 
                                 fabs(angular_accel), max_angular_accel_);
            }
            
            last_linear_vel_ = linear_x;
            last_angular_vel_ = angular_z;
            last_vel_time_ = current_time;
        }
        
        // 频率控制：每100ms记录一次速度数据（提高频率）
        static ros::Time last_vel_record = ros::Time::now();
        if ((ros::Time::now() - last_vel_record).toSec() > 0.1) {
            linear_vel_history_.push_back(fabs(linear_x));
            angular_vel_history_.push_back(fabs(angular_z));
            velocity_timestamps_.push_back(ros::Time::now());
            last_vel_record = ros::Time::now();
            
            // 限制数据量，防止内存过度增长
            if (linear_vel_history_.size() > 1000) {
                linear_vel_history_.erase(linear_vel_history_.begin());
                angular_vel_history_.erase(angular_vel_history_.begin());
                velocity_timestamps_.erase(velocity_timestamps_.begin());
            }
        }
    }
    
    // 转弯数据记录
    void logTurningData(double linear_vel, double angular_vel, double linear_accel, double angular_accel) {
        static int turn_log_count = 0;
        turn_log_count++;
        
        // 每3次记录一次，避免日志过多
        if (turn_log_count % 3 == 0) {
            double yaw_error = fabs(normalizeAngle(target_yaw_ - current_yaw_));
            
            ROS_INFO("🔄 转弯中 - 线速度: %.2f m/s, 角速度: %.2f rad/s, 偏航误差: %.2f°", 
                     linear_vel, angular_vel, yaw_error * 180.0 / M_PI);
            
            if (isTurnLogAvailable()) {
                turn_log_file_ << "  时间: " << getCurrentTimeString()
                             << ", 线速度: " << std::fixed << std::setprecision(3) << linear_vel << " m/s"
                             << ", 角速度: " << angular_vel << " rad/s"
                             << ", 线加速度: " << linear_accel << " m/s²"
                             << ", 角加速度: " << angular_accel << " rad/s²"
                             << ", 偏航误差: " << yaw_error << " rad (" << yaw_error * 180.0 / M_PI << "°)"
                             << ", 目标偏航: " << target_yaw_ * 180.0 / M_PI << "°"
                             << ", 当前偏航: " << current_yaw_ * 180.0 / M_PI << "°"
                             << std::endl;
                turn_log_file_.flush();
            }
        }
    }
    
    // 优化数据采集：降低频率和减少TF查询
    void collectData(const ros::TimerEvent&) {
        if (!navigation_active_) return;

        // 控制TF查询频率
        if ((ros::Time::now() - last_tf_query_time_).toSec() < tf_query_interval_) {
            return;
        }
        
        last_tf_query_time_ = ros::Time::now();
        
        // 更新当前位姿
        if (getCurrentPose(current_pose_) && getCurrentYaw(current_yaw_)) {
            pose_updated_ = true;

            // 记录偏航角历史用于转弯分析
            path_yaws_.push_back(current_yaw_);
            yaw_timestamps_.push_back(ros::Time::now());
            
            // 限制数据量
            if (path_yaws_.size() > 500) {
                path_yaws_.erase(path_yaws_.begin());
                yaw_timestamps_.erase(yaw_timestamps_.begin());
            }

            // 优化路径点记录：增加距离阈值
            recordPathPoint(current_pose_);

            // 判断是否到达目标
            if (!goal_reached_ && isGoalReached()) {
                goal_reached_ = true;
                reach_goal_time_ = std::chrono::steady_clock::now();
                ROS_INFO("✅ 到达目标区域 - 开始停稳检测");
            }
            
            // 判断是否完全停稳
            if (goal_reached_ && !full_stop_ && fabs(current_linear_vel_) < stop_velocity_threshold_) {
                full_stop_ = true;
                full_stop_time_ = std::chrono::steady_clock::now();
                ROS_INFO("🛑 完全停稳 - 速度 < %.3f m/s", stop_velocity_threshold_);
            }
        } else {
            pose_updated_ = false;
        }
    }
    
    void stopNavigation() {
        navigation_active_ = false;
        end_time_ = std::chrono::steady_clock::now();
        
        // 输出转弯分析总结
        printTurnAnalysis();
        
        printNavigationAnalysis();
        saveToFile();
    }
    
    bool isNavigationFinished() {
        return goal_reached_ && full_stop_;
    }
    
    void forceStop() {
        navigation_active_ = false;
        end_time_ = std::chrono::steady_clock::now();
        saveToFile();
    }
    
private:
    bool getCurrentPose(geometry_msgs::Point& current_pose) {
        try {
            geometry_msgs::TransformStamped transform;
            transform = tf_buffer_.lookupTransform("map", "base_footprint", ros::Time(0), ros::Duration(0.05)); // 减少超时时间
            
            current_pose.x = transform.transform.translation.x;
            current_pose.y = transform.transform.translation.y;
            current_pose.z = transform.transform.translation.z;
            return true;
        }
        catch (tf2::TransformException &ex) {
            ROS_WARN_THROTTLE(5.0, "TF查询失败: %s", ex.what());
            return false;
        }
    }
    
    bool getCurrentYaw(double& yaw) {
        try {
            geometry_msgs::TransformStamped transform;
            transform = tf_buffer_.lookupTransform("map", "base_footprint", ros::Time(0), ros::Duration(0.05));
            
            tf2::Quaternion q(
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w);
            tf2::Matrix3x3 m(q);
            double roll, pitch;
            m.getRPY(roll, pitch, yaw);
            return true;
        }
        catch (tf2::TransformException &ex) {
            ROS_WARN_THROTTLE(5.0, "TF查询失败: %s", ex.what());
            return false;
        }
    }
    
    bool isGoalReached() {
        if (!pose_updated_) return false;
        
        double dx = current_pose_.x - target_pose_.x;
        double dy = current_pose_.y - target_pose_.y;
        double distance_error = sqrt(dx*dx + dy*dy);
        
        double yaw_error = fabs(current_yaw_ - target_yaw_);
        if (yaw_error > M_PI) {
            yaw_error = 2 * M_PI - yaw_error;
        }
        
        return (distance_error <= xy_goal_tolerance_ && yaw_error <= yaw_goal_tolerance_);
    }
    
    // 优化路径点记录：增加距离阈值，减少数据量
    void recordPathPoint(const geometry_msgs::Point& pose) {
        if (path_points_.empty()) {
            path_points_.push_back(pose);
            return;
        }

        // 增加记录阈值到0.2m，大幅减少数据量
        const geometry_msgs::Point& last = path_points_.back();
        double dx = pose.x - last.x;
        double dy = pose.y - last.y;
        if (sqrt(dx*dx + dy*dy) > 0.2) {
            path_points_.push_back(pose);
            
            // 限制路径点数量
            if (path_points_.size() > 500) {
                path_points_.erase(path_points_.begin());
            }
        }
    }
    
    // 角度归一化到 [-π, π]
    double normalizeAngle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }
    
    // 获取当前时间字符串
    std::string getCurrentTimeString() {
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::string time_str = std::ctime(&now_time);
        time_str.pop_back(); // 移除换行符
        return time_str;
    }
    
    // 转弯分析
    void printTurnAnalysis() {
        if (path_yaws_.size() < 2) return;
        
        // 计算偏航角变化
        std::vector<double> yaw_changes;
        for (size_t i = 1; i < path_yaws_.size(); i++) {
            double change = normalizeAngle(path_yaws_[i] - path_yaws_[i-1]);
            yaw_changes.push_back(fabs(change));
        }
        
        // 统计转弯数据
        double total_turn_angle = std::accumulate(yaw_changes.begin(), yaw_changes.end(), 0.0);
        double max_turn_rate = *std::max_element(yaw_changes.begin(), yaw_changes.end());
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "             转弯分析报告" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << "总偏航变化: " << std::fixed << std::setprecision(2) << total_turn_angle * 180.0 / M_PI << "°" << std::endl;
        std::cout << "最大单次偏航变化: " << std::setprecision(2) << max_turn_rate * 180.0 / M_PI << "°/s" << std::endl;
        std::cout << "平均角速度: " << std::setprecision(3) << calculateAverageAngularSpeed() << " rad/s" << std::endl;
        std::cout << "最大角速度: " << std::setprecision(3) << calculateMaxAngularSpeed() << " rad/s" << std::endl;
        std::cout << "转弯事件次数: " << turn_event_count_ << std::endl;
        
        if (!linear_accel_history_.empty()) {
            double avg_linear_accel = std::accumulate(linear_accel_history_.begin(), linear_accel_history_.end(), 0.0) / linear_accel_history_.size();
            double max_linear_accel = *std::max_element(linear_accel_history_.begin(), linear_accel_history_.end(), 
                                                       [](double a, double b) { return fabs(a) < fabs(b); });
            std::cout << "平均线加速度: " << std::setprecision(3) << avg_linear_accel << " m/s²" << std::endl;
            std::cout << "最大线加速度: " << std::setprecision(3) << fabs(max_linear_accel) << " m/s²" << std::endl;
        }
        
        if (!angular_accel_history_.empty()) {
            double avg_angular_accel = std::accumulate(angular_accel_history_.begin(), angular_accel_history_.end(), 0.0) / angular_accel_history_.size();
            double max_angular_accel = *std::max_element(angular_accel_history_.begin(), angular_accel_history_.end(), 
                                                        [](double a, double b) { return fabs(a) < fabs(b); });
            std::cout << "平均角加速度: " << std::setprecision(3) << avg_angular_accel << " rad/s²" << std::endl;
            std::cout << "最大角加速度: " << std::setprecision(3) << fabs(max_angular_accel) << " rad/s²" << std::endl;
        }
        
        std::cout << std::string(50, '=') << std::endl;
    }
    
    // 保持原有的计算函数不变
    double calculateAverageSpeed() {
        if (linear_vel_history_.empty()) return 0.0;
        return std::accumulate(linear_vel_history_.begin(), linear_vel_history_.end(), 0.0) / linear_vel_history_.size();
    }
    
    double calculateMaxSpeed() {
        if (linear_vel_history_.empty()) return 0.0;
        return *std::max_element(linear_vel_history_.begin(), linear_vel_history_.end());
    }
    
    double calculateAverageAngularSpeed() {
        if (angular_vel_history_.empty()) return 0.0;
        return std::accumulate(angular_vel_history_.begin(), angular_vel_history_.end(), 0.0) / angular_vel_history_.size();
    }
    
    double calculateMaxAngularSpeed() {
        if (angular_vel_history_.empty()) return 0.0;
        return *std::max_element(angular_vel_history_.begin(), angular_vel_history_.end());
    }
    
    double calculateSpeedUtilization() {
        return (calculateAverageSpeed() / max_vel_x_) * 100.0;
    }
    
    double calculateStagnationRatio() {
        if (linear_vel_history_.empty()) return 0.0;
        
        int stagnation_count = 0;
        for (double vel : linear_vel_history_) {
            if (vel < 0.03) stagnation_count++;
        }
        
        return (static_cast<double>(stagnation_count) / linear_vel_history_.size()) * 100.0;
    }
    
    void calculateGoalErrors(double& position_error, double& angle_error) {
        if (pose_updated_) {
            double dx = current_pose_.x - target_pose_.x;
            double dy = current_pose_.y - target_pose_.y;
            position_error = sqrt(dx*dx + dy*dy);
            
            angle_error = fabs(current_yaw_ - target_yaw_);
            if (angle_error > M_PI) {
                angle_error = 2 * M_PI - angle_error;
            }
        } else {
            position_error = 0.0;
            angle_error = 0.0;
        }
    }
    
    double calculateStoppingTime() {
        if (!goal_reached_ || !full_stop_) return 0.0;
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(full_stop_time_ - reach_goal_time_);
        return duration.count() / 1000.0;
    }
    
    double calculatePathEfficiency() {
        if (path_points_.size() < 2) return 1.0;
        
        double actual_distance = 0.0;
        for (size_t i = 1; i < path_points_.size(); i++) {
            double dx = path_points_[i].x - path_points_[i-1].x;
            double dy = path_points_[i].y - path_points_[i-1].y;
            actual_distance += sqrt(dx*dx + dy*dy);
        }
        
        double straight_distance = 0.0;
        if (path_points_.size() >= 2) {
            double dx = path_points_.back().x - path_points_.front().x;
            double dy = path_points_.back().y - path_points_.front().y;
            straight_distance = sqrt(dx*dx + dy*dy);
        }
        
        return (straight_distance > 0) ? (actual_distance / straight_distance) : 1.0;
    }
    
    double calculateSpeedVariance() {
        if (linear_vel_history_.size() < 2) return 0.0;
        
        double mean = calculateAverageSpeed();
        double variance = 0.0;
        for (double vel : linear_vel_history_) {
            variance += (vel - mean) * (vel - mean);
        }
        return variance / linear_vel_history_.size();
    }
    
    double calculateTotalTime() {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time_);
        return duration.count() / 1000.0;
    }
    
    void saveToFile() {
        std::ofstream file("navigation_log.txt", std::ios::app);
        
        if (!file.is_open()) {
            ROS_WARN("无法打开日志文件");
            return;
        }
        
        double total_time = calculateTotalTime();
        double position_error, angle_error;
        calculateGoalErrors(position_error, angle_error);
        
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        
        file << "\n" << std::string(50, '=') << std::endl;
        file << "导航分析报告 - " << std::ctime(&now_time);
        file << "目标点: " << current_target_name_ << std::endl;
        file << std::string(50, '-') << std::endl;
        
        file << "总耗时: " << std::fixed << std::setprecision(2) << total_time << " s" << std::endl;
        file << "平均速度: " << std::setprecision(3) << calculateAverageSpeed() << " m/s" << std::endl;
        file << "速度利用率: " << std::setprecision(1) << calculateSpeedUtilization() << "%" << std::endl;
        file << "停滞时间占比: " << std::setprecision(1) << calculateStagnationRatio() << "%" << std::endl;
        file << "位置误差: " << std::setprecision(3) << position_error << " m" << std::endl;
        file << "角度误差: " << std::setprecision(3) << angle_error << " rad" << std::endl;
        file << "停稳时间: " << std::setprecision(2) << calculateStoppingTime() << " s" << std::endl;
        file << "路径效率: " << std::setprecision(3) << calculatePathEfficiency() << std::endl;
        file << "速度波动: " << std::setprecision(4) << calculateSpeedVariance() << std::endl;
        
        // 添加转弯分析数据
        if (path_yaws_.size() >= 2) {
            file << "转弯分析:" << std::endl;
            file << "  平均角速度: " << std::setprecision(3) << calculateAverageAngularSpeed() << " rad/s" << std::endl;
            file << "  最大角速度: " << std::setprecision(3) << calculateMaxAngularSpeed() << " rad/s" << std::endl;
            file << "  转弯事件次数: " << turn_event_count_ << std::endl;
        }
        
        file << std::string(50, '=') << std::endl;
        file.close();
        
        ROS_INFO("导航数据已保存到 navigation_log.txt");
    }
    
    void printNavigationAnalysis() {
        double total_time = calculateTotalTime();
        double position_error, angle_error;
        calculateGoalErrors(position_error, angle_error);
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "                 导航性能分析报告" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        std::cout << "总耗时: " << std::fixed << std::setprecision(2) << total_time << " s" << std::endl;
        std::cout << "平均速度: " << std::setprecision(3) << calculateAverageSpeed() << " m/s" << std::endl;
        std::cout << "速度利用率: " << std::setprecision(1) << calculateSpeedUtilization() << "%" << std::endl;
        std::cout << "停滞时间占比: " << std::setprecision(1) << calculateStagnationRatio() << "%" << std::endl;
        std::cout << "位置误差: " << std::setprecision(3) << position_error << " m" << std::endl;
        std::cout << "角度误差: " << std::setprecision(3) << angle_error << " rad" << std::endl;
        std::cout << "停稳时间: " << std::setprecision(2) << calculateStoppingTime() << " s" << std::endl;
        std::cout << "路径效率: " << std::setprecision(3) << calculatePathEfficiency() << std::endl;
        
        std::cout << std::string(60, '=') << std::endl;
    }
};

// 创建位姿辅助函数
geometry_msgs::Pose createPose(double x, double y, double yaw) {
    geometry_msgs::Pose pose;
    pose.position.x = x;
    pose.position.y = y;
    pose.position.z = 0.0;
    
    pose.orientation.x = 0.0;
    pose.orientation.y = 0.0;
    pose.orientation.z = sin(yaw / 2);
    pose.orientation.w = cos(yaw / 2);
    
    return pose;
}

// 一条龙测试函数
void runCompleteTest(MoveBaseClient& ac, NavigationAnalyzer& analyzer) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "          开始一条龙测试" << std::endl;
    std::cout << "  路线: 二维码 → 拣货 → 等待 → 交通灯 → B路口 → 终点A" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    // 定义一条龙测试的路径点（按顺序）
    std::vector<std::pair<std::string, geometry_msgs::Pose>> complete_route = {
        {"二维码区域 (qr_zone)", createPose(1.35, 0.92, 3.14)},
        {"拣货区域 (pick_zone)", createPose(1.7, 5.35, 0.0)},
        {"等待区域 (wait_zone)", createPose(1.7, 6.34, 0.0)},
        {"交通灯区域 (traffic_zone)", createPose(4.9, 6.4, 1.57)},
        {"B路口入口 (intersection_B)", createPose(7.3, 4.6, -1.57)},
        {"终点区域A (finish_zone_A)", createPose(4.9, 0.4, -1.57)}
    };
    
    double total_test_time = 0.0;
    int success_count = 0;
    int total_points = complete_route.size();
    
    // 创建一条龙测试的专属日志文件
    std::ofstream complete_test_log("complete_test_log.txt", std::ios::app);
    auto test_start_time = std::chrono::system_clock::now();
    std::time_t test_start_time_t = std::chrono::system_clock::to_time_t(test_start_time);
    
    complete_test_log << "\n" << std::string(60, '=') << std::endl;
    complete_test_log << "          一条龙测试开始 - " << std::ctime(&test_start_time_t);
    complete_test_log << "          测试路线: 二维码 → 拣货 → 等待 → 交通灯 → B路口 → 终点A" << std::endl;
    complete_test_log << std::string(60, '=') << std::endl;
    
    for (int i = 0; i < total_points; i++) {
        const auto& point = complete_route[i];
        
        std::cout << "\n▶️  第 " << (i+1) << "/" << total_points << " 站: " << point.first << std::endl;
        std::cout << "   目标位置: (" << point.second.position.x << ", " 
                  << point.second.position.y << ")" << std::endl;
        
        move_base_msgs::MoveBaseGoal goal;
        goal.target_pose.header.frame_id = "map";
        goal.target_pose.header.stamp = ros::Time::now();
        goal.target_pose.pose = point.second;
        
        geometry_msgs::Point target_point;
        target_point.x = point.second.position.x;
        target_point.y = point.second.position.y;
        target_point.z = 0.0;
        
        // 从四元数提取偏航角
        tf2::Quaternion q(
            point.second.orientation.x,
            point.second.orientation.y,
            point.second.orientation.z,
            point.second.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, target_yaw;
        m.getRPY(roll, pitch, target_yaw);
        
        analyzer.startNavigation(target_point, target_yaw, point.first);
        ac.sendGoal(goal);
        
        std::cout << "   导航进行中..." << std::endl;
        
        // 等待导航完成（带超时控制）
        bool finished = false;
        auto segment_start_time = std::chrono::steady_clock::now();
        const double timeout_seconds = 60.0; // 每段最大60秒
        
        while(!finished && ros::ok()) {
            finished = ac.waitForResult(ros::Duration(0.5));
            ros::spinOnce();
            
            // 检查超时
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - segment_start_time);
            if (elapsed.count() > timeout_seconds) {
                std::cout << "   ⚠️  超时，跳过此目标点" << std::endl;
                ac.cancelGoal();
                analyzer.forceStop();
                break;
            }
            
            // 检查用户取消
            if(std::cin.rdbuf()->in_avail() > 0) {
                std::string cmd;
                std::getline(std::cin, cmd);
                if(cmd == "c" || cmd == "C") {
                    ac.cancelGoal();
                    analyzer.forceStop();
                    std::cout << "   ❌ 用户取消测试" << std::endl;
                    complete_test_log.close();
                    return;
                }
            }
        }
        
        if(finished) {
            analyzer.stopNavigation();
            if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED) {
                success_count++;
                std::cout << "   ✅ 第 " << (i+1) << " 站完成" << std::endl;
            } else {
                std::cout << "   ❌ 第 " << (i+1) << " 站失败: " << ac.getState().toString() << std::endl;
            }
        }
        
        // 站点间短暂暂停
        if (i < total_points - 1) {
            std::cout << "   准备前往下一站..." << std::endl;
            ros::Duration(1.0).sleep();
        }
    }
    
    // 测试总结
    auto test_end_time = std::chrono::system_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(test_end_time - test_start_time);
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "          一条龙测试完成" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "总站点数: " << total_points << std::endl;
    std::cout << "成功站点: " << success_count << std::endl;
    std::cout << "成功率: " << std::fixed << std::setprecision(1) << (success_count * 100.0 / total_points) << "%" << std::endl;
    std::cout << "总耗时: " << total_duration.count() << " 秒" << std::endl;
    
    // 记录测试总结到日志文件
    complete_test_log << "\n测试总结:" << std::endl;
    complete_test_log << "总站点数: " << total_points << std::endl;
    complete_test_log << "成功站点: " << success_count << std::endl;
    complete_test_log << "成功率: " << std::fixed << std::setprecision(1) << (success_count * 100.0 / total_points) << "%" << std::endl;
    complete_test_log << "总耗时: " << total_duration.count() << " 秒" << std::endl;
    complete_test_log << std::string(60, '=') << std::endl;
    complete_test_log.close();
    
    std::cout << "详细数据已保存到 complete_test_log.txt" << std::endl;
}

int main(int argc, char** argv)
{   
    setlocale(LC_ALL,"");
    ros::init(argc, argv, "navigation_analyzer");
    
    // 优化：使用单独的节点句柄
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    
    MoveBaseClient ac("move_base", true);
    NavigationAnalyzer analyzer;
    
    if(!ac.waitForServer(ros::Duration(5.0))) {
        ROS_ERROR("无法连接到move_base action server");
        return 1;
    }
    ROS_INFO("已连接到move_base action server");
    
    // 按照任务流程顺序定义导航点
    std::vector<std::pair<std::string, geometry_msgs::Pose>> navigation_points = {
        {"1. 二维码区域 (qr_zone)", createPose(1.35, 0.92, 3.14)},
        {"2. 拣货区域 (pick_zone)", createPose(1.7, 5.35, 0.0)},
        {"3. 等待区域 (wait_zone)", createPose(1.7, 6.34, 0.0)},
        {"4. 交通灯区域 (traffic_zone)", createPose(4.9, 6.4, 1.57)},
        {"5. A路口入口 (intersection_A)", createPose(4.2, 4.3, -1.57)},
        {"6. B路口入口 (intersection_B)", createPose(7.3, 4.6, -1.57)},
        {"7. 终点区域A (finish_zone_A)", createPose(4.9, 0.4, -1.57)},
        {"8. 终点区域B (finish_zone_B)", createPose(6.5, 0.4, -1.57)},
        {"9. 原点 (home)", createPose(0.0, 0.0, 0.0)}
    };
    
    // 优化：使用更高效的消息队列
    ros::Subscriber cmd_vel_sub = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 5, 
        [&](const geometry_msgs::Twist::ConstPtr& msg) {
            analyzer.recordVelocityData(msg->linear.x, msg->angular.z);
        });
    
    while(ros::ok()) {
        std::cout << "\n" << std::string(40, '=') << std::endl;
        std::cout << "     导航性能分析系统" << std::endl;
        std::cout << std::string(40, '=') << std::endl;
        std::cout << "请选择测试模式:" << std::endl;
        std::cout << " 1-9. 单点测试" << std::endl;
        
        // 显示单点测试选项
        for (const auto& point : navigation_points) {
            std::cout << "  " << point.first << std::endl;
        }
        
        std::cout << " 10. 自定义坐标" << std::endl;
        std::cout << " 11. 一条龙测试 (二维码→拣货→等待→交通灯→B路口→终点A)" << std::endl;
        std::cout << " 0. 退出程序" << std::endl;
        std::cout << "请输入选项编号: ";
        
        std::string choice_str;
        std::getline(std::cin, choice_str);
        
        int choice = 0;
        try {
            choice = std::stoi(choice_str);
        } catch (...) {
            choice = -1;
        }
        
        if (choice == 0) break;
        
        if (choice == 11) {
            // 一条龙测试
            runCompleteTest(ac, analyzer);
            continue;
        }
        
        // 单点测试逻辑（保持原有代码）
        move_base_msgs::MoveBaseGoal goal;
        goal.target_pose.header.frame_id = "map";
        goal.target_pose.header.stamp = ros::Time::now();
        
        geometry_msgs::Point target_point;
        double target_yaw = 0.0;
        std::string target_name;
        
        if (choice > 0 && choice <= static_cast<int>(navigation_points.size())) {
            const auto& point = navigation_points[choice - 1];
            goal.target_pose.pose = point.second;
            target_point.x = goal.target_pose.pose.position.x;
            target_point.y = goal.target_pose.pose.position.y;
            target_point.z = 0.0;
            target_name = point.first;
            
            tf2::Quaternion q(
                goal.target_pose.pose.orientation.x,
                goal.target_pose.pose.orientation.y,
                goal.target_pose.pose.orientation.z,
                goal.target_pose.pose.orientation.w);
            tf2::Matrix3x3 m(q);
            double roll, pitch;
            m.getRPY(roll, pitch, target_yaw);
            
            ROS_INFO("选择目标: %s", point.first.c_str());
        } else if (choice == 10) {
            double x, y, yaw;
            
            std::cout << "X 坐标: ";
            std::getline(std::cin, choice_str);
            x = choice_str.empty() ? 0.0 : std::stod(choice_str);
            
            std::cout << "Y 坐标: ";
            std::getline(std::cin, choice_str);
            y = choice_str.empty() ? 0.0 : std::stod(choice_str);
            
            std::cout << "朝向角度 (度): ";
            std::getline(std::cin, choice_str);
            yaw = choice_str.empty() ? 0.0 : std::stod(choice_str);
            
            double yaw_rad = yaw * M_PI / 180.0;
            goal.target_pose.pose = createPose(x, y, yaw_rad);
            target_point.x = x;
            target_point.y = y;
            target_point.z = 0.0;
            target_yaw = yaw_rad;
            target_name = "自定义坐标 (" + std::to_string(x) + ", " + std::to_string(y) + ")";
            
            ROS_INFO("发送自定义目标");
        } else {
            std::cout << "无效选项" << std::endl;
            continue;
        }
        
        analyzer.startNavigation(target_point, target_yaw, target_name);
        ac.sendGoal(goal);
        
        std::cout << "导航进行中... (输入 'c' 取消)" << std::endl;
        
        bool finished = false;
        while(!finished && ros::ok()) {
            finished = ac.waitForResult(ros::Duration(0.5));
            ros::spinOnce();
            
            if(std::cin.rdbuf()->in_avail() > 0) {
                std::string cmd;
                std::getline(std::cin, cmd);
                if(cmd == "c" || cmd == "C") {
                    ac.cancelGoal();
                    analyzer.stopNavigation();
                    break;
                }
            }
        }
        
        if(finished) {
            analyzer.stopNavigation();
        }
        
        std::cout << "是否继续? (y/n): ";
        std::string continue_str;
        std::getline(std::cin, continue_str);
        if(continue_str == "n" || continue_str == "N") break;
    }
    
    ROS_INFO("导航分析程序退出");
    return 0;
}