#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/PoseStamped.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <std_srvs/Trigger.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Twist.h>
#include <chrono>
#include <iomanip>

class QRCodeTest {
private:
    ros::NodeHandle nh_;
    
    // Action client
    actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> action_client_;
    
    // Publishers
    ros::Publisher tts_publisher_;
    ros::Publisher cmd_vel_pub_;
    
    // Service client
    ros::ServiceClient qr_service_client_;
    
    // Navigation points
    std::map<std::string, geometry_msgs::PoseStamped> navigation_points_;
    
    // Timing
    ros::Time test_start_time_;
    ros::Time state_start_time_;
    std::string current_state_;
    
    // Flags
    bool qr_goal_sent_;
    bool qr_service_called_;
    bool navigation_in_progress_;
    std::string current_task_;

public:
    QRCodeTest() : 
        action_client_("move_base", true),
        qr_goal_sent_(false),
        qr_service_called_(false),
        navigation_in_progress_(false),
        current_state_("INIT")
    {
        // Initialize publishers
        tts_publisher_ = nh_.advertise<std_msgs::String>("/tts", 1);
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        
        // Initialize service client
        qr_service_client_ = nh_.serviceClient<std_srvs::Trigger>("/qr_recognition");
        
        // Load navigation points
        loadNavigationPoints();
        
        // Wait for action server
        ROS_INFO("等待move_base action server...");
        if (action_client_.waitForServer(ros::Duration(5.0))) {
            ROS_INFO("move_base action server连接成功");
        } else {
            ROS_WARN("move_base action server连接超时");
        }
        
        test_start_time_ = ros::Time::now();
        state_start_time_ = ros::Time::now();
        
        ROS_INFO("二维码识别测试初始化完成");
    }

    void execute() {
        printStateTime();
        
        if (current_state_ == "INIT") {
            handleInitState();
        } else if (current_state_ == "MOVE_TO_QR_ZONE") {
            handleMoveToQRZone();
        } else if (current_state_ == "WAITING_QR_SERVICE") {
            handleWaitingQRService();
        } else if (current_state_ == "TEST_COMPLETE") {
            handleTestComplete();
        }
    }

private:
    void handleInitState() {
        ROS_INFO("[INIT] 开始二维码识别测试");
        speak("开始二维码识别测试");
        setState("MOVE_TO_QR_ZONE");
    }

    void handleMoveToQRZone() {
        if (!qr_goal_sent_) {
            ROS_INFO("[MOVE_TO_QR_ZONE] 前往二维码区域");
            speak("正在前往二维码区域");
            sendNavigationGoal("qr_zone");
            qr_goal_sent_ = true;
        }
        
        // 时间统计
        double time_in_state = (ros::Time::now() - state_start_time_).toSec();
        ROS_INFO_THROTTLE(2, "[MOVE_TO_QR_ZONE] 导航中... 已耗时: %.1f 秒", time_in_state);
    }

    void handleWaitingQRService() {
        if (!qr_service_called_) {
            ROS_INFO("[WAITING_QR_SERVICE] 持续识别二维码...");
            
            // 简单无限重试
            if (callQRService()) {
                qr_service_called_ = true;
            } else {
                ros::Duration(0.02).sleep(); // 避免CPU占用过高
            }
        }
    }

    void handleTestComplete() {
        static bool complete_announced = false;
        
        if (!complete_announced) {
            ROS_INFO("[TEST_COMPLETE] 二维码识别测试完成");
            
            // 输出总时间统计
            printTotalTimeStatistics();
            
            speak("二维码识别测试完成，识别到的任务是" + current_task_);
            complete_announced = true;
        }
        
        ROS_INFO_THROTTLE(5, "[TEST_COMPLETE] 测试已完成，按Ctrl+C退出程序");
    }

    bool callQRService() {
        std_srvs::Trigger srv;
        
        ROS_INFO("调用二维码识别服务...");
        auto start_time = std::chrono::steady_clock::now();
        
        if (qr_service_client_.call(srv)) {
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            if (srv.response.success) {
                current_task_ = srv.response.message;
                ROS_INFO("✅ 二维码识别成功!");
                ROS_INFO("   识别结果: %s", current_task_.c_str());
                ROS_INFO("   服务耗时: %ld 毫秒", duration.count());
                
                speak("二维码识别成功，任务是" + current_task_);
                setState("TEST_COMPLETE");
                return true;
            } else {
                ROS_ERROR("❌ 二维码识别失败: %s", srv.response.message.c_str());
                ROS_ERROR("   服务耗时: %ld 毫秒", duration.count());
                return false;
            }
        } else {
            ROS_ERROR("🚫 无法调用二维码服务，请检查服务是否启动");
            return false;
        }
    }

    void sendNavigationGoal(const std::string& point_name) {
        auto it = navigation_points_.find(point_name);
        if (it != navigation_points_.end()) {
            if (navigation_in_progress_) {
                action_client_.cancelAllGoals();
                ROS_INFO("取消之前的导航目标");
            }
            
            move_base_msgs::MoveBaseGoal goal;
            goal.target_pose = it->second;
            
            action_client_.sendGoal(goal,
                boost::bind(&QRCodeTest::navDoneCallback, this, _1, _2),
                boost::bind(&QRCodeTest::navActiveCallback, this),
                boost::bind(&QRCodeTest::navFeedbackCallback, this, _1));
            
            navigation_in_progress_ = true;
            ROS_INFO("发送导航目标: %s", point_name.c_str());
            
        } else {
            ROS_ERROR("未知的导航点: %s", point_name.c_str());
        }
    }

    void navDoneCallback(const actionlib::SimpleClientGoalState& state,
                        const move_base_msgs::MoveBaseResultConstPtr& result) {
        navigation_in_progress_ = false;

        ROS_INFO("导航完成回调 - 状态: %s", state.toString().c_str());

        if (state == actionlib::SimpleClientGoalState::SUCCEEDED) {
            ROS_INFO("✅ 成功到达二维码区域");
            
            if (current_state_ == "MOVE_TO_QR_ZONE") {
                setState("WAITING_QR_SERVICE");
            }
        } else {
            ROS_ERROR("❌ 导航失败: %s", state.getText().c_str());
            
            // 失败后重试
            ROS_INFO("2秒后重新尝试导航...");
            ros::Duration(2.0).sleep();
            qr_goal_sent_ = false;
        }
    }

    void navActiveCallback() {
        ROS_INFO("导航目标已激活");
    }

    void navFeedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback) {
        ROS_INFO_THROTTLE(5, "导航进度 - 当前位置: (%.2f, %.2f)", 
                         feedback->base_position.pose.position.x,
                         feedback->base_position.pose.position.y);
    }

    void setState(const std::string& new_state) {
        ros::Time current_time = ros::Time::now();
        double duration = (current_time - state_start_time_).toSec();
        
        ROS_INFO("状态转换: %s (%.1f 秒) -> %s", 
                 current_state_.c_str(), duration, new_state.c_str());
        
        current_state_ = new_state;
        state_start_time_ = current_time;
    }

    void loadNavigationPoints() {
        // 二维码区域坐标 - 根据您的实际环境调整
        navigation_points_["qr_zone"] = createPose(1.35, 0.92, 3.14);
        
        ROS_INFO("加载导航点: qr_zone");
    }

    geometry_msgs::PoseStamped createPose(double x, double y, double yaw) {
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

    void speak(const std::string& text) {
        std_msgs::String msg;
        msg.data = text;
        tts_publisher_.publish(msg);
        ROS_INFO("语音播报: %s", text.c_str());
    }

    void stopMoving() {
        geometry_msgs::Twist stop_twist;
        cmd_vel_pub_.publish(stop_twist);
        ROS_INFO("停止移动");
    }

    void printStateTime() {
        static ros::Time last_print_time = ros::Time::now();
        ros::Time current_time = ros::Time::now();
        
        if ((current_time - last_print_time).toSec() >= 2.0) {
            double time_in_state = (current_time - state_start_time_).toSec();
            ROS_INFO_THROTTLE(1, "[%s] 当前状态已持续: %.1f 秒", 
                             current_state_.c_str(), time_in_state);
            last_print_time = current_time;
        }
    }

    void printTotalTimeStatistics() {
        ros::Time current_time = ros::Time::now();
        double total_time = (current_time - test_start_time_).toSec();
        
        ROS_INFO("========== 二维码测试时间统计 ==========");
        ROS_INFO("总测试时间: %.1f 秒", total_time);
        ROS_INFO("识别到的任务: %s", current_task_.c_str());
        ROS_INFO("测试结果: %s", current_task_.empty() ? "失败" : "成功");
        ROS_INFO("========================================");
    }
};

int main(int argc, char** argv) {
    setlocale(LC_ALL,"");
    ros::init(argc, argv, "qrcode_test");
    
    ROS_INFO("=== 二维码识别独立测试程序启动 ===");
    
    QRCodeTest qr_test;
    
    ros::Rate rate(10); // 10Hz
    
    while (ros::ok()) {
        qr_test.execute();
        ros::spinOnce();
        rate.sleep();
    }
    
    return 0;
}