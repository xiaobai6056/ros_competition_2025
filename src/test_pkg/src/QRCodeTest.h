#ifndef QRCODE_TEST_H
#define QRCODE_TEST_H

#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/PoseStamped.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <std_srvs/Trigger.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Twist.h>
#include <map>
#include <string>

class QRCodeTest {
public:
    QRCodeTest();
    void execute();

private:
    // ROS components
    ros::NodeHandle nh_;
    actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> action_client_;
    ros::Publisher tts_publisher_;
    ros::Publisher cmd_vel_pub_;
    ros::ServiceClient qr_service_client_;
    
    // Data members
    std::map<std::string, geometry_msgs::PoseStamped> navigation_points_;
    ros::Time test_start_time_;
    ros::Time state_start_time_;
    std::string current_state_;
    
    bool qr_goal_sent_;
    bool qr_service_called_;
    bool navigation_in_progress_;
    std::string current_task_;

    // Private methods
    void handleInitState();
    void handleMoveToQRZone();
    void handleWaitingQRService();
    void handleTestComplete();
    
    bool callQRService();
    void sendNavigationGoal(const std::string& point_name);
    void navDoneCallback(const actionlib::SimpleClientGoalState& state,
                        const move_base_msgs::MoveBaseResultConstPtr& result);
    void navActiveCallback();
    void navFeedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback);
    void setState(const std::string& new_state);
    void loadNavigationPoints();
    geometry_msgs::PoseStamped createPose(double x, double y, double yaw);
    void speak(const std::string& text);
    void stopMoving();
    void printStateTime();
    void printTotalTimeStatistics();
};

#endif // QRCODE_TEST_H