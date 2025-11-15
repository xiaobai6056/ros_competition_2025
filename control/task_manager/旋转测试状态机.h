#ifndef NAVIGATION_STATE_MACHINE_H
#define NAVIGATION_STATE_MACHINE_H

// ROS核心依赖
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <actionlib/client/simple_action_client.h>

// 消息类型
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/String.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <geometry_msgs/PointStamped.h>  // 新增：预识别坐标消息类型

// 服务类型
#include <std_srvs/Trigger.h>

// 标准库
#include <map>
#include <string>
#include <vector>
#include <cmath>

/**
 * @brief 机器人导航状态枚举
 * 描述机器人在任务流程中的所有可能状态
 */
enum class RobotState {
    // 初始化阶段
    INIT,                   // 初始化状态
    MOVE_TO_QR_ZONE,        // 移动到二维码识别区域
    WAITING_QR_SERVICE,     // 等待二维码识别服务结果
    
    // 物体识别阶段
    MOVE_TO_PICK_ZONE,      // 移动到拣货区域
    SCANNING_BOARDS,        // 扫描识别板（激光雷达聚类）
    NAVIGATING_TO_BOARD,    // 导航到目标识别板
    WAITING_VISUAL,         // 等待视觉识别结果
    OBJECT_CONFIRMED,       // 物体识别确认完成
    
    // 任务执行阶段
    MOVE_TO_WAIT_ZONE,      // 移动到等待区域
    WAITING_SIMULATION,     // 等待仿真任务完成
    
    // 最终导航阶段
    MOVE_TO_TRAFFIC_ZONE,   // 移动到路牌识别区域
    WAITING_TRAFFIC,        // 等待路牌识别结果
    MOVE_TO_INTERSECTION,   // 前往可通过的路口
    NAVIGATE_TO_FINISH,     // 导航到终点区域
    
    // 完成与错误状态
    TASK_COMPLETE,          // 任务完成
    ERROR                   // 错误状态（可恢复）
};

/**
 * @brief 导航状态机核心类
 * 管理机器人从初始化到任务完成的全流程状态转换与行为控制
 */
class NavigationStateMachine {
public:
    /**
     * @brief 激光雷达聚类信息结构体
     * 存储识别板聚类的关键特征（中心位置、距离、朝向等）
     */
    struct ClusterInfo {
        geometry_msgs::Point center;       // 聚类中心点（全局坐标）
        float average_distance;            // 聚类到机器人的平均距离（m）
        size_t size;                       // 聚类包含的激光点数量
        float angular_width;               // 聚类的角度跨度（rad）
        float board_yaw;                   // 识别板的朝向角（rad，全局坐标系）
    };

    /**
     * @brief 任务状态标志位结构体
     * 记录各阶段任务的执行状态（如目标是否发送、结果是否接收）
     */
    struct TaskFlags {
        bool navigation_in_progress = false;      // 导航是否正在进行
        bool qr_goal_sent = false;                // 二维码区域目标是否已发送
        bool pick_goal_sent = false;               // 拣货区目标是否已发送
        bool wait_goal_sent = false;               // 等待区目标是否已发送
        bool traffic_goal_sent = false;            // 路牌识别区目标是否已发送
        bool intersection_goal_sent_flag = false;  // 路口目标是否已发送
        bool finish_goal_sent = false;             // 终点目标是否已发送
        bool object_picked = false;                // 物体是否已确认拾取
        bool simulation_received = false;          // 仿真结果是否已接收
        bool traffic_received = false;             // 路牌识别结果是否已接收
    };

    /**
     * @brief 构造函数
     * @param nh ROS节点句柄
     */
    explicit NavigationStateMachine(ros::NodeHandle& nh);

    /**
     * @brief 状态机执行函数
     * 调用当前状态对应的处理函数，需在主循环中频繁调用
     */
    void execute();
    
private:
    // ========== 常量定义（集中管理魔法数字） ==========
    static constexpr float TARGET_OBSTACLE_DISTANCE = 0.5f;  // 障碍物检测阈值（m）
    static constexpr float DEFAULT_SAFE_DISTANCE = 0.4f;     // 默认安全距离（m）
    static constexpr float EXTENDED_SAFE_DISTANCE = 0.8f;    // 扩展安全距离（m，用于不可达区域）
    static constexpr int LASER_SCAN_TIMEOUT = 5;             // 激光扫描超时时间（s）
    static constexpr int VISUAL_RECOGNITION_TIMEOUT = 15;    // 视觉识别超时时间（s）
    static constexpr int SERVICE_RETRY_COUNT = 3;            // 服务调用重试次数

     // === 新增的旋转优化控制变量 ===
    bool rotation_optimization_active_ = false;
    float rotation_start_yaw_ = 0.0f;
    ros::Time rotation_start_time_;

    // ========== ROS 通信成员 ==========
    ros::NodeHandle nh_;                                      // 节点句柄
    
    // Action客户端
    actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> action_client_;  // move_base客户端
    
    // 发布器
    ros::Publisher tts_publisher_;       // 语音播报发布器（/tts）
    ros::Publisher cmd_vel_pub_;         // 速度控制发布器（/cmd_vel）
    ros::Publisher task_pub_;            // 当前任务发布器（/current_task）
    
    // 订阅器
    ros::Subscriber laser_sub_;          // 激光雷达订阅器（/scan）
    ros::Subscriber simulation_sub_;     // 仿真结果订阅器（/demo/simulation_result）
    ros::Subscriber traffic_sub_;        // 路牌识别结果订阅器（/demo/traffic_result）
    ros::Subscriber costmap_sub_;        // 代价地图订阅器（/move_base/global_costmap/costmap）
    ros::Subscriber pre_detection_sub_;  // 新增：预识别坐标订阅器（/pre_detection_target）

    // 服务客户端
    ros::ServiceClient qr_service_client_;       // 二维码识别服务客户端（/qr_recognition）
    ros::ServiceClient object_service_client_;   // 物体识别服务客户端（/object_recognition）
    ros::ServiceClient reset_pre_detection_client_;

    // ========== TF 相关 ==========
    tf2_ros::Buffer tf_buffer_;               // TF缓冲区
    tf2_ros::TransformListener tf_listener_;  // TF监听器

    // ========== 状态和数据成员 ==========
    RobotState current_state_;                // 当前机器人状态
    TaskFlags task_flags_;                    // 任务状态标志位
    
    // 任务数据
    std::string current_task_;                // 当前任务（从二维码识别获取）
    std::string picked_object_;               // 已拾取的物体名称
    std::string simulation_result_;           // 仿真结果（目标房间）
    std::string traffic_result_;              // 路牌识别结果（可通过路口）
    std::string current_goal_point_;          // 当前导航目标点名称
    double total_cost_ = 0.0;                 // 任务总花费
    
    // 新增：预识别坐标相关
    geometry_msgs::Point pre_detected_target_;  // 预识别目标坐标
    bool has_pre_detection_ = false;            // 是否有有效的预识别坐标
    
    // 导航点配置
    std::map<std::string, geometry_msgs::PoseStamped> navigation_points_;  // 预定义导航点

    // 代价地图相关
    nav_msgs::OccupancyGrid current_costmap_;  // 当前代价地图数据
    bool costmap_updated_ = false;             // 代价地图是否已更新

    // 激光雷达聚类相关
    std::vector<geometry_msgs::Point> detected_clusters_;       // 检测到的识别板安全目标点
    std::vector<ClusterInfo> detected_cluster_infos_;           // 检测到的识别板聚类详细信息
    bool clusters_calculated_;  // 新增：标记是否已计算过簇
    int current_target_cluster_ = -1;                            // 当前目标识别板索引
    bool clusters_detected_ = false;                             // 是否检测到识别板聚类
    bool moving_to_cluster_ = false;                             // 是否正在导航到识别板

    // 障碍物检测
    float obstacle_distance_ = std::numeric_limits<float>::max();  // 最近障碍物距离（m）
    bool obstacle_detected_ = false;                               // 是否检测到障碍物

    // 服务调用状态
    bool qr_service_called_ = false;      // 二维码服务是否已调用
    bool object_service_called_ = false;  // 物体识别服务是否已调用
    ros::Time service_call_time_;         // 服务调用时间戳
    ros::Time scan_start_time_;           // 激光扫描开始时间戳

    // TF缓存（优化性能，避免频繁查询）
    float scan_robot_x_ = 0.0f;     // 扫描时机器人X坐标（全局）
    float scan_robot_y_ = 0.0f;     // 扫描时机器人Y坐标（全局）
    float scan_robot_yaw_ = 0.0f;   // 扫描时机器人朝向角（全局，rad）

    // ========== 私有方法声明 ==========
    
    // 状态处理函数（按状态枚举顺序排列）
    void handleInitState();
    void handleMoveToQRZone();
    void handleWaitingQRService();
    void handleMoveToPickZone();
    void handleScanningBoards();
    void handleNavigatingToBoard();
    void handleWaitingVisual();
    void handleObjectConfirmed();
    void handleMoveToWaitZone();
    void handleWaitingSimulation();
    void handleMoveToTrafficZone();
    void handleWaitingTraffic();
    void handleMoveToIntersection();
    void handleNavigateToFinish();
    void handleTaskComplete();
    void handleErrorState();

    // 激光雷达与聚类处理
    void laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg);
    void detectObjectClusters(const sensor_msgs::LaserScan::ConstPtr& scan);
    ClusterInfo calculateClusterInfo(const std::vector<int>& cluster, const sensor_msgs::LaserScan::ConstPtr& scan);
    bool isValidObjectCluster(const ClusterInfo& cluster_info, const std::vector<int>& cluster, const sensor_msgs::LaserScan::ConstPtr& scan);
    float calculateBoardLength(const std::vector<int>& cluster, const sensor_msgs::LaserScan::ConstPtr& scan);
    void selectBestCluster();
    void moveToNextCluster();
    geometry_msgs::Point calculateSafeTarget(const ClusterInfo& cluster_info);
    float calculateAdaptiveSafeDistance(const geometry_msgs::Point& target_point);  // 自适应安全距离计算

    // 导航与运动控制
    void sendNavigationGoal(const std::string& point_name);
    void stopMoving();

    // ActionLib回调
    void navDoneCallback(const actionlib::SimpleClientGoalState& state, const move_base_msgs::MoveBaseResultConstPtr& result);
    void navActiveCallback();
    void navFeedbackCallback(const move_base_msgs::MoveBaseFeedbackConstPtr& feedback);
    void clusterArrivedCallback(const actionlib::SimpleClientGoalState& state, const move_base_msgs::MoveBaseResultConstPtr& result);

    // 服务调用
    bool callQRService();
    bool callObjectRecognitionService();

    // TF与位姿处理
    bool getRobotPose(float& x, float& y, float& yaw);
    bool validateTFData();  // 验证TF变换是否完整

    // 代价地图处理
    void costmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg);
    bool isPointReachable(const geometry_msgs::Point& point);
    bool isTargetReachable(const geometry_msgs::Point& point);

    // 工具函数
    void speak(const std::string& text);
    void setState(RobotState new_state);
    geometry_msgs::PoseStamped createPose(double x, double y, double yaw);
    void loadNavigationPoints();  // 加载预定义导航点
    void updateCostCalculation(const std::string& object);  // 更新任务花费
    void preDetectionCallback(const geometry_msgs::PointStamped::ConstPtr& msg);
    void resetPreDetectionData();

    // 数据回调
    void simulationCallback(const std_msgs::String::ConstPtr& msg);
    void trafficCallback(const std_msgs::String::ConstPtr& msg);
};

#endif  // NAVIGATION_STATE_MACHINE_H
