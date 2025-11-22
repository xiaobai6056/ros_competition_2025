#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class TrafficLightDetect:
    def __init__(self):
        rospy.init_node("tl_detect", anonymous=True)
        self.bridge = CvBridge()
        
        # 订阅原始图像
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback)
        
        # 发布识别结果：必须与状态机中的 /demo/traffic_result 对齐
        self.tl_info_pub = rospy.Publisher("/demo/traffic_result", String, queue_size=10)
        
        # 可视化调试（可选）
        self.final_img_pub = rospy.Publisher("/detect/final_image", Image, queue_size=10)
        
        rospy.loginfo("交通信号灯识别节点已启动，发布结果到 /demo/traffic_result")

    def get_tl_color(self, cv_image):
        # HSV 阈值定义
        red_low1 = np.array([0, 120, 70])
        red_high1 = np.array([10, 255, 255])
        red_low2 = np.array([170, 120, 70])
        red_high2 = np.array([180, 255, 255])
        yellow_low = np.array([20, 120, 70])
        yellow_high = np.array([30, 255, 255])
        green_low = np.array([35, 120, 70])
        green_high = np.array([77, 255, 255])

        blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(hsv_img, red_low1, red_high1)
        red_mask2 = cv2.inRange(hsv_img, red_low2, red_high2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        yellow_mask = cv2.inRange(hsv_img, yellow_low, yellow_high)
        green_mask = cv2.inRange(hsv_img, green_low, green_high)

        red_area = cv2.countNonZero(red_mask)
        yellow_area = cv2.countNonZero(yellow_mask)
        green_area = cv2.countNonZero(green_mask)

        min_area = 50
        if green_area > red_area and green_area > yellow_area and green_area > min_area:
            return "green"
        elif red_area > min_area or yellow_area > min_area:
            # 只要不是 green，都认为不可通行
            return "not_green"
        else:
            return "unknown"

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            color = self.get_tl_color(cv_image)

           
            if color == "green":
                
                traffic_result = "intersection_1"  # 必须与 navigation_points_ 中的 key 一致
            else:
                
                traffic_result = "none"  

            rospy.loginfo(f"交通灯识别结果: {color} → 发布路口: '{traffic_result}'")
            self.tl_info_pub.publish(traffic_result)

            #发布带标注的图像用于调试
            annotated_img = cv_image.copy()
            cv2.putText(annotated_img, f"TL: {color}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            final_img_msg = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8")
            self.final_img_pub.publish(final_img_msg)

        except Exception as e:
            rospy.logerr(f"信号灯识别错误: {str(e)}")

if __name__ == "__main__":
    try:
        detector = TrafficLightDetect()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("交通信号灯识别节点已停止")