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

        
        
        

        
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback)
        self.tl_info_pub = rospy.Publisher("/demo/traffic_result", String, queue_size=10)
        self.final_img_pub = rospy.Publisher("/detect/final_image", Image, queue_size=10)
        rospy.loginfo("交通信号灯识别节点已启动，发布结果到 /demo/traffic_result")
    

    def get_tl_color(self, cv_image):
        
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
       
        if red_area > yellow_area and red_area > green_area and red_area > 50:
            return "red"
        elif yellow_area > red_area and yellow_area > green_area and yellow_area > 50:
            return "yellow"
        elif green_area > red_area and green_area > yellow_area and green_area > 50:
            return "green"
        else:
            return "unknown"
           
           
    

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            tl_info = "未检测到交通信号灯"  

            tl_color = self.get_tl_color(cv_image)
            tl_label = f"{tl_color} light" 
            
            tl_info = tl_label  
            rospy.loginfo(f"检测到交通信号灯: {tl_label}")        
                     
         
            self.tl_info_pub.publish(tl_info)
            final_img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.final_img_pub.publish(final_img_msg)

        except Exception as e:
            rospy.logerr(f"信号灯识别错误: {str(e)}")

if __name__ == "__main__":
    try:
        TrafficLightDetect()
        rospy.spin()  
    except rospy.ROSInterruptException:
        rospy.loginfo("交通信号灯识别节点已停止")
