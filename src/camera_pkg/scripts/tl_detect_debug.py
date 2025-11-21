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
        
        # 输出频率控制
        self.last_log_time = rospy.Time.now()
        self.log_interval = rospy.Duration(2.0)
        
        # 存储上一次的识别结果（仅用于日志输出控制）
        self.last_tl_color = ""
        self.last_intersection_result = ""
        
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback)
        self.tl_info_pub = rospy.Publisher("/demo/traffic_result", String, queue_size=10)
        self.final_img_pub = rospy.Publisher("/detect/final_image", Image, queue_size=10)
        rospy.loginfo("交通信号灯识别节点已启动，发布结果到 /demo/traffic_result")
    
    def get_tl_color(self, cv_image):
        """
        准确识别当前帧的交通灯颜色
        返回: "red", "green", 或 "unknown"
        """
        # 精确的HSV颜色范围
        # 红色范围
        red_low1 = np.array([0, 150, 150])
        red_high1 = np.array([10, 255, 255])
        red_low2 = np.array([170, 150, 150])
        red_high2 = np.array([180, 255, 255])
        
        # 绿色范围
        green_low = np.array([45, 100, 100])
        green_high = np.array([85, 255, 255])

        # 图像预处理
        blurred = cv2.GaussianBlur(cv_image, (7, 7), 0)
        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 创建颜色掩码
        red_mask1 = cv2.inRange(hsv_img, red_low1, red_high1)
        red_mask2 = cv2.inRange(hsv_img, red_low2, red_high2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_mask = cv2.inRange(hsv_img, green_low, green_high)
        
        # 形态学操作去除噪声
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # 计算颜色区域面积
        red_area = cv2.countNonZero(red_mask)
        green_area = cv2.countNonZero(green_mask)
        
        # 设置合理的面积阈值
        min_area = 50
        
        # 简单的当前帧判断逻辑
        if red_area > min_area and red_area > green_area:
            return "red"
        elif green_area > min_area and green_area > red_area:
            return "green"
        else:
            return "unknown"
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 获取当前帧的信号灯颜色（准确返回当前帧结果）
            tl_color = self.get_tl_color(cv_image)
            
            # 根据信号灯颜色返回路口标识
            if tl_color == "green":
                intersection_result = "A"  # 绿灯返回A路口
            elif tl_color == "red":
                intersection_result = "B"  # 红灯返回B路口
            else:
                intersection_result = "unknown"
            
            # 控制输出频率：只在结果变化或超过时间间隔时输出
            current_time = rospy.Time.now()
            should_log = False
            
            # 如果结果发生变化，立即输出
            if (tl_color != self.last_tl_color or 
                intersection_result != self.last_intersection_result):
                should_log = True
                self.last_tl_color = tl_color
                self.last_intersection_result = intersection_result
            # 如果超过时间间隔，输出当前状态
            elif (current_time - self.last_log_time) > self.log_interval:
                should_log = True
                self.last_log_time = current_time
            
            # 根据条件输出日志
            if should_log:
                if intersection_result == "A":
                    rospy.loginfo("🚦 当前帧识别：绿灯亮起，A路口可通过")
                elif intersection_result == "B":
                    rospy.loginfo("🚦 当前帧识别：红灯亮起，B路口可通过")
                else:
                    rospy.loginfo(f"🚦 当前帧识别：未知信号灯状态: {tl_color}")
            
            # 重要：每次都发布当前帧的识别结果，不进行历史平均
            self.tl_info_pub.publish(intersection_result)
            
            # 在图像上标注识别结果（用于调试）
            annotated_image = cv_image.copy()
            status_text = f"Current: {tl_color} -> {intersection_result}"
            cv2.putText(annotated_image, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 发布处理后的图像
            final_img_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            self.final_img_pub.publish(final_img_msg)

        except Exception as e:
            rospy.logerr(f"信号灯识别错误: {str(e)}")

if __name__ == "__main__":
    try:
        TrafficLightDetect()
        rospy.spin()  
    except rospy.ROSInterruptException:
        rospy.loginfo("交通信号灯识别节点已停止")
