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

        
        self.hsv_ranges = {
            "red": [
                (0, 120, 70), (10, 255, 255),   
                (170, 120, 70), (180, 255, 255) 
            ],
            "yellow": [(20, 120, 70), (35, 255, 255)],  
            "green": [(40, 120, 70), (77, 255, 255)]    
        }
        
        self.min_contour_area = 50
        self.max_contour_area = 2000

        
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback)
        self.tl_info_pub = rospy.Publisher("/demo/traffic_result", String, queue_size=10)
        self.final_img_pub = rospy.Publisher("/detect/final_image", Image, queue_size=10)
        rospy.loginfo("交通信号灯识别节点已启动，发布结果到 /demo/traffic_result")
    

    def get_tl_color(self, cv_image):
        h, w = cv_image.shape[:2]  
        tl_color = None  
        tl_bbox = None   
        blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

       
        for color_name, ranges in self.hsv_ranges.items():
            if color_name == "red":
                mask1 = cv2.inRange(hsv_img, ranges[0], ranges[1])
                mask2 = cv2.inRange(hsv_img, ranges[2], ranges[3])
                color_mask = cv2.bitwise_or(mask1, mask2)  
            else:
                color_mask = cv2.inRange(hsv_img, ranges[0], ranges[1])  

           
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            
            for cnt in contours:
                contour_area = cv2.contourArea(cnt)
                if not (self.min_contour_area < contour_area < self.max_contour_area):
                    continue
                perimeter = cv2.arcLength(cnt, True)  
                if perimeter == 0:
                    continue 
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)  
                if len(approx) < 8: 
                    continue
                x, y, w_cnt, h_cnt = cv2.boundingRect(approx)
                if contour_area > max_light_area:
                    max_light_area = contour_area
                    tl_color = color_name
                    tl_bbox = (
                        max(0, x - 20),
                        max(0, y - 20),
                        min(w, x + w_cnt + 20),
                        min(h, y + h_cnt + 20)
                    )

        return tl_color, tl_bbox
    

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            tl_info = "未检测到交通信号灯"  

            tl_color, tl_bbox = self.get_tl_color(cv_image)
            if tl_color and tl_bbox:
                x1, y1, x2, y2 = tl_bbox
                
                color_bgr_map = {
                    "red": (0, 0, 255),    
                    "yellow": (0, 255, 255), 
                    "green": (0, 255, 0)   
                }
               
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color_bgr_map[tl_color], 2)
                tl_label = f"{tl_color} light" 
                cv2.putText(
                    cv_image, tl_label,
                    (x1, max(0, y1 - 10)),  
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr_map[tl_color], 2
                )
                tl_info = tl_label  

         
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
