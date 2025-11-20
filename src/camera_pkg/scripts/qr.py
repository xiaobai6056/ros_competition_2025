#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from pyzbar.pyzbar import decode
from std_srvs.srv import Trigger, TriggerResponse


class QRCodeDetect:
    def __init__(self):
        rospy.init_node("qrcode_detect_node", anonymous=True)
        self.bridge = CvBridge()

        # 二维码内容到任务类型的映射
        self.task_mapping = {
            # 英文映射
            "dessert": "食品",
            "Dessert": "食品",
            "fruit": "水果", 
            "Fruit": "水果",
            "vegetable": "蔬菜",
            "Vegetable": "蔬菜",
            "vage": "蔬菜",
            "Vage": "蔬菜",
            "drink": "饮料",
            "Drink": "饮料",
            "beverage": "饮料",
            "Beverage": "饮料",
            # 中文直接支持
            "食品": "食品",
            "水果": "水果", 
            "蔬菜": "蔬菜",
            "饮料": "饮料",
            # 默认值处理
            "cake": "食品",
            "Cake": "食品"
        }
        
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback)
        
        
        self.qr_service = rospy.Service("/qr_recognition", Trigger, self.handle_qr_service)

        # 添加发布 /demo/qr_result 的 Publisher
        self.qr_result_pub = rospy.Publisher("/demo/qr_result", String, queue_size=1)
        
        
        self.annotated_img_pub = rospy.Publisher("/detect/image_with_qr_code", Image, queue_size=10)

        
        self.latest_task_type = ""
        self.latest_qr_content = ""
        
        rospy.loginfo("QR Code Detection started--Service: /qr_recognition")

    def handle_qr_service(self, req):
        
        rospy.loginfo("收到二维码识别服务请求")
        
        response = TriggerResponse()
        
        if self.latest_task_type:
            response.success = True
            response.message = self.latest_task_type
            rospy.loginfo(f"二维码服务返回: {self.latest_task_type} (来自: {self.latest_qr_content})")
        else:
            response.success = False
            response.message = "未识别到二维码"
            rospy.logwarn("二维码识别失败：未识别到二维码")
        
        return response

    def map_qr_content_to_task(self, qr_content):
        
        if not qr_content:
            return ""
            
        
        if qr_content in self.task_mapping:
            return self.task_mapping[qr_content]
        
        
        qr_lower = qr_content.lower()
        for key, value in self.task_mapping.items():
            if key.lower() == qr_lower:
                return value
        
        
        rospy.logwarn(f"未映射的二维码内容: {qr_content}")
        return qr_content

    def image_callback(self, msg):
        """持续识别二维码并更新最新结果"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            qrcode_content = ""  
            task_type = ""

            
            gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            decode_objs = decode(gray_img)  
            
            if decode_objs:
                for obj in decode_objs:
                   
                    qrcode_content = obj.data.decode("utf-8")
                    
                    
                    task_type = self.map_qr_content_to_task(qrcode_content)
                    
                    
                    self.latest_task_type = task_type
                    self.latest_qr_content = qrcode_content
                    
                    rospy.loginfo(f"识别到二维码: {qrcode_content} -> 任务: {task_type}")

                    
                    points = obj.polygon
                    if len(points) == 4:
                        pts = np.array(points, dtype=np.int32).reshape(-1, 2)
                        cv2.polylines(cv_image, [pts], True, (0, 255, 0), 2)

                    
                    display_text = f"{qrcode_content} -> {task_type}" if task_type else qrcode_content
                    cv2.putText(
                        cv_image, 
                        display_text,
                        (obj.rect[0], obj.rect[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2
                    )

                    # 新增：发布到 /demo/qr_result <<<
                    if task_type:
                        qr_msg = String()
                        qr_msg.data = task_type
                        self.qr_result_pub.publish(qr_msg)
                        rospy.logdebug(f"Published to /demo/qr_result: {task_type}")
            else:
                
                rospy.logdebug("未识别到二维码")

            
            annotated_img = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.annotated_img_pub.publish(annotated_img)

        except Exception as e:
            rospy.logerr(f"二维码识别错误: {str(e)}")


if __name__ == "__main__":
    try:
        QRCodeDetect()
        rospy.spin()  
    except rospy.ROSInterruptException:
        rospy.loginfo("QR Code Detection stopped")