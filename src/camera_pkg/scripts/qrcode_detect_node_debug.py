#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from pyzbar.pyzbar import decode
from std_srvs.srv import Trigger, TriggerResponse

class QRCodeDetect:
    def __init__(self):
        rospy.init_node("qrcode_detect_node")
        self.bridge = CvBridge()

        # 预编译优化映射表
        self.task_mapping = {
            "dessert": "食品", "fruit": "水果", "vegetable": "蔬菜", 
            "vage": "蔬菜", "drink": "饮料", "beverage": "饮料",
            "cake": "食品", "食品": "食品", "水果": "水果", 
            "蔬菜": "蔬菜", "饮料": "饮料",
        }
        
        rospy.Subscriber("/cam", Image, self.image_callback)
        self.qr_service = rospy.Service("/qr_recognition", Trigger, self.handle_qr_service)
        
        self.latest_task_type = ""
        self.processing_active = True

    def image_callback(self, msg):
        # 快速返回检查
        if not self.processing_active or self.latest_task_type:
            return
            
        try:
            # 极速处理流水线
            gray_img = self.bridge.imgmsg_to_cv2(msg, "mono8")
            decode_objs = decode(gray_img)
            
            if decode_objs:
                content = decode_objs[0].data.decode("utf-8").lower()
                self.latest_task_type = self.task_mapping.get(content, content)
                self.processing_active = False  # 识别成功，停止处理

        except Exception:
            pass

    def handle_qr_service(self, req):
        response = TriggerResponse()
        response.success = bool(self.latest_task_type)
        response.message = self.latest_task_type or "未识别到二维码"
        return response

if __name__ == "__main__":
    QRCodeDetect()
    rospy.spin()
