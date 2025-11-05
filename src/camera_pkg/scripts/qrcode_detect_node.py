#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from pyzbar.pyzbar import decode


class QRCodeDetect:
    def __init__(self):
        rospy.init_node("qrcode_detect_node", anonymous=True)
        self.bridge = CvBridge()

        
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback)
        
        
        self.qrcode_pub = rospy.Publisher("/demo/qr_result", String, queue_size=10)
        self.annotated_img_pub = rospy.Publisher("/detect/image_with_qr_code", Image, queue_size=10)

        rospy.loginfo("QR Code Detection started--Publishing to /demo/qr_result")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            qrcode_content = ""  

            
            gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            decode_objs = decode(gray_img)  
            if decode_objs:
                for obj in decode_objs:
                    
                    qrcode_content = obj.data.decode("utf-8")
                    rospy.loginfo(f"识别到二维码: {qrcode_content}")

                   
                    points = obj.polygon
                    if len(points) == 4:
                        pts = np.array(points, dtype=np.int32).reshape(-1, 2)
                        cv2.polylines(cv_image, [pts], True, (0, 255, 0), 2)

                    
                    cv2.putText(
                        cv_image, 
                        qrcode_content,
                        (obj.rect[0], obj.rect[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2
                    )

            
            self.qrcode_pub.publish(qrcode_content)
            
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
