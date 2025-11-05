#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraInput:
    def __init__(self):
        rospy.init_node("camera_input_node")
        self.bridge = CvBridge()  # ROS图像↔OpenCV转换工具
        self.latest_frame = None  # 存储最新图像
        
        rospy.Subscriber("/cam", Image, self.image_callback)
        self.image_pub = rospy.Publisher("/detect/raw_image", Image, queue_size=10)
        
        cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Preview", 640, 480)

        rospy.loginfo("摄像头转发节点启动：从/cam转发到/detect/raw_image")

    def image_callback(self, msg):
        
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            self.image_pub.publish(msg)

        except Exception as e:
            rospy.logerr(f"图像处理失败：{str(e)}")

    def run(self):
        
        rate = rospy.Rate(30)  # 30FPS刷新率
        while not rospy.is_shutdown():
            if self.latest_frame is not None:
                cv2.imshow("Camera Preview", self.latest_frame)
                cv2.waitKey(1)  
            rate.sleep()
        cv2.destroyAllWindows()  

if __name__ == "__main__":
    try:
        cam_input = CameraInput()
        cam_input.run()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        rospy.loginfo("摄像头转发节点停止")
