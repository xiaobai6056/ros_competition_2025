#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger, TriggerResponse

class SimulationObjectDetector:
    def __init__(self):
        rospy.init_node('simulation_object_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # 仿真任务服务
        self.object_service = rospy.Service("/simulation_object_recognition", Trigger, self.handle_object_service)
        
        # 模型配置
        self.model_path = rospy.get_param("~model", "/home/hxx/catkin_ws/src/camera_pkg/models/best.pt")
        
        # 类别映射
        self.class_map = {
            0: ("水果", "香蕉"),
            1: ("水果", "西瓜"),  
            2: ("水果", "苹果"),
            3: ("食品", "蛋糕"),
            4: ("食品", "牛奶"),
            5: ("食品", "可乐"),
            6: ("蔬菜", "土豆"),
            7: ("蔬菜", "番茄"),
            8: ("蔬菜", "辣椒"),
        }
        
        # 加载模型
        self.model = self.load_model()
        
        # 图像订阅
        self.latest_frame = None
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback, queue_size=1)
        
        rospy.loginfo("仿真物体识别节点启动完成 - 当前帧检测")

    def load_model(self):
        """加载模型"""
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
            model.conf = 0.3
            rospy.loginfo("模型加载成功")
            return model
        except Exception as e:
            rospy.logerr("模型加载失败: {}".format(e))
            return None

    def image_callback(self, msg):
        """接收图像"""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn("图像接收异常: {}".format(e))

    def handle_object_service(self, req):
        """仿真任务服务处理 - 直接使用当前帧"""
        rospy.loginfo("=== 仿真识别服务调用 ===")
        
        try:
            if self.model is None or self.latest_frame is None:
                rospy.logwarn("模型或图像未就绪")
                return TriggerResponse(True, "NO_OBJECT_DETECTED")
            
            # 直接检测当前帧
            results = self.model(self.latest_frame)
            detections = results.pandas().xyxy[0].values
            
            best_obj = None
            best_confidence = 0
            
            for det in detections:
                confidence = det[4]
                cls_id = int(det[5])
                
                if cls_id in self.class_map and confidence >= 0.3:
                    category, obj_name = self.class_map[cls_id]
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_obj = obj_name
                        rospy.loginfo("检测到: {} (置信度: {:.3f})".format(obj_name, best_confidence))
            
            response = TriggerResponse()
            response.success = True
            
            if best_obj:
                response.message = best_obj
                rospy.loginfo("仿真识别结果: {}".format(best_obj))
            else:
                response.message = "NO_OBJECT_DETECTED"
                rospy.logwarn("当前帧未检测到物体")
            
            return response
            
        except Exception as e:
            rospy.logerr("仿真识别异常: {}".format(e))
            return TriggerResponse(False, "ERROR")

    def run(self):
        """主循环"""
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = SimulationObjectDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("仿真识别节点已停止")
