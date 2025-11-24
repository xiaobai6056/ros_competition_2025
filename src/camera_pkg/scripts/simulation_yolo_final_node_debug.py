#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import torch
import os
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger, TriggerResponse

# ===================== 核心修改1：添加YOLO类的导入 =====================
# 先添加ultralytics的库路径（防止ROS环境找不到）
sys.path.insert(0, "/home/hxx/anaconda3/envs/yolov11/lib/python3.8/site-packages")
try:
    from ultralytics import YOLO
    print("✅ 成功导入ultralytics的YOLO类")
except ImportError as e:
    print(f"❌ 导入YOLO失败: {e}")
    sys.exit(1)

# 全局禁用CUDA（如果需要）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
torch.cuda.is_available = lambda: False

class SimulationObjectDetector:
    def __init__(self):
        rospy.init_node('simulation_object_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # 仿真任务服务
        self.object_service = rospy.Service("/simulation_object_recognition", Trigger, self.handle_object_service)
        
        # 模型配置
        self.model_path = rospy.get_param("~model", "/home/hxx/catkin_ws/src/camera_pkg/models/best2.pt")
        
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
        
        # 加载模型 - 核心修改：使用YOLO11方式
        self.model = self.load_model()
        
        # 图像订阅
        self.latest_frame = None
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback, queue_size=1)
        
        rospy.loginfo("仿真物体识别节点启动完成 - YOLOv11当前帧检测")

    def load_model(self):
        """加载模型 - 使用YOLO11官方API"""
        try:
            rospy.loginfo(f"加载YOLO11模型，权重路径: {self.model_path}")
            
            # 核心修改：使用YOLO类加载权重
            model = YOLO(self.model_path)
            
            # 设置模型参数
            model.conf = 0.3  # 置信度阈值
            model.iou = 0.4   # NMS IoU阈值
            model.max_det = 10  # 最大检测数
            
            rospy.loginfo("YOLO11模型加载成功（CPU模式）！")
            return model
            
        except Exception as e:
            rospy.logerr(f"模型加载失败: {e}")
            return None

    def image_callback(self, msg):
        """接收图像 - 保持不变"""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn("图像接收异常: {}".format(e))

    def handle_object_service(self, req):
        """仿真任务服务处理 - 修改为YOLOv11推理方式"""
        rospy.loginfo("=== 仿真识别服务调用 ===")
        
        try:
            if self.model is None or self.latest_frame is None:
                rospy.logwarn("模型或图像未就绪")
                return TriggerResponse(True, "NO_OBJECT_DETECTED")
            
            # 核心修改：YOLO11推理方式
            results = self.model(self.latest_frame, verbose=False)
            
            best_obj = None
            best_confidence = 0
            
            # 解析YOLO11的检测结果
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # 提取检测框坐标、置信度、类别ID
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        if cls_id in self.class_map and conf >= 0.3:
                            category, obj_name = self.class_map[cls_id]
                            if conf > best_confidence:
                                best_confidence = conf
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
        """主循环 - 保持不变"""
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = SimulationObjectDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("仿真识别节点已停止")
