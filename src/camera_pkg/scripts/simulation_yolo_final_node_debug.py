#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import torch
import os
import sys
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from std_srvs.srv import Trigger, TriggerResponse

# ===================== 核心修改1：添加YOLO类的导入 =====================
sys.path.insert(0, "/home/hxx/anaconda3/envs/yolov11/lib/python3.8/site-packages")
try:
    from ultralytics import YOLO
    print("✅ 成功导入ultralytics的YOLO类")
except ImportError as e:
    print(f"❌ 导入YOLO失败: {e}")
    sys.exit(1)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
torch.cuda.is_available = lambda: False

class SimulationObjectDetector:
    def __init__(self):
        rospy.init_node('simulation_object_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # 仿真任务服务
        self.object_service = rospy.Service("/simulation_object_recognition", Trigger, self.handle_object_service)
        
        # 🆕 新增：任务类型订阅
        self.current_task = ""
        self.task_sub = rospy.Subscriber("/simulation_start", String, self.task_callback)
        
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
        
        # 🆕 新增：任务类型到物品的映射
        self.task_to_objects = {
            "水果": ["香蕉", "西瓜", "苹果"],
            "食品": ["蛋糕", "牛奶", "可乐"], 
            "蔬菜": ["土豆", "番茄", "辣椒"]
        }
        
        # 加载模型
        self.model = self.load_model()
        
        # 图像订阅
        self.latest_frame = None
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback, queue_size=1)
        
        rospy.loginfo("仿真物体识别节点启动完成 - 基于老版本+任务匹配")

    def task_callback(self, msg):
        """🆕 新增：任务回调"""
        self.current_task = msg.data
        rospy.loginfo("🎯 收到任务类型: {}".format(self.current_task))

    def load_model(self):
        """加载模型"""
        try:
            rospy.loginfo(f"加载YOLO11模型，权重路径: {self.model_path}")
            model = YOLO(self.model_path)
            model.conf = 0.3
            model.iou = 0.4
            model.max_det = 10
            rospy.loginfo("YOLO11模型加载成功（CPU模式）！")
            return model
        except Exception as e:
            rospy.logerr(f"模型加载失败: {e}")
            return None

    def image_callback(self, msg):
        """接收图像"""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn("图像接收异常: {}".format(e))

    def does_object_match_task(self, obj_name, category):
        """🆕 新增：检查物体是否匹配当前任务"""
        if not self.current_task:
            rospy.loginfo("⚠️ 无任务类型，所有物体都匹配")
            return True
        
        # 检查物体类别是否匹配任务类型
        if category == self.current_task:
            return True
        
        # 额外检查：物体是否在任务对应的物品列表中
        if self.current_task in self.task_to_objects:
            if obj_name in self.task_to_objects[self.current_task]:
                return True
        
        rospy.loginfo("❌ 物体不匹配任务: {} (需要: {})".format(obj_name, self.current_task))
        return False

    def handle_object_service(self, req):
        """仿真任务服务处理 - 修改为支持任务匹配"""
        rospy.loginfo("=== 仿真识别服务调用 ===")
        rospy.loginfo("当前任务: {}".format(self.current_task))
        
        try:
            if self.model is None or self.latest_frame is None:
                rospy.logwarn("模型或图像未就绪")
                return TriggerResponse(True, "NO_OBJECT_DETECTED")
            
            # YOLO11推理
            results = self.model(self.latest_frame, verbose=False)
            
            best_matching_obj = None
            best_matching_conf = 0
            best_non_matching_obj = None
            best_non_matching_conf = 0
            
            # 解析检测结果
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        if cls_id in self.class_map and conf >= 0.3:
                            category, obj_name = self.class_map[cls_id]
                            
                            # 🆕 新增：任务匹配检查
                            matches_task = self.does_object_match_task(obj_name, category)
                            
                            if matches_task:
                                if conf > best_matching_conf:
                                    best_matching_conf = conf
                                    best_matching_obj = obj_name
                                    rospy.loginfo("✅ 匹配物体: {} (置信度: {:.3f})".format(obj_name, conf))
                            else:
                                if conf > best_non_matching_conf:
                                    best_non_matching_conf = conf
                                    best_non_matching_obj = obj_name
                                    rospy.loginfo("⚠️ 不匹配物体: {} (置信度: {:.3f})".format(obj_name, conf))
            
            response = TriggerResponse()
            response.success = True
            
            # 🆕 修改返回逻辑：支持任务匹配
            if best_matching_obj and best_matching_conf >= 0.4:
                # 找到匹配任务的物体
                response.message = best_matching_obj
                rospy.loginfo("🎯 返回匹配物体: {}".format(best_matching_obj))
            elif best_non_matching_obj and best_non_matching_conf >= 0.4:
                # 找到不匹配任务的物体，返回警告
                response.message = "WARN:" + best_non_matching_obj
                rospy.logwarn("🚨 返回不匹配物体警告: {}".format(best_non_matching_obj))
            else:
                # 没有检测到任何物体
                response.message = "NO_OBJECT_DETECTED"
                rospy.logwarn("❌ 未检测到任何物体")
            
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
