#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
import time
import math
from collections import deque
from cv_bridge import CvBridge

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # 服务接口
        self.object_service = rospy.Service("/object_recognition", Trigger, self.handle_object_service)
        self.reset_service = rospy.Service("/reset_vision_state", Trigger, self.handle_reset_service)
        
        # 物体检测信号发布
        self.object_detected_pub = rospy.Publisher("/object_detected", String, queue_size=1)
        
        # 配置参数
        self.config = {
            'min_confidence': 0.40,
            'camera_hfov': 1.3962634,
            'image_width': 1920,
            'image_height': 1080,
        }
        
        # 状态管理
        self.session_active = False
        self.current_task = ""
        self.detection_history = deque(maxlen=15)
        self.service_called = False
        self.frame_counter = 0
        
        # 订阅当前任务类型
        self.task_sub = rospy.Subscriber("/current_task", String, self.task_callback)
        
        # 图像发布
        self.annotated_img_pub = rospy.Publisher("/detect/object_annotated", Image, queue_size=5)
        
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
        
        # 初始化模型
        self.model = self.load_model()
        
        # 订阅摄像头图像
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback, queue_size=1)
        
        rospy.loginfo("物体识别节点启动完成 - 简化模式")

    def load_model(self):
        """加载模型"""
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
            model.conf = self.config['min_confidence']
            model.iou = 0.4
            model.max_det = 20
            
            if torch.cuda.is_available():
                model.cuda()
            model.eval()
            
            rospy.loginfo("模型加载成功")
            return model
            
        except Exception as e:
            rospy.logerr("模型加载失败: {}".format(e))
            return None

    def is_valid_detection(self, detection, obj_name):
        """检查检测框尺寸合理性"""
        x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        
        min_size = 80
        max_size = 450
        
        if bbox_height < min_size or bbox_width < min_size:
            return False
        
        if bbox_height > max_size or bbox_width > max_size:
            return False
        
        if obj_name == '西瓜' and bbox_height < 100:
            return False
            
        aspect_ratio = bbox_width / bbox_height
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            return False
        
        return True

    def get_class_specific_threshold(self, obj_name):
        """为易误识别类别设置更高阈值"""
        high_threshold_classes = {
            '西瓜': 0.65,   
            '蛋糕': 0.45,   
            '香蕉': 0.45,   
            '苹果': 0.45,
            '牛奶': 0.45,
            '可乐': 0.45,
            '土豆': 0.45,
            '番茄': 0.45,
            '辣椒': 0.45
        }
        return high_threshold_classes.get(obj_name, self.config['min_confidence'])

    def task_callback(self, msg):
        """任务回调"""
        self.current_task = msg.data
        rospy.loginfo("任务类型更新: {}".format(self.current_task))

    def image_callback(self, msg):
        """图像回调"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.frame_counter += 1
            
            self.detection_pipeline(cv_image, self.frame_counter)
            
        except Exception as e:
            rospy.logwarn("图像回调异常: {}".format(e))

    def detection_pipeline(self, frame, frame_id):
        """检测流水线"""
        if self.model is None:
            return
            
        try:
            with torch.no_grad():
                results = self.model(frame)
                detections = results.pandas().xyxy[0].values
                
                if len(detections) > 0:
                    self.process_detections(detections, frame, frame_id)
                else:
                    if not self.service_called and len(self.detection_history) % 20 == 0:
                        self.publish_object_detected("")
                    
        except Exception as e:
            rospy.logwarn("检测流水线异常: {}".format(e))

    def process_detections(self, detections, frame, frame_id):
        """处理检测结果"""
        current_time = time.time()
        
        for i, detection in enumerate(detections):
            confidence = detection[4]
            cls_id = int(detection[5])
            
            if cls_id not in self.class_map:
                continue
                
            category, obj_name = self.class_map[cls_id]
            
            confidence_threshold = self.get_class_specific_threshold(obj_name)
            
            if confidence >= confidence_threshold:
                if not self.is_valid_detection(detection, obj_name):
                    continue
                    
                detection_record = {
                    'object': obj_name,
                    'category': category,
                    'confidence': confidence,
                    'timestamp': current_time,
                    'frame_id': frame_id
                }
                
                self.detection_history.append(detection_record)
                
                # 高置信度检测立即发布信号
                if (confidence >= 0.75 and not self.service_called):
                    rospy.loginfo("高置信度检测: {} (置信度: {:.3f})".format(obj_name, confidence))
                    self.publish_object_detected(obj_name)

    def publish_object_detected(self, object_name):
        """发布物体检测信号"""
        try:
            msg = String()
            msg.data = object_name
            self.object_detected_pub.publish(msg)
            if object_name:
                rospy.loginfo("发布物体检测信号: {}".format(object_name))
                
        except Exception as e:
            rospy.logwarn("物体检测信号发布异常: {}".format(e))

    def handle_object_service(self, req):
        """服务处理 - 修复版本：降低对历史数据的要求"""
        rospy.loginfo("=== 收到识别请求 ===")
        rospy.loginfo("当前任务: {}".format(self.current_task))
        rospy.loginfo("检测历史长度: {}".format(len(self.detection_history)))
        
        response = TriggerResponse()
        response.success = True
        
        if self.service_called:
            rospy.logwarn("服务正在处理中，拒绝重复调用")
            response.message = "SERVICE_BUSY"
            return response
        
        self.session_active = True
        self.service_called = True
        
        try:
            current_time = time.time()
            
            # 放宽时间窗口到5秒，降低数量要求
            recent_detections = []
            for det in self.detection_history:
                time_diff = current_time - det['timestamp']
                if time_diff < 5.0:  # 从3秒增加到5秒
                    recent_detections.append(det)
            
            rospy.loginfo("最近5秒内的检测数量: {}".format(len(recent_detections)))
            
            if not recent_detections:
                rospy.logwarn("没有最近的检测")
                response.message = "NO_OBJECT_DETECTED"
                return response
            
            # 统计物体频率
            object_stats = {}
            for det in recent_detections:
                obj = det['object']
                cat = det['category']
                if obj not in object_stats:
                    object_stats[obj] = {'count': 0, 'category': cat, 'max_confidence': 0}
                object_stats[obj]['count'] += 1
                object_stats[obj]['max_confidence'] = max(object_stats[obj]['max_confidence'], det['confidence'])
            
            rospy.loginfo("物体频率统计: {}".format(
                {obj: f"{stats['count']}次(置信度{stats['max_confidence']:.2f})" 
                for obj, stats in object_stats.items()}))
            
            # 选择策略：优先次数多，其次置信度高
            if object_stats:
                best_obj = max(object_stats.items(), 
                            key=lambda x: (x[1]['count'], x[1]['max_confidence']))
                obj_name, stats = best_obj
                category = stats['category']
                count = stats['count']
                max_conf = stats['max_confidence']
                
                rospy.loginfo("最佳物体: {} (类别: {}, 出现次数: {}, 最高置信度: {:.3f})".format(
                    obj_name, category, count, max_conf))
                
                # 关键修改：降低数量要求，考虑置信度
                if count >= 1 and max_conf >= 0.6:  # 只要有1次高置信度检测即可
                    if self.current_task and category != self.current_task:
                        response.message = "WARN:" + obj_name
                        rospy.logwarn("任务不匹配: 需要 {}, 检测到 {}".format(self.current_task, category))
                    else:
                        response.message = obj_name
                        rospy.loginfo("确认物体: {}".format(obj_name))
                        self.session_active = False
                elif count >= 2:  # 或者2次较低置信度检测
                    if self.current_task and category != self.current_task:
                        response.message = "WARN:" + obj_name
                    else:
                        response.message = obj_name
                        rospy.loginfo("确认物体: {} (基于多次检测)".format(obj_name))
                        self.session_active = False
                else:
                    response.message = "CONTINUE_DETECTING"
                    rospy.loginfo("继续检测: {} ({}/{}次, 置信度{:.3f})".format(
                        obj_name, count, 2, max_conf))
            else:
                response.message = "NO_OBJECT_DETECTED"
                
            rospy.loginfo("服务返回: {}".format(response.message))
            return response
            
        except Exception as e:
            rospy.logerr("服务处理异常: {}".format(e))
            response.success = False
            response.message = "SERVICE_ERROR"
            return response
            
        finally:
            self.service_called = False
            rospy.loginfo("=== 服务处理完成 ===")

    def handle_reset_service(self, req):
        """重置服务"""
        rospy.loginfo("=== 重置视觉状态 ===")
        
        self.session_active = False
        self.detection_history.clear()
        self.service_called = False
        self.frame_counter = 0
        
        # 发布空检测信号
        self.publish_object_detected("")
        
        response = TriggerResponse()
        response.success = True
        response.message = "视觉状态已重置"
        
        rospy.loginfo("重置完成")
        return response

    def run(self):
        """主循环"""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        detector = ObjectDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("物体识别节点已停止")
