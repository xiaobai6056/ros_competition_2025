#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
import torch
import os
import sys
import time
import math
import threading
from collections import deque

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

from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # 添加线程锁
        self.lock = threading.RLock()  # 使用可重入锁，避免死锁
        
        # 服务接口
        self.object_service = rospy.Service("/object_recognition", Trigger, self.handle_object_service)
        self.reset_service = rospy.Service("/reset_vision_state", Trigger, self.handle_reset_service)
        
        # 物体检测信号发布
        self.object_detected_pub = rospy.Publisher("/object_detected", String, queue_size=1)
        
        # 新增：重置完成信号发布
        self.reset_complete_pub = rospy.Publisher("/vision_reset_complete", String, queue_size=1)
        
        # 配置参数
        self.config = {
            'min_confidence': 0.40,
            'camera_hfov': 1.3962634,
            'image_width': 1920,
            'image_height': 1080,
            'max_det': 20  # 新增：YOLO11的最大检测数
        }
        
        # 状态管理
        self.session_active = False
        self.current_task = ""
        self.detection_history = deque(maxlen=15)
        self.service_called = False
        self.frame_counter = 0
        self.last_publish_time = 0  # 发布频率控制
        
        # 新增：话题发布历史记录（带时间戳的智能清理）
        self.published_objects_topic = {}  # 改为字典，存储对象名和发布时间戳
        self.max_history_size = 10  # 最大历史记录数量
        
        # 新增：重置状态管理
        self.resetting = False
        self.skip_frames_count = 0
        
        # 订阅当前任务类型
        self.task_sub = rospy.Subscriber("/current_task", String, self.task_callback)
        
        # 图像发布
        self.annotated_img_pub = rospy.Publisher("/detect/object_annotated", Image, queue_size=5)
        
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
            6: ("蔬菜", "番茄"),
            7: ("蔬菜", "土豆"),
            8: ("蔬菜", "辣椒"),
        }
        
        # 初始化模型 - 使用YOLO11方式
        self.model = self.load_model()
        
        # 订阅摄像头图像
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback, queue_size=1)
        
        rospy.loginfo("物体识别节点启动完成 - YOLOv11智能评分模式 + 线程安全版本")

    def load_model(self):
        """加载模型 - 使用YOLO11官方API"""
        try:
            rospy.loginfo(f"加载YOLO11模型，权重路径: {self.model_path}")
            
            # 核心修改：使用YOLO类加载权重
            model = YOLO(self.model_path)
            
            # 设置模型参数
            model.conf = self.config["min_confidence"]  # 置信度阈值
            model.iou = 0.4  # NMS IoU阈值
            model.max_det = self.config["max_det"]  # 最大检测数
            
            rospy.loginfo("YOLO11模型加载成功（CPU模式）！")
            return model
            
        except Exception as e:
            rospy.logerr(f"模型加载失败: {e}")
            return None

    def get_class_specific_threshold(self, obj_name):
        """为易误识别类别设置更高阈值"""
        high_threshold_classes = {
            '西瓜': 0.45,   
            '蛋糕': 0.45,   
            '香蕉': 0.45,   
            '苹果': 0.45,
            '牛奶': 0.45,
            '可乐': 0.45,
            '土豆': 0.45,
            '番茄': 0.25,
            '辣椒': 0.45
        }
        return high_threshold_classes.get(obj_name, self.config['min_confidence'])

    def task_callback(self, msg):
        """任务回调 - 添加线程锁"""
        with self.lock:
            self.current_task = msg.data
            rospy.loginfo("任务类型更新: {}".format(self.current_task))

    def image_callback(self, msg):
            """第一步优化：只降低分辨率"""
            # 原有的跳帧和状态检查逻辑保持不变
            with self.lock:
                if self.resetting and self.skip_frames_count > 0:
                    self.skip_frames_count -= 1
                    rospy.logdebug("🚫 跳帧处理: 跳过当前帧，剩余%d帧", self.skip_frames_count)
                    return
                elif self.resetting and self.skip_frames_count == 0:
                    self.resetting = False
                    rospy.loginfo("✅ 跳帧完成，开始正常图像处理")
            
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                
                # ========== 关键修改：降低分辨率 ==========
                # 从1920x1080降到960x540 (1/4计算量，保持较好精度)
                small_image = cv2.resize(cv_image, (960, 540))
                # ========================================
                
                with self.lock:
                    self.frame_counter += 1
                    frame_id = self.frame_counter
                
                self.clean_publish_history()
                
                # 使用小图像进行检测，其他逻辑完全不变
                self.detection_pipeline(small_image, frame_id)
                
            except Exception as e:
                rospy.logwarn("图像回调异常: {}".format(e))

    def clean_publish_history(self):
        """智能清理话题发布历史记录 - 添加线程锁"""
        with self.lock:
            current_time = time.time()
            
            # 清理策略：按数量限制清理（最旧优先）
            if len(self.published_objects_topic) > self.max_history_size:
                # 找到最旧的记录并删除
                oldest_obj = min(self.published_objects_topic.items(), key=lambda x: x[1])[0]
                del self.published_objects_topic[oldest_obj]
                rospy.loginfo("📭 按数量限制清理最旧记录: {}".format(oldest_obj))

    def detection_pipeline(self, frame, frame_id):
        """检测流水线 - 适配YOLO11推理逻辑"""
        if self.model is None:
            return
            
        try:
            # 核心修改：YOLO11推理方式 - 模型推理不需要锁
            results = self.model(frame, verbose=False)
            detections = []
            
            # 解析YOLO11的检测结果
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # 提取检测框坐标、置信度、类别ID
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        # 按原格式组装detections（x1,y1,x2,y2,conf,cls_id）
                        detections.append([x1, y1, x2, y2, conf, cls_id])
            
            if len(detections) > 0:
                self.process_detections(detections, frame, frame_id)
            else:
                with self.lock:
                    service_called = self.service_called
                    history_len = len(self.detection_history)
                
                if not service_called and history_len % 20 == 0:
                    self.publish_object_detected("")
                    
        except Exception as e:
            rospy.logwarn("检测流水线异常: {}".format(e))

    def process_detections(self, detections, frame, frame_id):
        """处理检测结果 - 使用评分机制选择最佳检测"""
        current_time = time.time()
        
        # 收集当前帧的所有有效检测
        current_frame_detections = []
        
        for i, detection in enumerate(detections):
            confidence = detection[4]
            cls_id = int(detection[5])
            
            if cls_id not in self.class_map:
                continue
                
            category, obj_name = self.class_map[cls_id]
            confidence_threshold = self.get_class_specific_threshold(obj_name)
            
            if confidence >= confidence_threshold:
                detection_record = {
                    'object': obj_name,
                    'category': category,
                    'confidence': confidence,
                    'timestamp': current_time,
                    'frame_id': frame_id
                }
                
                # 添加检测记录到历史 - 需要锁
                with self.lock:
                    self.detection_history.append(detection_record)
                
                current_frame_detections.append(detection_record)
        
        # 如果当前帧有检测，使用评分机制选择最佳的一个发布
        with self.lock:
            service_called = self.service_called
        
        if current_frame_detections and not service_called:
            best_detection = self.select_best_detection(current_frame_detections)
            if best_detection:
                # 话题发布历史检查：检查是否已发布过 - 需要锁
                with self.lock:
                    if best_detection['object'] in self.published_objects_topic:
                        rospy.logdebug("🚫 话题跳过已发布物品: {} (置信度: {:.3f})".format(
                            best_detection['object'], best_detection['confidence']))
                        return
                
                rospy.loginfo("📢 话题发布: {} (置信度: {:.3f})".format(
                    best_detection['object'], best_detection['confidence']))
                self.publish_object_detected(best_detection['object'])

    def select_best_detection(self, current_detections):
        """选择最佳检测 - 置信度权重显著高于频率"""
        if not current_detections:
            return None
        
        # 获取最近10秒的历史数据用于频率计算 - 需要锁
        current_time = time.time()
        with self.lock:
            recent_history = [det for det in self.detection_history 
                             if current_time - det['timestamp'] < 10.0]
        
        # 计算每个物体的频率
        object_frequency = {}
        for det in recent_history:
            obj = det['object']
            object_frequency[obj] = object_frequency.get(obj, 0) + 1
        
        # 对当前帧的每个检测计算评分
        best_score = -1
        best_detection = None
        
        for detection in current_detections:
            obj_name = detection['object']
            confidence = detection['confidence']
            frequency = object_frequency.get(obj_name, 0)
            
            # 评分公式：置信度权重远高于频率
            # 置信度占80%权重，频率占20%权重
            confidence_weight = 0.8
            frequency_weight = 0.2
            
            # 归一化频率（0-1范围）
            max_freq = max(object_frequency.values()) if object_frequency else 1
            normalized_freq = frequency / max_freq if max_freq > 0 else 0
            
            # 计算综合评分
            score = (confidence * confidence_weight + 
                    normalized_freq * frequency_weight)
            
            rospy.logdebug("物体 {} 评分: 置信度{:.3f}(权重{:.1f}) + 频率{}/{}(权重{:.1f}) = {:.3f}".format(
                obj_name, confidence, confidence_weight, 
                frequency, max_freq, frequency_weight, score))
            
            # 选择评分最高的检测
            if score > best_score:
                best_score = score
                best_detection = detection
        
        # 只有评分达到阈值才发布
        if best_score >= 0.6:  # 可调整的发布阈值
            return best_detection
        else:
            rospy.logdebug("最佳检测评分 {:.3f} 低于阈值，不发布".format(best_score))
            return None

    def publish_object_detected(self, object_name):
        """发布物体检测信号 - 添加发布频率限制和历史记录"""
        current_time = time.time()
        
        # 发布频率限制：至少间隔1秒 - 需要锁
        with self.lock:
            if hasattr(self, 'last_publish_time'):
                time_since_last_publish = current_time - self.last_publish_time
                if time_since_last_publish < 1.0:  # 1秒内不重复发布
                    rospy.logdebug("发布频率限制: 上次发布 {:.1f}秒前".format(time_since_last_publish))
                    return
        
        try:
            msg = String()
            msg.data = object_name
            self.object_detected_pub.publish(msg)
            
            # 更新发布时间和历史记录 - 需要锁
            with self.lock:
                self.last_publish_time = current_time
                
                # 记录话题发布历史（带时间戳）
                if object_name:
                    self.published_objects_topic[object_name] = current_time
                    rospy.loginfo("📝 记录话题发布物品: {} (历史数量: {})".format(
                        object_name, len(self.published_objects_topic)))
            
            rospy.loginfo("✅ 发布物体检测信号: {}".format(object_name))
            
        except Exception as e:
            rospy.logwarn("物体检测信号发布异常: {}".format(e))

    def handle_object_service(self, req):
        """服务处理 - 严格符合状态机要求，无历史检查"""
        rospy.loginfo("=== 收到识别请求 ===")
        
        response = TriggerResponse()
        response.success = True
        
        # 检查服务是否正在处理 - 需要锁
        with self.lock:
            if self.service_called:
                rospy.logwarn("服务正在处理中，拒绝重复调用")
                response.message = "SERVICE_BUSY"
                return response
            
            self.service_called = True
            self.session_active = True
            current_task = self.current_task
        
        try:
            current_time = time.time()
            
            # 获取检测历史数据 - 需要锁
            with self.lock:
                recent_detections = list(self.detection_history)
                published_objects = list(self.published_objects_topic.keys())
            
            rospy.loginfo("当前任务: {}".format(current_task))
            rospy.loginfo("检测历史长度: {}".format(len(recent_detections)))
            rospy.loginfo("话题已发布物品: {}".format(published_objects))
            rospy.loginfo("使用所有 {} 个检测数据".format(len(recent_detections)))
            
            if not recent_detections:
                rospy.logwarn("没有检测数据可用")
                response.message = "NO_OBJECT_DETECTED"
                return response
            
            # 统计物体频率 - 关键修改：服务调用无历史检查
            object_stats = {}
            for det in recent_detections:
                obj = det['object']
                cat = det['category']
                if obj not in object_stats:
                    object_stats[obj] = {'count': 0, 'category': cat, 'max_confidence': 0}
                object_stats[obj]['count'] += 1
                object_stats[obj]['max_confidence'] = max(object_stats[obj]['max_confidence'], det['confidence'])
            
            rospy.loginfo("服务物体频率统计: {}".format(
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
                
                rospy.loginfo("服务最佳物体: {} (类别: {}, 出现次数: {}, 最高置信度: {:.3f})".format(
                    obj_name, category, count, max_conf))
                
                # 关键修改：严格符合状态机要求
                if count >= 1 and max_conf >= 0.8:  # 只要有1次高置信度检测即可
                    if current_task and category != current_task:
                        response.message = "WARN:" + obj_name
                        rospy.logwarn("任务不匹配: 需要 {}, 检测到 {}".format(current_task, category))
                    else:
                        response.message = obj_name
                        rospy.loginfo("服务确认物体: {}".format(obj_name))
                        with self.lock:
                            self.session_active = False
                elif count >= 2:  # 或者2次较低置信度检测
                    if current_task and category != current_task:
                        response.message = "WARN:" + obj_name
                    else:
                        response.message = obj_name
                        rospy.loginfo("服务确认物体: {} (基于多次检测)".format(obj_name))
                        with self.lock:
                            self.session_active = False
                else:
                    response.message = "CONTINUE_DETECTING"
                    rospy.loginfo("服务继续检测: {} ({}/{}次, 置信度{:.3f})".format(
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
            # 清理服务调用标志 - 需要锁
            with self.lock:
                self.service_called = False
            rospy.loginfo("=== 服务处理完成 ===")

    def handle_reset_service(self, req):
        """重置服务 - 清空所有历史数据并添加跳帧处理"""
        rospy.loginfo("=== 收到重置请求，开始清空视觉历史数据 ===")
        
        # 使用锁确保重置操作的原子性
        with self.lock:
            try:
                # 第一步：设置重置标志，开始跳帧处理
                self.resetting = True
                self.skip_frames_count = 8  # 跳过8帧图像处理
                
                # 第二步：清空所有状态
                self.session_active = False
                self.detection_history.clear()  # 清空检测历史
                self.published_objects_topic.clear()  # 清空话题发布历史
                self.service_called = False
                self.frame_counter = 0
                
                response = TriggerResponse()
                response.success = True
                response.message = f"视觉历史数据已清空，将跳过{self.skip_frames_count}帧图像"
                
                rospy.loginfo("✅ 视觉状态重置完成，将跳过%d帧图像处理", self.skip_frames_count)
                
            except Exception as e:
                rospy.logerr("重置服务处理异常: {}".format(e))
                response = TriggerResponse()
                response.success = False
                response.message = "重置失败: {}".format(str(e))
                return response
        
        # 在锁外执行发布操作，避免阻塞
        try:
            # 第三步：发布空检测信号
            self.publish_object_detected("")
            
            # 第四步：发布重置完成信号
            self.reset_complete_pub.publish("reset_complete")
            
            return response
            
        except Exception as e:
            rospy.logerr("重置发布操作异常: {}".format(e))
            response.success = False
            response.message = "重置发布失败: {}".format(str(e))
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
