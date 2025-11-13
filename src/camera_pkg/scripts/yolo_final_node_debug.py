#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
import time

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # 服务接口
        self.object_service = rospy.Service("/object_recognition", Trigger, self.handle_object_service)
        self.reset_service = rospy.Service("/reset_vision_state", Trigger, self.handle_reset_service)
        
        # 状态管理变量
        self.session_start_time = time.time()
        self.is_new_detection_session = True
        self.last_detection_time = 0
        self.consecutive_empty_detections = 0
        self.previous_frame = None
        self.has_previous_frame = False
        
        # 订阅当前任务类型
        self.current_task = ""
        self.task_sub = rospy.Subscriber("/current_task", String, self.task_callback)
        
        # 图像发布
        self.annotated_img_pub = rospy.Publisher("/detect/object_annotated", Image, queue_size=10)
        
        # 模型配置
        self.model_path = rospy.get_param("~model", "/home/hxx/catkin_ws/src/camera_pkg/models/best.pt")
        self.conf_thres = 0.4
        
        # 类别映射
        self.class_map = {
            0: ("水果", "苹果"),
            1: ("水果", "香蕉"),
            2: ("水果", "西瓜"),
            3: ("蔬菜", "辣椒"),
            4: ("蔬菜", "土豆"),
            5: ("蔬菜", "番茄"),
            6: ("饮料", "牛奶"),
            7: ("饮料", "可乐"),
            8: ("食品", "蛋糕"),
        }
        
        # 检测参数
        self.session_timeout = rospy.get_param("~session_timeout", 10.0)  # 会话超时时间
        self.init_delay = rospy.get_param("~init_delay", 2.0)  # 新会话初始化延迟
        self.scene_change_threshold = rospy.get_param("~scene_change_threshold", 30.0)  # 场景变化阈值
        
        # 初始化变量
        self.latest_frame = None
        self.latest_detection = None
        self.latest_detection_time = None
        self.latest_object_name = ""
        self.model = self.load_model()
        
        # 订阅摄像头图像
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback)
        rospy.loginfo("物体识别节点启动（带状态管理），服务：/object_recognition")

    def task_callback(self, msg):
        """接收当前任务类型"""
        self.current_task = msg.data
        rospy.loginfo(f"更新当前任务类型: {self.current_task}")

    def load_model(self):
        try:
            model = torch.hub.load(
                'ultralytics/yolov5',  
                'custom', 
                path=self.model_path,
                force_reload=False
            )
            model.conf = self.conf_thres
            model.eval()
            rospy.loginfo(f".pt 模型加载成功：{self.model_path}")
            return model
        except Exception as e:
            rospy.logerr(f"模型加载失败：{e}")
            rospy.signal_shutdown("模型加载失败，退出节点")
            return None

    def image_callback(self, msg):
        """接收摄像头图像并持续检测"""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.continuous_detection()
        except Exception as e:
            rospy.logerr(f"图像转换错误：{e}")

    def is_scene_changed(self, current_frame):
        """检测场景是否发生显著变化"""
        if not self.has_previous_frame:
            self.previous_frame = current_frame.copy()
            self.has_previous_frame = True
            return True  # 第一帧总是认为是新场景
            
        # 计算帧间差异
        gray_prev = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray_prev, gray_curr)
        change_score = np.mean(diff)
        
        # 更新前一帧
        self.previous_frame = current_frame.copy()
        
        rospy.logdebug(f"场景变化得分: {change_score:.2f}")
        return change_score > self.scene_change_threshold

    def continuous_detection(self):
        """持续检测物体（带状态管理）"""
        if self.latest_frame is None or self.model is None:
            return
        
        frame = self.latest_frame.copy()
        
        # 检查会话超时
        current_time = time.time()
        if current_time - self.session_start_time > self.session_timeout:
            rospy.loginfo("检测会话超时，自动重置状态")
            self.reset_detection_state()
        
        # 检测场景变化
        scene_changed = self.is_scene_changed(frame)
        
        # 用YOLO模型检测
        results = self.model(frame)
        detections = results.pandas().xyxy[0].values
        
        # 筛选最佳结果
        best_match = None
        
        if len(detections) > 0:
            # 选择置信度最高的物体
            best_det = detections[np.argmax(detections[:, 4])]  
            cls_id = int(best_det[5])
            if cls_id in self.class_map:
                category, obj_name = self.class_map[cls_id]
                best_match = (category, obj_name, best_det)
            
            if best_match:
                category, obj_name, detection = best_match
                
                # 只有在场景变化或新会话时才更新检测结果
                if scene_changed or self.is_new_detection_session:
                    self.latest_detection = detection
                    self.latest_detection_time = current_time
                    self.consecutive_empty_detections = 0
                    rospy.loginfo(f"检测到：{obj_name}({category}) - 场景变化: {scene_changed}")
                
                # 标注图像
                x1, y1, x2, y2 = map(int, detection[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{obj_name}({detection[4]:.2f})", 
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # 没有检测到物体
            self.consecutive_empty_detections += 1
            if self.consecutive_empty_detections > 10:  # 连续10次空检测
                self.latest_detection = None
                rospy.logdebug("连续空检测，清空最新检测结果")
        
        # 发布标注图像
        try:
            annotated_img = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.annotated_img_pub.publish(annotated_img)
        except Exception as e:
            rospy.logwarn(f"发布标注图像失败：{e}")

    def reset_detection_state(self):
        """重置检测状态，开始新的检测会话"""
        rospy.loginfo("重置视觉检测状态")
        self.latest_detection = None
        self.latest_detection_time = None
        self.is_new_detection_session = True
        self.session_start_time = time.time()
        self.consecutive_empty_detections = 0
        self.has_previous_frame = False  # 重置帧缓存
        self.previous_frame = None
        
        # 可选：清除模型缓存（如果支持）
        if hasattr(self.model, 'reset'):
            try:
                self.model.reset()
            except:
                pass

    def handle_reset_service(self, req):
        """处理重置服务请求"""
        rospy.loginfo("收到视觉状态重置请求")
        self.reset_detection_state()
        
        response = TriggerResponse()
        response.success = True
        response.message = "视觉状态重置完成"
        return response

    def handle_object_service(self, req):
        """处理物体识别服务请求 - 带状态管理"""
        rospy.loginfo(f"收到物体识别请求，当前任务: {self.current_task}")
        
        response = TriggerResponse()
        
        if self.latest_frame is None or self.model is None:
            response.success = False
            response.message = "图像或模型未就绪"
            return response
        
        current_time = time.time()
        
        # 检查会话超时
        if current_time - self.session_start_time > self.session_timeout:
            rospy.loginfo("检测到会话超时，自动重置")
            self.reset_detection_state()
        
        # 新会话初始化延迟
        if self.is_new_detection_session:
            if current_time - self.session_start_time < self.init_delay:
                response.success = False
                response.message = "CONTINUE_DETECTING"
                rospy.loginfo("新会话初始化中，继续检测...")
                return response
            else:
                self.is_new_detection_session = False
                rospy.loginfo("新会话初始化完成")
        
        # 返回检测结果
        if self.latest_detection is not None:
            category, obj_name = self.class_map[int(self.latest_detection[5])]
            
            # 检查任务类型匹配
            if self.current_task and category != self.current_task:
                rospy.logwarn(f"任务类型不匹配：期望{self.current_task}，识别到{category}")
                response.message = f"WARN:{obj_name}"
            else:
                response.message = obj_name
            
            response.success = True
            rospy.loginfo(f"服务返回结果: {obj_name}({category})")
        else:
            # 没有检测结果
            response.success = True
            response.message = "NO_OBJECT_DETECTED"
            rospy.loginfo("当前无检测结果")
        
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
