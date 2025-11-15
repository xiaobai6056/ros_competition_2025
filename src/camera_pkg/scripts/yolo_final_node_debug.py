#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
import time
import math
from collections import deque
import tf2_ros

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # 服务接口
        self.object_service = rospy.Service("/object_recognition", Trigger, self.handle_object_service)
        self.reset_service = rospy.Service("/reset_vision_state", Trigger, self.handle_reset_service)
        self.reset_pre_detection_service = rospy.Service("/reset_pre_detection", Trigger, self.handle_reset_pre_detection)
        
        # 预识别坐标话题
        self.target_pub = rospy.Publisher("/pre_detection_target", PointStamped, queue_size=1)
        
        # TF2支持
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 精确配置
        self.config = {
            'min_confidence': 0.30,
            
            # 相机参数
            'camera_hfov': 1.3962634,    # 80度水平视野
            'camera_vfov': 0.785,        # 45度垂直视野
            'camera_position': (0.125, 0, 0.175),
            'image_width': 1920,
            'image_height': 1080,
            
            # 参考尺寸
            'reference_object_height': 0.3,
            'reference_object_width': 0.7,
            'distance_weights': [0.9, 0.1, 0.0],
        }
        
        # 预识别配置
        self.pre_detection_config = {
            'time_window': 3.0,
            'min_score_threshold': 0.55,
            'target_freshness': 2.0
        }
        
        # 边界框配置（基于你提供的四个点）
        self.boundary_points = [
            (-0.3113971948623657, 2.8663101196289062),    # 左下
            (3.790703773498535, 2.8181376457214355),      # 右下  
            (3.7462081909179688, 7.67902946472168),       # 右上
            (-0.33761417865753174, 7.638247489929199)     # 左上
        ]
        
        # 状态管理
        self.session_active = False
        self.current_task = ""
        self.detection_history = deque(maxlen=30)
        self.service_called = False
        self.direction_published = False
        self.current_best_target = None
        self.last_pre_detection_time = 0
        
        # 订阅当前任务类型
        self.task_sub = rospy.Subscriber("/current_task", String, self.task_callback)
        
        # 图像发布
        self.annotated_img_pub = rospy.Publisher("/detect/object_annotated", Image, queue_size=5)
        
        # 模型配置
        self.model_path = rospy.get_param("~model", "/home/hxx/catkin_ws/src/camera_pkg/models/best.pt")
        
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
        
        # 初始化模型
        self.model = self.load_model()
        
        # 订阅摄像头图像
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback, queue_size=1)
        
        rospy.loginfo("物体识别节点启动完成")
        rospy.loginfo("边界区域: [{:.2f}, {:.2f}] -> [{:.2f}, {:.2f}]".format(
            self.boundary_points[0][0], self.boundary_points[0][1],
            self.boundary_points[2][0], self.boundary_points[2][1]
        ))

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

    def is_point_in_boundary(self, x, y):
        """检查坐标是否在边界矩形内"""
        # 提取边界矩形的四个角点
        left = min(point[0] for point in self.boundary_points)
        right = max(point[0] for point in self.boundary_points)
        bottom = min(point[1] for point in self.boundary_points)
        top = max(point[1] for point in self.boundary_points)
        
        # 检查点是否在矩形内
        in_boundary = left <= x <= right and bottom <= y <= top
        
        if not in_boundary:
            rospy.logwarn("坐标超出边界: ({:.2f}, {:.2f}) 边界: [{:.2f}-{:.2f}]x[{:.2f}-{:.2f}]".format(
                x, y, left, right, bottom, top))
        
        return in_boundary

    def calculate_focal_lengths(self):
        """计算焦距"""
        focal_length_h = self.config['image_width'] / (2 * math.tan(self.config['camera_hfov'] / 2))
        focal_length_v = self.config['image_height'] / (2 * math.tan(self.config['camera_vfov'] / 2))
        return focal_length_h, focal_length_v

    def estimate_distance(self, detection, frame_shape):
        """距离估算"""
        try:
            x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
            
            focal_length_h, focal_length_v = self.calculate_focal_lengths()
            
            bbox_height = y2 - y1
            bbox_width = x2 - x1
            
            # 高度法
            if bbox_height > 0:
                distance_height = (focal_length_v * self.config['reference_object_height']) / bbox_height
            else:
                distance_height = 2.0
            
            # 宽度法
            if bbox_width > 0:
                distance_width = (focal_length_h * self.config['reference_object_width']) / bbox_width
            else:
                distance_width = 2.0
            
            # 加权平均
            weights = self.config['distance_weights']
            distances = [distance_height, distance_width, 10.0]
            weighted_avg = sum(d * w for d, w in zip(distances, weights))
            
            return max(0.5, min(8.0, weighted_avg))
            
        except Exception as e:
            rospy.logwarn("距离估算失败: {}".format(e))
            return 2.0

    def calculate_horizontal_angle(self, detection, frame_shape):
        """角度计算"""
        try:
            x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
            img_width = frame_shape[1]
            
            bbox_center_x = (x1 + x2) / 2
            img_center_x = img_width / 2
            
            pixel_offset = bbox_center_x - img_center_x
            pixels_per_radian = img_width / self.config['camera_hfov']
            horizontal_angle = pixel_offset / pixels_per_radian
            
            max_angle = self.config['camera_hfov'] / 2
            return max(-max_angle, min(max_angle, horizontal_angle))
            
        except Exception as e:
            rospy.logwarn("角度计算失败: {}".format(e))
            return 0.0

    def transform_to_world_coordinates(self, detection, frame_shape, obj_name):
        """世界坐标转换"""
        try:
            robot_x, robot_y, robot_yaw = self.get_robot_pose()
            distance = self.estimate_distance(detection, frame_shape)
            horizontal_angle = self.calculate_horizontal_angle(detection, frame_shape)
            
            camera_x, camera_y, _ = self.config['camera_position']
            target_x_robot = camera_x + distance * math.cos(horizontal_angle)
            target_y_robot = camera_y + distance * math.sin(horizontal_angle)
            
            cos_yaw = math.cos(robot_yaw)
            sin_yaw = math.sin(robot_yaw)
            target_x_world = robot_x + target_x_robot * cos_yaw - target_y_robot * sin_yaw
            target_y_world = robot_y + target_x_robot * sin_yaw + target_y_robot * cos_yaw
            
            rospy.loginfo("坐标计算: {} -> ({:.2f}, {:.2f})m".format(obj_name, target_x_world, target_y_world))
            
            return target_x_world, target_y_world
            
        except Exception as e:
            rospy.logwarn("坐标转换失败: {}".format(e))
            robot_x, robot_y, robot_yaw = self.get_robot_pose()
            return robot_x + 2.0 * math.cos(robot_yaw), robot_y + 2.0 * math.sin(robot_yaw)

    def get_robot_pose(self):
        """获取机器人位姿"""
        try:
            transform = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(0.1))
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            q = transform.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return x, y, yaw
            
        except Exception as e:
            rospy.logwarn("TF获取失败: {}".format(e))
            return 0.0, 0.0, 0.0

    def task_callback(self, msg):
        """任务回调"""
        self.current_task = msg.data
        rospy.loginfo("任务类型更新: {}".format(self.current_task))
        self.direction_published = False
        self.current_best_target = None

    def image_callback(self, msg):
        """图像回调"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.detection_pipeline(cv_image)
            
        except Exception as e:
            rospy.logwarn("图像回调异常: {}".format(e))

    def detection_pipeline(self, frame):
        """检测流水线"""
        if self.model is None:
            rospy.logwarn("模型未加载，跳过检测")
            return
            
        try:
            with torch.no_grad():
                results = self.model(frame)
                detections = results.pandas().xyxy[0].values
                
                if len(detections) > 0:
                    self.process_detections(detections, frame)
                    if not self.service_called:
                        self.smart_pre_detection_publish(frame)
                else:
                    if not self.session_active and len(self.detection_history) % 30 == 0:
                        self.publish_target_position(0.0, 0.0, 0.0)
                    
        except Exception as e:
            rospy.logwarn("检测流水线异常: {}".format(e))

    def process_detections(self, detections, frame):
        """处理检测结果"""
        current_time = time.time()
        rospy.loginfo("处理 {} 个检测".format(len(detections)))
        
        valid_detection_count = 0
        
        for i, detection in enumerate(detections):
            confidence = detection[4]
            cls_id = int(detection[5])
            
            if cls_id not in self.class_map:
                continue
                
            category, obj_name = self.class_map[cls_id]
            
            if confidence >= self.config['min_confidence']:
                detection_record = {
                    'object': obj_name,
                    'category': category,
                    'confidence': confidence,
                    'detection': detection,
                    'timestamp': current_time,
                }
                
                self.detection_history.append(detection_record)
                valid_detection_count += 1
                rospy.loginfo("✅ 有效检测: {} 置信度: {:.3f}".format(obj_name, confidence))
        
        rospy.loginfo("本次处理完成: {} 个有效检测，检测历史长度: {}".format(
            valid_detection_count, len(self.detection_history)))
        
        # 更新最佳预识别目标
        if not self.service_called:
            self.update_best_pre_detection_target()

    def update_best_pre_detection_target(self):
        """更新最佳预识别目标"""
        if not self.detection_history:
            return
            
        if self.service_called:
            return
        
        current_time = time.time()
        recent_detections = [d for d in self.detection_history 
                           if current_time - d['timestamp'] < self.pre_detection_config['time_window']]
        
        if not recent_detections:
            return
        
        # 筛选任务相关检测
        task_related_detections = []
        if self.current_task:
            task_related_detections = [d for d in recent_detections 
                                     if d['category'] == self.current_task]
            if not task_related_detections:
                return
        else:
            task_related_detections = recent_detections
        
        # 统计物体出现情况
        object_stats = {}
        for det in task_related_detections:
            obj_name = det['object']
            if obj_name not in object_stats:
                object_stats[obj_name] = {'count': 0, 'total_confidence': 0, 'last_detection': det}
            object_stats[obj_name]['count'] += 1
            object_stats[obj_name]['total_confidence'] += det['confidence']
        
        # 计算评分
        best_object = None
        best_score = -1
        
        for obj_name, stats in object_stats.items():
            avg_confidence = stats['total_confidence'] / stats['count']
            frequency_score = min(stats['count'] / 3.0, 1.0) * 0.3
            confidence_score = avg_confidence * 0.7
            total_score = frequency_score + confidence_score
            
            if total_score > best_score:
                best_score = total_score
                best_object = obj_name
                best_stats = stats
        
        # 更新最佳目标
        if best_object and best_score > self.pre_detection_config['min_score_threshold']:
            self.current_best_target = {
                'object': best_object,
                'detection': best_stats['last_detection']['detection'],
                'score': best_score,
                'update_time': current_time,
            }
            rospy.loginfo("🎯 更新最佳目标: {} (得分: {:.3f})".format(best_object, best_score))
        else:
            self.current_best_target = None

    def smart_pre_detection_publish(self, frame):
        """智能预识别发布"""
        if self.current_best_target is None:
            return
            
        if self.direction_published:
            return
            
        if self.service_called:
            return
            
        current_time = time.time()
        if current_time - self.last_pre_detection_time < 0.5:
            return
        
        if current_time - self.current_best_target['update_time'] > self.pre_detection_config['target_freshness']:
            return
        
        detection = self.current_best_target['detection']
        obj_name = self.current_best_target['object']
        score = self.current_best_target['score']
        
        rospy.loginfo("🎯 准备发布预识别: {} (得分: {:.3f})".format(obj_name, score))
        
        target_x, target_y = self.transform_to_world_coordinates(detection, frame.shape, obj_name)
        
        # 检查坐标是否在边界内
        if not self.is_point_in_boundary(target_x, target_y):
            rospy.logwarn("❌ 坐标超出边界，取消发布: {} -> ({:.2f}, {:.2f})".format(obj_name, target_x, target_y))
            return
        
        rospy.loginfo("📍 发布坐标: {} -> ({:.2f}, {:.2f})".format(obj_name, target_x, target_y))
        self.publish_target_position(target_x, target_y, 0.0)
        
        self.direction_published = True
        self.last_pre_detection_time = current_time
        
        rospy.loginfo("✅ 预识别发布完成")

    def publish_target_position(self, x, y, z):
        """发布目标位置"""
        try:
            point_msg = PointStamped()
            point_msg.header.stamp = rospy.Time.now()
            point_msg.header.frame_id = "map"
            point_msg.point.x = x
            point_msg.point.y = y
            point_msg.point.z = z
            
            self.target_pub.publish(point_msg)
            rospy.loginfo("📤 坐标发布成功: ({:.2f}, {:.2f})".format(x, y))
            
        except Exception as e:
            rospy.logwarn("坐标发布异常: {}".format(e))

    def handle_object_service(self, req):
        """服务处理"""
        rospy.loginfo("=== 收到识别请求 ===")
        rospy.loginfo("当前任务: {}".format(self.current_task))
        rospy.loginfo("检测历史长度: {}".format(len(self.detection_history)))
        
        response = TriggerResponse()
        response.success = True
        
        # 防止重复调用
        if self.service_called:
            rospy.logwarn("服务正在处理中，拒绝重复调用")
            response.message = "SERVICE_BUSY"
            return response
        
        self.session_active = True
        self.service_called = True
        
        try:
            current_time = time.time()
            
            # 查找最近检测结果
            recent_detections = []
            for det in reversed(self.detection_history):
                time_diff = current_time - det['timestamp']
                if time_diff < 5.0:
                    recent_detections.append(det)
                    rospy.loginfo("有效检测: {} ({}秒前)".format(det['object'], time_diff))
                if len(recent_detections) >= 15:
                    break
            
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
                    object_stats[obj] = {'count': 0, 'category': cat}
                object_stats[obj]['count'] += 1
            
            rospy.loginfo("物体频率统计: {}".format(
                {obj: stats['count'] for obj, stats in object_stats.items()}))
            
            if not object_stats:
                rospy.logwarn("物体统计为空")
                response.message = "NO_OBJECT_DETECTED"
                return response
            
            best_obj = max(object_stats.items(), key=lambda x: x[1]['count'])
            obj_name, stats = best_obj
            category = stats['category']
            count = stats['count']
            
            rospy.loginfo("最佳物体: {} (类别: {}, 出现次数: {})".format(obj_name, category, count))
            
            # 检查任务匹配
            if self.current_task and category != self.current_task:
                rospy.logwarn("任务不匹配: 需要 {}, 检测到 {}".format(self.current_task, category))
                response.message = "WARN:" + obj_name
            else:
                required_count = min(3, len(recent_detections) // 2 + 1)
                rospy.loginfo("要求次数: {} (当前: {})".format(required_count, count))
                
                if count >= required_count:
                    response.message = obj_name
                    self.session_active = False
                    rospy.loginfo("✅ 确认物体: {}".format(obj_name))
                else:
                    response.message = "CONTINUE_DETECTING"
                    rospy.loginfo("🔄 继续检测: {} ({}/{})".format(obj_name, count, required_count))
            
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
    

    def handle_reset_pre_detection(self, req):
        """由状态机调用，在开始新扫描时重置预检测数据"""
        rospy.loginfo("=== 重置预检测数据 ===")
        
        # 清空检测历史
        self.detection_history.clear()
        self.current_best_target = None
        self.last_pre_detection_time = 0
        
        # 重要：发布零点坐标覆盖之前的错误坐标
        self.publish_target_position(0.0, 0.0, 0.0)
        rospy.loginfo("发布零点坐标覆盖之前的预识别结果")
        
        # 重置发布状态，允许重新发布
        self.direction_published = False
        
        response = TriggerResponse()
        response.success = True
        response.message = "预检测数据已重置，已发布零点坐标"
        return response

    def handle_reset_service(self, req):
        """重置服务"""
        rospy.loginfo("=== 重置视觉状态 ===")
        
        self.session_active = False
        self.detection_history.clear()
        self.service_called = False
        self.direction_published = False
        self.current_best_target = None
        
        self.publish_target_position(0.0, 0.0, 0.0)
        
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
