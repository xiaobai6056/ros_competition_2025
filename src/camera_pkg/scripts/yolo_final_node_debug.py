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
        
        # 精确配置 - 复用测试节点的相机参数
        self.config = {
            'min_confidence': 0.40,
            
            # 相机参数 - 复用测试节点的精确参数
            'camera_hfov': 1.3962634,    # 80度水平视野
            'image_width': 1920,
            'image_height': 1080,
            'camera_position': (0.125, 0, 0.175),
            
            # 参考尺寸 - 使用单一高度参考
            'reference_object_height': 0.3,
            
            # 移除复杂的距离权重，使用单一高度法
        }
        
        # 初始化相机参数 - 复用测试节点的精确计算
        self.initialize_camera_params()
        
        # 预识别配置 - 方案三：实时性优先策略
        self.pre_detection_config = {
            'time_window': 1.0,           # 从3.0秒降至1.0秒，提高实时性
            'min_score_threshold': 0.60,
            'target_freshness': 0.5,      # 从2.0秒降至0.5秒，确保数据新鲜
            'immediate_confidence': 0.75, # 单帧高置信度阈值
            'max_detection_age': 0.3      # 检测最大年龄限制
        }
        
        # 边界框配置（基于你提供的四个点）
        self.boundary_points = [
           (-0.81, 2.57),    # 左下
           (4.21, 2.52),     # 右下
           (4.16, 7.98),     # 右上
           (-0.84, 7.94)     # 左上
        ]
        
        # 状态管理
        self.session_active = False
        self.current_task = ""
        self.detection_history = deque(maxlen=20)  # 减少历史长度
        self.service_called = False
        self.direction_published = False
        self.current_best_target = None
        self.last_pre_detection_time = 0
        self.last_frame_timestamp = None
        self.frame_counter = 0  # 帧计数器
        
        # 订阅当前任务类型
        self.task_sub = rospy.Subscriber("/current_task", String, self.task_callback)
        
        # 图像发布
        self.annotated_img_pub = rospy.Publisher("/detect/object_annotated", Image, queue_size=5)
        
        # 模型配置
        self.model_path = rospy.get_param("~model", "/home/hxx/catkin_ws/src/camera_pkg/models/best.pt")
        
        # 类别映射
        self.class_map = {
            0: ("水果", "香蕉"),        # banana
            1: ("水果", "西瓜"),        # watermelon  
            2: ("水果", "苹果"),        # apple
            3: ("食品", "蛋糕"),        # cake
            4: ("食品", "牛奶"),        # milk
            5: ("食品", "可乐"),        # coke
            6: ("蔬菜", "土豆"),        # potato
            7: ("蔬菜", "番茄"),        # tomato
            8: ("蔬菜", "辣椒"),        # chilli
        }
        # 初始化模型
        self.model = self.load_model()
        
        # 订阅摄像头图像
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback, queue_size=1)
        
        rospy.loginfo("物体识别节点启动完成 - 帧ID严格匹配模式")
        rospy.loginfo("边界区域: [{:.2f}, {:.2f}] -> [{:.2f}, {:.2f}]".format(
            self.boundary_points[0][0], self.boundary_points[0][1],
            self.boundary_points[2][0], self.boundary_points[2][1]
        ))
        rospy.loginfo("相机参数: HFOV={:.2f}°, VFOV={:.2f}°".format(
            math.degrees(self.config['camera_hfov']), math.degrees(self.config['camera_vfov'])))
        rospy.loginfo("焦距: f_h={:.1f}, f_v={:.1f}".format(self.focal_length_h, self.focal_length_v))

    def initialize_camera_params(self):
        """初始化相机参数 - 复用测试节点的精确计算"""
        # 计算垂直视场角
        aspect_ratio = self.config['image_height'] / self.config['image_width']
        hfov_rad = self.config['camera_hfov']
        self.config['camera_vfov'] = 2 * math.atan(math.tan(hfov_rad/2) * aspect_ratio)
        
        # 计算焦距 - 复用测试节点的精确公式
        self.focal_length_h = self.config['image_width'] / (2 * math.tan(self.config['camera_hfov'] / 2))
        self.focal_length_v = self.config['image_height'] / (2 * math.tan(self.config['camera_vfov'] / 2))
        
        rospy.loginfo("相机参数初始化完成: VFOV={:.2f}°, f_h={:.1f}, f_v={:.1f}".format(
            math.degrees(self.config['camera_vfov']), self.focal_length_h, self.focal_length_v))

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

    def calculate_visual_distance(self, bbox_height):
        """视觉距离估算 - 复用测试节点的精确方法"""
        if bbox_height <= 10:
            rospy.logwarn("检测框高度过小({:.1f}px)，使用默认距离2.0m".format(bbox_height))
            return 2.0
        
        # 复用测试节点的精确公式：距离 = (垂直焦距 × 参考物体高度) / 检测框高度
        distance = (self.focal_length_v * self.config['reference_object_height']) / bbox_height
        
        # 限制距离范围
        distance = max(0.5, min(8.0, distance))
        
        rospy.logdebug("距离估算: 框高={:.1f}px, 焦距_v={:.1f}, 参考高={:.2f}m -> 距离={:.2f}m".format(
            bbox_height, self.focal_length_v, self.config['reference_object_height'], distance))
        
        return distance

    def calculate_visual_angle(self, bbox_center_x):
        """视觉角度估算 - 复用测试节点的精确方法"""
        # 复用测试节点的精确公式：角度 = 像素偏移量 / 水平焦距
        pixel_offset = bbox_center_x - self.config['image_width'] / 2
        angle = pixel_offset / self.focal_length_h
        
        # 限制角度范围
        max_angle = self.config['camera_hfov'] / 2
        angle = max(-max_angle, min(max_angle, angle))
        
        rospy.logdebug("角度估算: 中心_x={:.1f}px, 像素偏移={:.1f}, 焦距_h={:.1f} -> 角度={:.3f}rad".format(
            bbox_center_x, pixel_offset, self.focal_length_h, angle))
        
        return angle

    def visual_to_robot_coords(self, distance, angle):
        """视觉坐标转换到机器人坐标系 - 复用测试节点的精确方法"""
        camera_x, camera_y, _ = self.config['camera_position']
        target_x = camera_x + distance * math.cos(angle)
        target_y = camera_y + distance * math.sin(angle)
        
        rospy.logdebug("机器人坐标: 相机位置=({:.3f}, {:.3f}), 距离={:.2f}m, 角度={:.3f}rad -> 目标=({:.2f}, {:.2f})".format(
            camera_x, camera_y, distance, angle, target_x, target_y))
        
        return target_x, target_y

    def transform_to_world_coordinates(self, detection, frame_shape, obj_name, stamp=None, frame_id=None):
        """世界坐标转换 - 使用测试节点的精确计算逻辑"""
        try:
            # 提取检测框信息
            x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
            bbox_height = y2 - y1
            bbox_center_x = (x1 + x2) / 2
            
            rospy.loginfo("🔍 坐标计算开始[帧{}]: {} 框高={:.1f}px, 中心_x={:.1f}px".format(
                frame_id, obj_name, bbox_height, bbox_center_x))
            
            # 1. 使用测试节点的距离估算方法
            distance = self.calculate_visual_distance(bbox_height)
            
            # 2. 使用测试节点的角度估算方法  
            horizontal_angle = self.calculate_visual_angle(bbox_center_x)
            
            # 3. 转换到机器人坐标系
            target_x_robot, target_y_robot = self.visual_to_robot_coords(distance, horizontal_angle)
            
            # 4. 转换到世界坐标系
            target_x_world, target_y_world = self.robot_to_world_coords(target_x_robot, target_y_robot, stamp)
            
            rospy.loginfo("🎯 坐标计算结果[帧{}]: {} -> 机器人坐标=({:.2f}, {:.2f}), 世界坐标=({:.2f}, {:.2f})m".format(
                frame_id, obj_name, target_x_robot, target_y_robot, target_x_world, target_y_world))
            
            return target_x_world, target_y_world
            
        except Exception as e:
            rospy.logwarn("坐标转换失败: {}".format(e))
            # 出错时返回机器人前方2米的位置
            robot_x, robot_y, robot_yaw = self.get_robot_pose(stamp)
            fallback_x = robot_x + 2.0 * math.cos(robot_yaw)
            fallback_y = robot_y + 2.0 * math.sin(robot_yaw)
            rospy.logwarn("使用备用坐标: ({:.2f}, {:.2f})".format(fallback_x, fallback_y))
            return fallback_x, fallback_y

    def robot_to_world_coords(self, robot_x, robot_y, stamp=None):
        """机器人坐标系转世界坐标系"""
        try:
            if stamp is None:
                stamp = rospy.Time.now()
                
            transform = self.tf_buffer.lookup_transform("map", "base_link", stamp, rospy.Duration(0.1))
            world_x = transform.transform.translation.x
            world_y = transform.transform.translation.y
            
            q = transform.transform.rotation
            robot_yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))
            
            # 坐标转换 - 复用测试节点的精确方法
            cos_yaw = math.cos(robot_yaw)
            sin_yaw = math.sin(robot_yaw)
            target_x = world_x + robot_x * cos_yaw - robot_y * sin_yaw
            target_y = world_y + robot_x * sin_yaw + robot_y * cos_yaw
            
            rospy.logdebug("世界坐标转换: 机器人位置=({:.2f}, {:.2f}), 偏航角={:.3f}rad -> 世界坐标=({:.2f}, {:.2f})".format(
                world_x, world_y, robot_yaw, target_x, target_y))
            
            return target_x, target_y
            
        except Exception as e:
            rospy.logwarn("坐标转换失败: {}".format(e))
            return 0.0, 0.0

    def get_robot_pose(self, stamp=None):
        """获取机器人位姿"""
        try:
            if stamp is None:
                stamp = rospy.Time.now()
                
            transform = self.tf_buffer.lookup_transform("map", "base_link", stamp, rospy.Duration(0.1))
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
            self.last_frame_timestamp = msg.header.stamp
            self.frame_counter += 1  # 增加帧计数器
            current_frame_id = self.frame_counter
            
            rospy.loginfo("📷 收到图像帧: ID={}, 时间戳={}.{}".format(
                current_frame_id, msg.header.stamp.secs, msg.header.stamp.nsecs))
            
            self.detection_pipeline(cv_image, msg.header.stamp, current_frame_id)
            
        except Exception as e:
            rospy.logwarn("图像回调异常: {}".format(e))

    def detection_pipeline(self, frame, stamp, frame_id):
        """检测流水线"""
        if self.model is None:
            rospy.logwarn("模型未加载，跳过检测")
            return
            
        try:
            with torch.no_grad():
                results = self.model(frame)
                detections = results.pandas().xyxy[0].values
                
                if len(detections) > 0:
                    self.process_detections(detections, frame, stamp, frame_id)
                    if not self.service_called:
                        self.smart_pre_detection_publish(frame, stamp, frame_id)
                else:
                    if not self.session_active and len(self.detection_history) % 30 == 0:
                        self.publish_target_position(0.0, 0.0, 0.0)
                    
        except Exception as e:
            rospy.logwarn("检测流水线异常: {}".format(e))
    
    def is_valid_detection(self, detection, obj_name):
        """检查检测框尺寸合理性 - 基于实际数据分析"""
        x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        
        rospy.logdebug("📏 检测框尺寸检查: {} -> {}x{}px".format(obj_name, bbox_width, bbox_height))
        
        # 基于实际数据的精确阈值
        min_size = 80   # 最小尺寸限制
        max_size = 450  # 最大尺寸限制
        
        # 1. 最小尺寸限制
        if bbox_height < min_size or bbox_width < min_size:
            rospy.logwarn("🚫 检测框过小被过滤: {} {}x{}px < {}px".format(
                obj_name, bbox_width, bbox_height, min_size))
            return False
        
        # 2. 最大尺寸限制  
        if bbox_height > max_size or bbox_width > max_size:
            rospy.logwarn("🚫 检测框过大被过滤: {} {}x{}px > {}px".format(
                obj_name, bbox_width, bbox_height, max_size))
            return False
        
        # 3. 特殊处理：西瓜需要更大尺寸才可信（基于误识别分析）
        if obj_name == '西瓜' and bbox_height < 100:
            rospy.logwarn("🚫 西瓜检测框过小被过滤: {}x{}px < 100px".format(bbox_width, bbox_height))
            return False
            
        # 4. 宽高比检查（可选，进一步过滤异常检测）
        aspect_ratio = bbox_width / bbox_height
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            rospy.logwarn("🚫 检测框宽高比异常: {} {:.2f}".format(obj_name, aspect_ratio))
            return False
        
        rospy.loginfo("✅ 检测框尺寸合法: {} {}x{}px".format(obj_name, bbox_width, bbox_height))
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

    def process_detections(self, detections, frame, stamp, frame_id):
        """处理检测结果 - 使用帧ID严格匹配 + 检测合法性检查"""
        current_time = time.time()
        rospy.loginfo("处理[帧{}] {} 个检测".format(frame_id, len(detections)))
        
        valid_detection_count = 0
        frame_detections = []  # 当前帧的所有检测
        
        for i, detection in enumerate(detections):
            confidence = detection[4]
            cls_id = int(detection[5])
            
            if cls_id not in self.class_map:
                continue
                
            category, obj_name = self.class_map[cls_id]
            
            # 使用类别特异性阈值
            confidence_threshold = self.get_class_specific_threshold(obj_name)
            
            if confidence >= confidence_threshold:
                # 1. 检测框尺寸合法性检查
                if not self.is_valid_detection(detection, obj_name):
                    rospy.logwarn("🚫 [帧{}]检测框不合法被过滤: {} (置信度: {:.3f})".format(
                        frame_id, obj_name, confidence))
                    continue
                    
                # 方案三：立即计算并存储坐标，确保名称与坐标匹配
                target_x, target_y = self.transform_to_world_coordinates(
                    detection, frame.shape, obj_name, stamp, frame_id)
                
                detection_record = {
                    'object': obj_name,
                    'category': category,
                    'confidence': confidence,
                    'detection': detection,
                    'timestamp': current_time,
                    'calculated_coords': (target_x, target_y),  # 存储计算好的坐标
                    'frame_timestamp': stamp,  # 存储图像时间戳
                    'frame_id': frame_id  # 关键：存储帧ID
                }
                
                self.detection_history.append(detection_record)
                frame_detections.append(detection_record)
                valid_detection_count += 1
                
                # 单帧高置信度立即发布 - 使用当前帧的检测
                if (confidence >= self.pre_detection_config['immediate_confidence'] and 
                    not self.service_called and 
                    not self.direction_published):
                    
                    # 检查任务匹配
                    if self.current_task and category != self.current_task:
                        rospy.logwarn("🚫 [帧{}]任务不匹配，取消单帧发布: 需要 {}, 检测到 {}".format(
                            frame_id, self.current_task, category))
                        continue
                    
                    # 检查坐标边界
                    if not self.is_point_in_boundary(target_x, target_y):
                        rospy.logwarn("❌ [帧{}]坐标超出边界，取消单帧发布: {} -> ({:.2f}, {:.2f})".format(
                            frame_id, obj_name, target_x, target_y))
                        continue
                    
                    rospy.loginfo("🚀 [帧{}]单帧高置信度立即发布: {} (置信度: {:.3f}, 任务匹配)".format(
                        frame_id, obj_name, confidence))
                    self.publish_target_position(target_x, target_y, 0.0)
                    self.direction_published = True
                    self.last_pre_detection_time = current_time
                    
                rospy.loginfo("✅ [帧{}]有效检测: {} 置信度: {:.3f} 坐标: ({:.2f}, {:.2f})".format(
                    frame_id, obj_name, confidence, target_x, target_y))
        
        rospy.loginfo("[帧{}]处理完成: {} 个有效检测，检测历史长度: {}".format(
            frame_id, valid_detection_count, len(self.detection_history)))
        
        # 更新最佳预识别目标 - 使用当前帧的数据
        if not self.service_called and frame_detections:
            self.update_best_pre_detection_target(frame_detections, frame_id)

    def update_best_pre_detection_target(self, current_frame_detections, frame_id):
        """更新最佳预识别目标 - 基于当前帧数据"""
        if not current_frame_detections:
            return
            
        if self.service_called:
            return
        
        current_time = time.time()
        
        # 只使用当前帧的检测数据进行统计
        detection_source = current_frame_detections
        
        rospy.loginfo("[帧{}]使用当前帧 {} 个检测进行统计".format(frame_id, len(detection_source)))
        
        # 筛选任务相关检测
        task_related_detections = []
        if self.current_task:
            task_related_detections = [d for d in detection_source 
                                    if d['category'] == self.current_task]
            if not task_related_detections:
                rospy.loginfo("⚠️ [帧{}]无任务相关检测: 需要 {}".format(frame_id, self.current_task))
                self.current_best_target = None
                return
            else:
                rospy.loginfo("✅ [帧{}]找到 {} 个任务相关检测: {}".format(
                    frame_id, len(task_related_detections), self.current_task))
        else:
            task_related_detections = detection_source
            rospy.loginfo("📋 [帧{}]无任务限制，使用所有检测数据".format(frame_id))
        
        if not task_related_detections:
            self.current_best_target = None
            return
        
        # 统计物体出现情况（在当前帧内）
        object_stats = {}
        for det in task_related_detections:
            obj_name = det['object']
            if obj_name not in object_stats:
                object_stats[obj_name] = {'count': 0, 'total_confidence': 0, 'detections': []}
            object_stats[obj_name]['count'] += 1
            object_stats[obj_name]['total_confidence'] += det['confidence']
            object_stats[obj_name]['detections'].append(det)
        
        # 计算评分 - 基于当前帧数据
        best_object = None
        best_score = -1
        best_detection = None
        
        for obj_name, stats in object_stats.items():
            avg_confidence = stats['total_confidence'] / stats['count']
            
            # 使用当前帧内的频率和置信度
            frequency_score = min(stats['count'] / 3.0, 1.0) * 0.2
            confidence_score = avg_confidence * 0.8
            total_score = frequency_score + confidence_score
            
            rospy.loginfo("📈 [帧{}]物体评分: {} -> 频率={:.3f}(计数{}), 置信度={:.3f}, 总分={:.3f}".format(
                frame_id, obj_name, frequency_score, stats['count'], confidence_score, total_score))
            
            if total_score > best_score:
                best_score = total_score
                best_object = obj_name
                # 选择置信度最高的检测作为代表
                best_detection = max(stats['detections'], key=lambda x: x['confidence'])
        
        # 更新最佳目标
        if best_object and best_score > self.pre_detection_config['min_score_threshold']:
            self.current_best_target = {
                'object': best_object,
                'detection': best_detection['detection'],
                'calculated_coords': best_detection['calculated_coords'],  # 使用存储坐标
                'score': best_score,
                'update_time': current_time,
                'frame_id': frame_id  # 关键：存储帧ID
            }
            rospy.loginfo("🎯 [帧{}]更新最佳目标: {} (得分: {:.3f}, 帧ID: {})".format(
                frame_id, best_object, best_score, frame_id))
        else:
            self.current_best_target = None
            if best_object:
                rospy.loginfo("📉 [帧{}]目标评分不足: {} (得分: {:.3f}, 阈值: {:.3f})".format(
                    frame_id, best_object, best_score, self.pre_detection_config['min_score_threshold']))

    def smart_pre_detection_publish(self, frame, stamp, frame_id):
        """智能预识别发布 - 严格使用帧ID匹配的坐标"""
        if self.current_best_target is None:
            return
            
        if self.direction_published:
            return
            
        if self.service_called:
            return
            
        current_time = time.time()
        if current_time - self.last_pre_detection_time < 0.5:
            return
        
        # 检查帧ID匹配
        if self.current_best_target.get('frame_id') != frame_id:
            rospy.logwarn("⚠️ 帧ID不匹配: 最佳目标帧ID={}, 当前帧ID={}".format(
                self.current_best_target.get('frame_id'), frame_id))
            return
        
        # 严格的新鲜度检查
        if current_time - self.current_best_target['update_time'] > self.pre_detection_config['target_freshness']:
            rospy.logwarn("⚠️ 最佳目标已过期: {} (年龄: {:.2f}s)".format(
                self.current_best_target['object'], 
                current_time - self.current_best_target['update_time']))
            return
        
        obj_name = self.current_best_target['object']
        score = self.current_best_target['score']
        target_frame_id = self.current_best_target['frame_id']
        
        # 严格使用存储的坐标
        target_x, target_y = self.current_best_target['calculated_coords']
        
        rospy.loginfo("🎯 [帧{}]准备发布预识别: {} (得分: {:.3f}, 坐标: ({:.2f}, {:.2f}))".format(
            target_frame_id, obj_name, score, target_x, target_y))
        
        # 检查坐标是否在边界内
        if not self.is_point_in_boundary(target_x, target_y):
            rospy.logwarn("❌ [帧{}]坐标超出边界，取消发布: {} -> ({:.2f}, {:.2f})".format(
                target_frame_id, obj_name, target_x, target_y))
            return
        
        rospy.loginfo("📍 [帧{}]发布坐标: {} -> ({:.2f}, {:.2f})".format(
            target_frame_id, obj_name, target_x, target_y))
        self.publish_target_position(target_x, target_y, 0.0)
        
        self.direction_published = True
        self.last_pre_detection_time = current_time
        
        rospy.loginfo("✅ [帧{}]预识别发布完成 - 帧ID严格匹配".format(target_frame_id))

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
            
            # 查找最近检测结果 - 使用更短的时间窗口
            recent_detections = []
            for det in reversed(self.detection_history):
                time_diff = current_time - det['timestamp']
                if time_diff < 3.0:  # 从5秒降至3秒
                    recent_detections.append(det)
                    rospy.loginfo("有效检测: {} ({}秒前, 帧ID:{})".format(
                        det['object'], time_diff, det.get('frame_id', 'N/A')))
                if len(recent_detections) >= 10:  # 减少最大数量
                    break
            
            rospy.loginfo("最近3秒内的检测数量: {}".format(len(recent_detections)))
            
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
                required_count = min(2, len(recent_detections) // 2 + 1)  # 降低要求
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
        self.frame_counter = 0  # 重置帧计数器
        
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
