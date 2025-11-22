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
from collections import deque

# ===================== 核心修复1：添加YOLO类的导入 =====================
# 先添加ultralytics的库路径（防止ROS环境找不到）
sys.path.insert(0, "/home/x/anaconda3/envs/yolov11/lib/python3.8/site-packages")
try:
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator
    print("✅ 成功导入ultralytics的YOLO类")
except ImportError as e:
    print(f"❌ 导入YOLO失败: {e}")
    sys.exit(1)

# 解决libp11-kit符号冲突：强制加载系统libffi
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libffi.so.7'  # 若不存在则改为libffi.so.8
os.environ['LD_LIBRARY_PATH'] = '/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu'

# 全局禁用CUDA（针对RTX 5060 sm_120架构）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
torch.cuda.is_available = lambda: False

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
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
        
        # TF2支持：增加超时容错
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))  # 缓冲区10秒
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 精确配置
        self.config = {
            'min_confidence': 0.30,
            'camera_hfov': 1.3962634,    # 80度水平视野
            'camera_vfov': 0.785,        # 45度垂直视野
            'camera_position': (0.125, 0, 0.175),
            'image_width': 1920,
            'image_height': 1080,
            'reference_object_height': 0.3,
            'reference_object_width': 0.7,
            'distance_weights': [0.9, 0.1, 0.0],
            'max_det': 20  # 新增：YOLO11的最大检测数
        }
        
        # 预识别配置
        self.pre_detection_config = {
            'time_window': 3.0,
            'min_score_threshold': 0.55,
            'target_freshness': 2.0
        }
        
        # 边界框配置
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
        
        # 模型配置：本地路径（核心修改1）
        self.model_path = rospy.get_param("~model", "/home/x/Downloads/ultralytics-main/runs/detect/train25/weights/best.pt")
        
        # 类别映射
        self.class_map = {
            0: ("水果", "香蕉"),
            1: ("水果", "西瓜"), 
            2: ("水果", "苹果"),
            3: ("食品", "蛋糕"),
            4: ("饮料", "牛奶"),
            5: ("饮料", "可乐"),
            6: ("蔬菜", "番茄"),
            7: ("蔬菜", "土豆"),
            8: ("蔬菜", "辣椒"),
        }
        
        # 初始化模型：增加重试机制
        self.model = self.load_model(retry=3)
        
        # 订阅摄像头图像
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback, queue_size=1, buff_size=2**24)  # 增大缓冲区
        
        rospy.loginfo("物体识别节点启动完成")
        rospy.loginfo("边界区域: [{:.2f}, {:.2f}] -> [{:.2f}, {:.2f}]".format(
            self.boundary_points[0][0], self.boundary_points[0][1],
            self.boundary_points[2][0], self.boundary_points[2][1]
        ))

    def load_model(self, retry=3):
        """使用Ultralytics官方API加载YOLO11权重，CPU模式"""
        model = None
        for i in range(retry):
            try:
                rospy.loginfo(f"第{i+1}次尝试加载YOLO11模型，权重路径: {self.model_path}")
                # 核心：直接用YOLO类加载权重，自动适配YOLO11架构
                model = YOLO(self.model_path)
                # 设置模型参数
                model.conf = self.config["min_confidence"]  # 置信度阈值
                model.iou = 0.4  # NMS IoU阈值
                model.max_det = self.config["max_det"]  # 最大检测数
                rospy.loginfo("YOLO11模型加载成功（CPU模式）！")
                return model
            except Exception as e:
                rospy.logerr(f"第{i+1}次模型加载失败: {str(e)}")
                if i == retry - 1:
                    rospy.logerr("所有重试均失败，模型加载失败")
                    return None
                rospy.sleep(1)  # 重试间隔1秒
        return model

    def is_point_in_boundary(self, x, y):
        """检查坐标是否在边界矩形内"""
        left = min(point[0] for point in self.boundary_points)
        right = max(point[0] for point in self.boundary_points)
        bottom = min(point[1] for point in self.boundary_points)
        top = max(point[1] for point in self.boundary_points)
        
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
            distance_height = (focal_length_v * self.config['reference_object_height']) / bbox_height if bbox_height > 0 else 2.0
            # 宽度法
            distance_width = (focal_length_h * self.config['reference_object_width']) / bbox_width if bbox_width > 0 else 2.0
            
            # 加权平均
            weights = self.config['distance_weights']
            distances = [distance_height, distance_width, 10.0]
            weighted_avg = sum(d * w for d, w in zip(distances, weights))
            
            return max(0.5, min(8.0, weighted_avg))
            
        except Exception as e:
            rospy.logwarn(f"距离估算失败: {e}")
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
            rospy.logwarn(f"角度计算失败: {e}")
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
            
            # 坐标变换：机器人坐标系转世界坐标系
            cos_yaw = math.cos(robot_yaw)
            sin_yaw = math.sin(robot_yaw)
            target_x_world = robot_x + target_x_robot * cos_yaw - target_y_robot * sin_yaw
            target_y_world = robot_y + target_x_robot * sin_yaw + target_y_robot * cos_yaw
            
            rospy.loginfo(f"坐标计算: {obj_name} -> ({target_x_world:.2f}, {target_y_world:.2f})m")
            return target_x_world, target_y_world
            
        except Exception as e:
            rospy.logwarn(f"坐标转换失败: {e}")
            robot_x, robot_y, robot_yaw = self.get_robot_pose()
            return robot_x + 2.0 * math.cos(robot_yaw), robot_y + 2.0 * math.sin(robot_yaw)

    def get_robot_pose(self):
        """获取机器人位姿：增加超时容错"""
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", 
                "base_link", 
                rospy.Time(0), 
                rospy.Duration(0.5)  # 超时0.5秒
            )
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            # 四元数转偏航角
            q = transform.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return x, y, yaw
            
        except Exception as e:
            rospy.logwarn(f"TF获取失败: {e}")
            return 0.0, 0.0, 0.0

    def task_callback(self, msg):
        """任务回调"""
        self.current_task = msg.data
        rospy.loginfo(f"任务类型更新: {self.current_task}")
        self.direction_published = False
        self.current_best_target = None

    def image_callback(self, msg):
        """图像回调：增加异常捕获和缓冲区处理"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.detection_pipeline(cv_image)
            
        except Exception as e:
            rospy.logwarn(f"图像回调异常: {e}")

    def detection_pipeline(self, frame):
        """检测流水线：适配YOLO11的推理逻辑"""
        if self.model is None:
            rospy.logwarn("模型未加载，跳过检测")
            return
            
        try:
            # ===================== 核心修复2：YOLO11的推理逻辑 =====================
            # YOLO11推理（CPU模式，禁用CUDA）
            results = self.model(frame, verbose=False)
            detections = []
            
            # 解析YOLO11的检测结果（替代原pandas写法）
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
                self.process_detections(detections, frame)
                if not self.service_called:
                    self.smart_pre_detection_publish(frame)
            else:
                if not self.session_active and len(self.detection_history) % 30 == 0:
                    self.publish_target_position(0.0, 0.0, 0.0)
                    
        except Exception as e:
            rospy.logwarn(f"检测流水线异常: {e}")

    def process_detections(self, detections, frame):
        """处理检测结果：增加有效检测过滤"""
        current_time = time.time()
        rospy.loginfo(f"处理 {len(detections)} 个检测")
        
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
                rospy.loginfo(f"✅ 有效检测: {obj_name} 置信度: {confidence:.3f}")
        
        rospy.loginfo(f"本次处理完成: {valid_detection_count} 个有效检测，检测历史长度: {len(self.detection_history)}")
        
        # 更新最佳预识别目标
        if not self.service_called:
            self.update_best_pre_detection_target()

    def update_best_pre_detection_target(self):
        """更新最佳预识别目标：增加判空"""
        if not self.detection_history or self.service_called:
            return
        
        current_time = time.time()
        recent_detections = [d for d in self.detection_history 
                           if current_time - d['timestamp'] < self.pre_detection_config['time_window']]
        
        if not recent_detections:
            return
        
        # 筛选任务相关检测
        task_related_detections = [d for d in recent_detections 
                                 if d['category'] == self.current_task] if self.current_task else recent_detections
        
        if not task_related_detections:
            return
        
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
            rospy.loginfo(f"🎯 更新最佳目标: {best_object} (得分: {best_score:.3f})")
        else:
            self.current_best_target = None

    def smart_pre_detection_publish(self, frame):
        """智能预识别发布：增加多重判空"""
        if (self.current_best_target is None or 
            self.direction_published or 
            self.service_called):
            return
            
        current_time = time.time()
        if (current_time - self.last_pre_detection_time < 0.5 or 
            current_time - self.current_best_target['update_time'] > self.pre_detection_config['target_freshness']):
            return
        
        detection = self.current_best_target['detection']
        obj_name = self.current_best_target['object']
        score = self.current_best_target['score']
        
        rospy.loginfo(f"🎯 准备发布预识别: {obj_name} (得分: {score:.3f})")
        
        target_x, target_y = self.transform_to_world_coordinates(detection, frame.shape, obj_name)
        
        # 检查坐标是否在边界内
        if not self.is_point_in_boundary(target_x, target_y):
            rospy.logwarn(f"❌ 坐标超出边界，取消发布: {obj_name} -> ({target_x:.2f}, {target_y:.2f})")
            return
        
        rospy.loginfo(f"📍 发布坐标: {obj_name} -> ({target_x:.2f}, {target_y:.2f})")
        self.publish_target_position(target_x, target_y, 0.0)
        
        self.direction_published = True
        self.last_pre_detection_time = current_time
        
        rospy.loginfo("✅ 预识别发布完成")

    def publish_target_position(self, x, y, z):
        """发布目标位置：增加消息头配置"""
        try:
            point_msg = PointStamped()
            point_msg.header.stamp = rospy.Time.now()
            point_msg.header.frame_id = "map"
            point_msg.point.x = x
            point_msg.point.y = y
            point_msg.point.z = z
            
            self.target_pub.publish(point_msg)
            rospy.loginfo(f"📤 坐标发布成功: ({x:.2f}, {y:.2f})")
            
        except Exception as e:
            rospy.logwarn(f"坐标发布异常: {e}")

    def handle_object_service(self, req):
        """服务处理：增加状态锁和异常捕获"""
        rospy.loginfo("=== 收到识别请求 ===")
        rospy.loginfo(f"当前任务: {self.current_task}")
        rospy.loginfo(f"检测历史长度: {len(self.detection_history)}")
        
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
                    rospy.loginfo(f"有效检测: {det['object']} ({time_diff:.1f}秒前)")
                if len(recent_detections) >= 15:
                    break
            
            rospy.loginfo(f"最近5秒内的检测数量: {len(recent_detections)}")
            
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
            
            rospy.loginfo(f"物体频率统计: { {obj: stats['count'] for obj, stats in object_stats.items()} }")
            
            if not object_stats:
                rospy.logwarn("物体统计为空")
                response.message = "NO_OBJECT_DETECTED"
                return response
            
            best_obj = max(object_stats.items(), key=lambda x: x[1]['count'])
            obj_name, stats = best_obj
            category = stats['category']
            count = stats['count']
            
            rospy.loginfo(f"最佳物体: {obj_name} (类别: {category}, 出现次数: {count})")
            
            # 检查任务匹配
            if self.current_task and category != self.current_task:
                rospy.logwarn(f"任务不匹配: 需要 {self.current_task}, 检测到 {category}")
                response.message = "WARN:" + obj_name
            else:
                required_count = min(3, len(recent_detections) // 2 + 1)
                rospy.loginfo(f"要求次数: {required_count} (当前: {count})")
                
                if count >= required_count:
                    response.message = obj_name
                    self.session_active = False
                    rospy.loginfo(f"✅ 确认物体: {obj_name}")
                else:
                    response.message = "CONTINUE_DETECTING"
                    rospy.loginfo(f"🔄 继续检测: {obj_name} ({count}/{required_count})")
            
            rospy.loginfo(f"服务返回: {response.message}")
            return response
            
        except Exception as e:
            rospy.logerr(f"服务处理异常: {e}")
            response.success = False
            response.message = "SERVICE_ERROR"
            return response
            
        finally:
            self.service_called = False
            rospy.loginfo("=== 服务处理完成 ===")
    

    def handle_reset_pre_detection(self, req):
        """重置预检测数据"""
        rospy.loginfo("=== 重置预检测数据 ===")
        
        # 清空检测历史
        self.detection_history.clear()
        self.current_best_target = None
        self.last_pre_detection_time = 0
        
        # 发布零点坐标覆盖之前的错误坐标
        self.publish_target_position(0.0, 0.0, 0.0)
        rospy.loginfo("发布零点坐标覆盖之前的预识别结果")
        
        # 重置发布状态
        self.direction_published = False
        
        response = TriggerResponse()
        response.success = True
        response.message = "预检测数据已重置，已发布零点坐标"
        return response

    def handle_reset_service(self, req):
        """重置视觉状态"""
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
    except Exception as e:
        rospy.logerr(f"节点启动失败: {e}")
