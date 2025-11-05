# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
import time
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # 1. 与导航状态机通信的发布器（固定话题）
        self.object_pub = rospy.Publisher("/demo/object_result", String, queue_size=10)
        self.annotated_img_pub = rospy.Publisher("/detect/object_annotated", Image, queue_size=10)
        
        # 2. 模型配置
        self.model_path = rospy.get_param("~model", "/home/xxy/ros_competition_2025-main/camera/models/best.pt")  # 你的 best.pt 路径
        self.conf_thres = 0.5  # 置信度阈值
        
        # 3. 类别映射
        self.class_map = {
            0: ("水果", "苹果"),
            1: ("水果", "香蕉"),
            2: ("水果", "西瓜"),
            3: ("蔬菜", "辣椒"),
            4: ("蔬菜", "土豆"),
            5: ("蔬菜", "番茄"),
            6: ("饮料", "牛奶"),
            7: ("饮料", "可乐"),
            8: ("食品", "蛋糕")
        }
        
        # 4. 初始化变量
        self.latest_frame = None
        self.model = self.load_model()  # 加载 .pt 模型
        
        # 5. 订阅摄像头图像
        rospy.Subscriber("/detect/raw_image", Image, self.image_callback)
        rospy.loginfo("物体识别节点启动（适配 .pt 模型），已连接导航状态机")

    def load_model(self):
        try:
            # 加载本地 .pt 模型
            model = torch.hub.load(
                'ultralytics/yolov5',  
                'custom', 
                path=self.model_path,  # best.pt 路径
                force_reload=False  # 不强制重新下载
            )
            model.conf = self.conf_thres  # 设置置信度阈值
            model.eval()  # 推理模式
            rospy.loginfo(f".pt 模型加载成功：{self.model_path}")
            return model
        except Exception as e:
            rospy.logerr(f"模型加载失败：{e}")
            rospy.signal_shutdown("模型加载失败，退出节点")
            return None

    def image_callback(self, msg):
        """接收摄像头图像"""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"图像转换错误：{e}")

    def process_and_publish(self):
        """处理图像并发布结果"""
        if self.latest_frame is None or self.model is None:
            return
        
        frame = self.latest_frame.copy()
        
        # 用YOLO模型检测
        results = self.model(frame)  # 模型推理
        detections = results.pandas().xyxy[0].values  # 转换为numpy格式（x1,y1,x2,y2,conf,cls,name）
        
        # 筛选最佳结果
        best_det = None
        if len(detections) > 0:
            best_det = detections[np.argmax(detections[:, 4])]  
            cls_id = int(best_det[5])
            
            # 生成导航需要的格式：“类别:物体名”
            if cls_id in self.class_map:
                category, obj_name = self.class_map[cls_id]
                publish_str = f"{category}:{obj_name}"
                self.object_pub.publish(publish_str)
                rospy.loginfo(f"发布识别结果：{publish_str}（导航已接收）")
                
                # 标注图像
                x1, y1, x2, y2 = map(int, best_det[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{obj_name}({best_det[4]:.2f})", 
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            self.object_pub.publish("")  
        # 发布标注图像
        self.annotated_img_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    def run(self):
        """主循环"""
        rate = rospy.Rate(10)  # 10Hz处理
        while not rospy.is_shutdown():
            self.process_and_publish()
            rate.sleep()

if __name__ == '__main__':
    try:
        detector = ObjectDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("物体识别节点已停止")
