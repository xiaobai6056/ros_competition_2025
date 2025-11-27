#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from service.srv import Service, ServiceRequest
import threading

class AClientNode:
    def __init__(self):
        rospy.init_node("a_client")
        
        # 存储任务类型和物品
        self.picked_object = None
        self.task_type = None
        
        # 订阅任务类型（提前接收，但不立即调用B服务）
        self.task_sub = rospy.Subscriber('/current_task', String, self.task_callback)
        
        # 订阅拾取物品（在WAITING_SIMULATION状态发布，触发B服务调用）
        self.object_sub = rospy.Subscriber('/picked_object', String, self.object_callback)
        
        # 连接到B服务器服务
        rospy.loginfo("A客户端初始化，连接B服务器...")
        rospy.wait_for_service("task")  # 等待B的服务
        self.b_task_client = rospy.ServiceProxy("task", Service)
        rospy.loginfo("A客户端就绪")
        
    def task_callback(self, msg):
        """提前接收任务类型，但不调用B服务"""
        self.task_type = msg.data
        rospy.loginfo(f"预接收任务类型: {self.task_type}")
        # 不立即调用B服务，等待物品发布
        
    def object_callback(self, msg):
        """在WAITING_SIMULATION状态接收物品，立即调用B服务"""
        self.picked_object = msg.data
        rospy.loginfo(f"收到拾取物品: {self.picked_object}")
        
        # ✅ 此时状态机已进入WAITING_SIMULATION，开始调用B服务器
        if self.task_type is not None:
            self.call_b_server()
        else:
            rospy.logwarn("收到物品但任务类型未设置，无法调用B服务")
        
    def call_b_server(self):
        """调用B服务器执行任务"""
        try:
            rospy.loginfo(f"调用B服务器，任务类型: {self.task_type}, 物品: {self.picked_object}")
            
            # 创建服务请求
            request = ServiceRequest()
            request.task_type = self.task_type  # "食品"、"水果"、"蔬菜"
            
            # 调用B服务器
            response = self.b_task_client(request)
            rospy.loginfo(f"B服务器返回 - 成功: {response.success}, 结果: {response.result}")
            
            # ✅ 结果会通过B服务器的/simulation_result话题发布
            # 状态机在WAITING_SIMULATION状态等待这个结果
            
        except rospy.ServiceException as e:
            rospy.logerr(f"调用B服务失败: {e}")

if __name__ == "__main__":
    client = AClientNode()
    rospy.spin()
