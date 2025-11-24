#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from service.srv import Service, ServiceResponse
import threading

class AClientNode:
    def __init__(self):
        rospy.init_node("a_client")
        
        # 仿真协调器
        self.coordinator = SimulationCoordinator()
        
        # 从主任务获取拾取的物品
        self.picked_object = None
        self.object_received = threading.Event()
        self.object_sub = rospy.Subscriber('/picked_object', String, self.object_callback)
        
        # 等待B服务器服务
        rospy.loginfo("A客户端初始化，等待B服务器...")
        rospy.wait_for_service("task")
        self.task_client = rospy.ServiceProxy("task", Service)
        rospy.loginfo("A客户端就绪")
        
    def object_callback(self, msg):
        """从主任务状态机接收拾取的物品"""
        self.picked_object = msg.data
        self.object_received.set()
        rospy.loginfo(f"收到主任务拾取的物品: {self.picked_object}")
    
    def wait_for_picked_object(self, timeout=10.0):
        """等待获取拾取的物品"""
        if self.object_received.wait(timeout=timeout):
            return self.picked_object
        else:
            rospy.logwarn(f"等待拾取物品超时 ({timeout}秒)")
            return None

class SimulationCoordinator:
    def __init__(self):
        self.simulation_result = None
        self.result_received = threading.Event()
        
        # 发布开始指令
        self.start_pub = rospy.Publisher('/simulation_start', String, queue_size=1)
        # 订阅结果
        self.result_sub = rospy.Subscriber('/simulation_result', String, self.result_callback)
    
    def result_callback(self, msg):
        """接收仿真结果"""
        self.simulation_result = msg.data
        self.result_received.set()
        rospy.loginfo(f"收到仿真结果: {self.simulation_result}")
    
    def start_simulation(self, picked_object):
        """启动仿真并等待结果"""
        self.simulation_result = None
        self.result_received.clear()
        
        # 发布开始指令和拾取的物品
        start_msg = String()
        start_msg.data = picked_object
        self.start_pub.publish(start_msg)
        rospy.loginfo(f"已发布仿真开始指令，寻找物品: {picked_object}")
        
        # 等待结果（带超时）
        if self.result_received.wait(timeout=60.0):  # 60秒超时
            return self.simulation_result
        else:
            return "仿真任务超时"

def handle_task_service(req):
    """B服务器服务处理函数"""
    rospy.loginfo("A收到开始指令，启动仿真任务")
    
    # 创建A客户端实例
    a_client = AClientNode()
    
    # 等待从主任务获取拾取的物品
    rospy.loginfo("等待主任务发布拾取的物品...")
    picked_object = a_client.wait_for_picked_object(timeout=10.0)
    
    if not picked_object:
        error_msg = "错误：未收到主任务拾取的物品，无法启动仿真任务"
        rospy.logerr(error_msg)
        return ServiceResponse(success=False, result=error_msg)
    
    rospy.loginfo(f"获取到目标物品: {picked_object}")
    
    # 启动仿真任务
    result_message = a_client.coordinator.start_simulation(picked_object)
    
    rospy.loginfo(f"A返回结果: {result_message}")
    
    # 根据结果判断成功与否
    if "超时" in result_message or "失败" in result_message:
        return ServiceResponse(success=False, result=result_message)
    else:
        return ServiceResponse(success=True, result=result_message)

if __name__ == "__main__":
    # 作为服务端（被B服务器调用）
    rospy.init_node("a_client")
    
    # 创建服务
    service = rospy.Service("task", Service, handle_task_service)
    
    rospy.loginfo("A客户端服务启动，等待B服务器调用...")
    rospy.loginfo("注意：需要主任务在OBJECT_CONFIRMED状态发布/picked_object话题")
    rospy.spin()
