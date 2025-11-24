#!/usr/bin/env python3
import rospy
from service.srv import Service, ServiceResponse
from std_msgs.msg import String
import threading

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
    
    def start_simulation(self, target_object):
        """启动仿真并等待结果"""
        self.simulation_result = None
        self.result_received.clear()
        
        # 发布开始指令和目标物品
        start_msg = String()
        start_msg.data = target_object
        self.start_pub.publish(start_msg)
        rospy.loginfo(f"已发布仿真开始指令，目标: {target_object}")
        
        # 等待结果（带超时）
        if self.result_received.wait(timeout=60.0):  # 60秒超时
            return self.simulation_result
        else:
            return "仿真任务超时"

def handle_task_service(req):
    rospy.loginfo(f"B收到开始指令，目标物品: {req.target_object}")
    
    coordinator = SimulationCoordinator()
    result_message = coordinator.start_simulation(req.target_object)
    
    rospy.loginfo(f"B返回结果: {result_message}")
    return ServiceResponse(success=True, result=result_message)

if __name__ == "__main__":
    rospy.init_node("b_server")
    service = rospy.Service("task", Service, handle_task_service)
    rospy.loginfo("B服务启动")
    rospy.spin()
