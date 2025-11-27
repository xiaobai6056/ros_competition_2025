#!/usr/bin/env python3
import rospy
from service.srv import Service, ServiceResponse
from std_msgs.msg import String
import threading

class SimulationCoordinator:
    def __init__(self):
        self.simulation_result = None
        self.result_received = threading.Event()
        
        # 发布开始指令 - 使用latch=True确保消息不会丢失
        self.start_pub = rospy.Publisher('/simulation_start', String, queue_size=1, latch=True)
        # 订阅结果
        self.result_sub = rospy.Subscriber('/simulation_result', String, self.result_callback)
        
        # 给发布器时间建立连接
        rospy.sleep(0.1)
    
    def result_callback(self, msg):
        """接收仿真结果"""
        self.simulation_result = msg.data
        self.result_received.set()
        rospy.loginfo(f"收到仿真结果: {self.simulation_result}")
    
    def start_simulation(self, task_type):
        """启动仿真并等待结果"""
        self.simulation_result = None
        self.result_received.clear()
        
        # 发布开始指令和任务类型
        start_msg = String()
        start_msg.data = task_type
        
        # 确保消息被接收 - 发布多次
        for i in range(2):
            self.start_pub.publish(start_msg)
            rospy.loginfo(f"发布仿真开始指令，任务类型: {task_type}")
            if i == 0:  # 第一次发布后稍等
                rospy.sleep(0.1)
        
        # 等待结果（带超时）
        rospy.loginfo("等待仿真状态机返回结果...")
        if self.result_received.wait(timeout=60.0):  # 60秒超时
            rospy.loginfo(f"仿真任务完成，结果: {self.simulation_result}")
            return self.simulation_result
        else:
            rospy.logwarn("仿真任务超时")
            return "仿真任务超时"

def handle_task_service(req):
    """B服务器服务处理函数"""
    rospy.loginfo(f"B收到任务请求，任务类型: {req.task_type}")
    
    # 验证任务类型
    valid_tasks = ["食品", "水果", "蔬菜"]
    if req.task_type not in valid_tasks:
        error_msg = f"无效的任务类型: {req.task_type}，有效类型: {valid_tasks}"
        rospy.logerr(error_msg)
        return ServiceResponse(success=False, result=error_msg)
    
    coordinator = SimulationCoordinator()
    result_message = coordinator.start_simulation(req.task_type)
    
    rospy.loginfo(f"B返回结果: {result_message}")
    
    # 根据结果判断成功与否
    if "超时" in result_message or "失败" in result_message:
        return ServiceResponse(success=False, result=result_message)
    else:
        return ServiceResponse(success=True, result=result_message)

if __name__ == "__main__":
    rospy.init_node("b_server")
    
    # 预先初始化一次，确保发布器就绪
    rospy.loginfo("预初始化仿真协调器...")
    _ = SimulationCoordinator()
    
    service = rospy.Service("task", Service, handle_task_service)
    rospy.loginfo("B服务启动，等待A客户端调用...")
    rospy.loginfo("支持的任务类型: 食品、水果、蔬菜")
    rospy.spin()
