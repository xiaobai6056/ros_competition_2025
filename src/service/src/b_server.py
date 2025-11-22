#!/usr/bin/env python3
import rospy
from service.srv import Service, ServiceResponse

#模拟任务
def task():
    rospy.loginfo("B:开始运行(等待4秒)")
    rospy.sleep(4)
    rospy.loginfo("B:运行完成")
    return "目标物体识别:杯子"


def handle_task_service(req):
    rospy.loginfo("B收到开始指令")
    result_message = task()

    rospy.loginfo(f"B返回结果: {result_message}")
    return ServiceResponse(success=True, result=result_message)    # 返回结果

if __name__ == "__main__":
    rospy.init_node("b_server")
    service = rospy.Service("task", Service, handle_task_service)
    rospy.loginfo("B服务启动")
    rospy.spin()

#上面功能只是一个模板
#识别结果通过变量result_message(字符串)发送给A,到时候可以改成其他的
