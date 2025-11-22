#!/usr/bin/env python3
import rospy
from service.srv import Service

if __name__ == "__main__":
    rospy.init_node("a_client")

    rospy.loginfo("A:等待机器B的服务")
    rospy.wait_for_service("task")
    rospy.loginfo("A:服务已就绪，发送开始指令")

    try:
        task_client = rospy.ServiceProxy("task", Service)
        response = task_client()

        if response.success:
            rospy.loginfo("任务完成")
            rospy.loginfo(f"收到来自B的结果: {response.result}")
        else:
            rospy.logerr("任务失败")

    except rospy.ServiceException as e:
        rospy.logerr(f"error: {e}")

#通过变量response.result接受结果