#!/usr/bin/env python
#coding:utf-8

import roslib
roslib.load_manifest('learning_tf')
import rospy
import math
import tf
import geometry_msgs.msg
import turtlesim.srv

if __name__ == '__main__':
    rospy.init_node('tf_turtle')
    listener = tf.TransformListener()

    rospy.wait_for_service('spawn') # 等待‘spawn’服务出现
    spawner = rospy.ServiceProxy('spawn', turtlesim.srv.Spawn)
    spawner(4, 2, 0, 'turtle2')

    turtle_vel = rospy.Publisher('turtle2/cmd_vel', geometry_msgs.msg.Twist, queue_size=1) # 发布话题，控制turtle2
    rate = rospy.Rate(10.0) # 话题中内容更新评率：10Hz

    while not rospy.is_shutdown():
        try: # 得到以turtle2为坐标系原点的turtle1的姿态信息（平移和旋转），储存在trans和rot中
            (trans,rot) = listener.lookupTransform('/turtle2', '/turtle1',rospy.Time(0)) 
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        angular = 4 * math.atan2(trans[1], trans[0]) # 计算turtle2前往turtle1的角速度
        linear = 0.5 * math.sqrt(trans[0] ** 2 + trans[1] ** 2) # 计算线速度
        cmd = geometry_msgs.msg.Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        turtle_vel.publish(cmd) # 发布话题，控制turtle2
        rate.sleep()
