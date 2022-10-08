#!/usr/bin/env python 
#coding:utf-8

import roslib
roslib.load_manifest('learning_tf')
import rospy
import tf
import turtlesim.msg

def handle_turtle_pose(msg, turtlename):
    br = tf.TransformBroadcaster() # 创建坐标系广播员
    br.sendTransform((msg.x, msg.y, 0), # 平移信息
                    tf.transformations.quaternion_from_euler(0, 0, msg.theta), #旋转信息
                    rospy.Time.now(),
                    turtlename, # 子类坐标系
                    "world") # 父类坐标系

if __name__ == '__main__':
    rospy.init_node('turtle_tf_broadcaster')
    turtlename = rospy.get_param('~turtle') # 指定turtle名字 e.g. turtle1
    rospy.Subscriber('/%s/pose' % turtlename, # 话题名：格式：turtle*/pose
                    turtlesim.msg.Pose, # 话题中消息的变量类型
                    handle_turtle_pose, # 回调函数
                    turtlename)
    rospy.spin()
