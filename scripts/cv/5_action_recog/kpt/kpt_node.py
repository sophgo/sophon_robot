#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2 as cv
import datetime
import imutils
import time
import argparse
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sophon_robot.msg import Bbox,Bboxes,Frame

import numpy as np
from cv_bridge import CvBridge, CvBridgeError

from KptModel import Kpt
from KptModel_pipeline import Kpt_pipeline

class Kpt_node(object):
    def __init__(self, bmodel_path):
        self.sub_topic_name = rospy.get_param("~object_detect","object_detect")
        self.topic_name = rospy.get_param("~topic_name","kpt_detect")

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(self.sub_topic_name+"/data",Frame,self.callback)
        self.image_pub = rospy.Publisher(self.topic_name+"/compressed",CompressedImage,queue_size=10)

        # init infer engine
        self.engine = Kpt_pipeline(bmodel_path)
        

    def callback(self,data):
        # get msg and convert to cv::mat
        ori_data = data.img
        bboxes = data.bboxes
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(ori_data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        # return result frame
        # t1 = time.time() 
        res_frame = self.engine.inference(frame, bboxes)
        # t2 = time.time()
        # print("time: ", t2 -t1)

        try:
            self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(res_frame))
        except CvBridgeError as e:
            print(e)

def main(args):
    rospy.init_node('guesture_recognizator', anonymous=True)
    bmodel_path = "/home/linaro/workspace/robot_ws/src/sophon_robot/data/cv/5_action_recog/res50_coco_256x192-ec54d7f3_20200709.bmodel"
    ic = Kpt_node(bmodel_path)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    # cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
