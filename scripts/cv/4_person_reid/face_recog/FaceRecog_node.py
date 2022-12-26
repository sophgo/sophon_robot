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
from sensor_msgs.msg import CompressedImage, Image
from sophon_robot.msg import Bbox,Bboxes,Frame

import numpy as np
from cv_bridge import CvBridge, CvBridgeError

from FaceRecog import face

class FaceRecog_node(object):
    def __init__(self, faceDP,faceRP, known_people_path=""):
        self.camera_name = rospy.get_param("~camera_name","usb_cam")
        self.topic_name = rospy.get_param("~topic_name","face_recog")

        self.bridge = CvBridge()
        if(self.camera_name == "usb_cam"):
            self.image_sub = rospy.Subscriber(self.camera_name+"/image_raw/compressed",CompressedImage,self.callback)
        elif self.camera_name == "camera":
            self.image_sub = rospy.Subscriber(self.camera_name+"/color/image_raw",Image,self.callback)
        self.image_pub = rospy.Publisher(self.topic_name+"/compressed",CompressedImage,queue_size=10)

        # init infer engine
        self.engine = face(faceDP, faceRP, known_people_path)
        print("init node success.")

    def callback(self,data):
        # get msg and convert to cv::mat
        try:
            if self.camera_name == "usb_cam":
                frame = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            elif self.camera_name == "camera":
                frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        # return result frame
        # t1 = time.time() 
        res_frame = self.engine.inference(frame)
        # t2 = time.time()
        # print("time: ", t2 -t1)

        try:
            self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(res_frame))
        except CvBridgeError as e:
            print(e)

def main(args):
    rospy.init_node('Face_recog', anonymous=True)
    ic = FaceRecog_node("/home/linaro/robot_ws/src/sophon_robot/data/cv/4_person_reid/bmodel/detect/compilation.bmodel",
                        "/home/linaro/robot_ws/src/sophon_robot/data/cv/4_person_reid/bmodel/feature/compilation.bmodel",
                        "/home/linaro/robot_ws/src/sophon_robot/data/cv/4_person_reid/know_persons")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    # cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
