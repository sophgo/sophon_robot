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

from FaceRecog import face

class FaceRecog_node(object):
    def __init__(self, faceDP,faceRP, known_people_path=""):
        # self.camera_name = rospy.get_param("~camera_name","usb_cam") 
        self.topic_name = rospy.get_param("~topic_name","face_recog")

        self.bridge = CvBridge() #ros图像消息和opencv图像间进行转换
        
        # self.image_sub = rospy.Subscriber(self.camera_name+"/image_raw/compressed",CompressedImage,self.callback)
        self.cap = cv.VideoCapture("/home/linaro/video_test.mp4") #本地视频读取
        self.image_pub = rospy.Publisher(self.topic_name+"/compressed",CompressedImage,queue_size=10)

        # init infer engine
        self.engine = face(faceDP, faceRP, known_people_path)
        print("init node success.")

    def run(self):
        # get msg and convert to cv::mat
        try:
            # frame = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8") #读入视频流
            while not rospy.is_shutdown():
                success, frame = self.cap.read()
                #import pdb;pdb.set_trace()
                if not success:
                    self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    _, frame = self.cap.read()
                res_frame = self.engine.inference(frame)
                try:
                    self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(res_frame))
                except CvBridgeError as e:
                    print(e)
        except CvBridgeError as e:
            print(e)
        
        # return result frame
        # t1 = time.time() 
        # t2 = time.time()
        # print("time: ", t2 -t1)

        

def main(args):
    rospy.init_node('Face_recog', anonymous=True)
    ic = FaceRecog_node("/home/linaro/workspace/robot_ws/src/sophon_robot/data/cv/4_person_reid/bmodel/detect/compilation.bmodel",
                        "/home/linaro/workspace/robot_ws/src/sophon_robot/data/cv/4_person_reid/bmodel/feature/compilation.bmodel",
                        "/home/linaro/workspace/robot_ws/src/sophon_robot/data/cv/4_person_reid/know_persons")
    try:
        # rospy.spin() #打开话题
        ic.run()
    except KeyboardInterrupt:
        print("Shutting down")
    # cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
