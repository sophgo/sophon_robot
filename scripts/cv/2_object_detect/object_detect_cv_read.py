#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import datetime
import imutils
import time
import argparse
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

import SophonModel

class Object_Detector(object):
    def __init__(self, bmodel_path, labels_path):
        self.topic_name = rospy.get_param("~topic_name","object_detect")
        self.image_pub = rospy.Publisher(self.topic_name+"/compressed",CompressedImage,queue_size=10)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture("/dev/video0")

        # init infer engine
        self.engine = SophonModel.BmodelEngine(bmodel_path, labels_path)
        self.avg = None
    
    def get_results(self):
         

    def run(self):
        # get msg and convert to cv::mat
        while rospy.is_shutdown():
            success, frame = self.cap.read()
            if not success:
                print("unable to read frame from usb camera.")
                break 
                    
            # use ssd model infer, and draw box on frame
            res_frame = self.engine.get_inference_img(frame)
            
            # publish msg
            try:
                self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(res_frame))
                pass
            except CvBridgeError as e:
                print(e)

def main(args):
    rospy.init_node('object_detector', anonymous=True)
    ic = Object_Detector(bmodel_path="/home/linaro/test_ws/src/SogoBot/data/ssd-vgg-300x300-int8-1b.bmodel",
                        labels_path = "/home/linaro/test_ws/src/SogoBot/data/voc_labels.txt")
    try:
        ic.run()
    except KeyboardInterrupt:
        print("Shutting down")
    # cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)