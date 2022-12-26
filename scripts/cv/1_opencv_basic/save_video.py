#!/usr/bin/env python3

import message_filters
from sensor_msgs.msg import Image
import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
import datetime
import time
import os

global Saver 
Saver = None
global video_path 
video_path = None

class SaveVideo():
    def __init__(self,):
        video_name = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S.mp4')
        video_path = os.path.join("/data/videos", video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.Saver = cv2.VideoWriter(video_path, fourcc=fourcc, fps=30,frameSize=(640,480))
        self.suber = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        self.cv_bridge = CvBridge()

    def callback(self, rgb_msg):
        #import pdb;pdb.set_trace()
        rgb_img = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        h,w,c = rgb_img.shape
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        self.Saver.write(bgr_img)
        print(h,w)

def main():
    rospy.init_node('save_rgb_video', anonymous=True)
    sv = SaveVideo()
    rospy.spin()

if __name__ == "__main__":
    main()
