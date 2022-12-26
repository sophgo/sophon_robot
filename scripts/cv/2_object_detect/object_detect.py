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

import SSD_Model
from yolov5_model import YOLOV5_Detector
from yolov5_bmcv_model import YOLOV5_BMCV_Detector
from yolov5_bmcv_pipeline_model import YOLOV5_BMCV_Detector2

class Object_Detector(object):
    def __init__(self):
        self.camera_name = rospy.get_param("~camera_name","usb_cam")
        self.topic_name = rospy.get_param("~topic_name","object_detect")
        self.bmodel_path = rospy.get_param("~bmodel", "/home/linaro/robot_ws/src/sophon_robot/data/cv/2_object_detect/yolov5s_480x640_v6.1_1output_int8_1b.bmodel")
        self.labels_path = rospy.get_param("~label", "/home/linaro/robot_ws/src/sophon_robot/data/cv/2_object_detect/coco_labels.txt")

        self.bridge = CvBridge()
        if(self.camera_name == "usb_cam"):
            self.image_sub = rospy.Subscriber(self.camera_name+"/image_raw/compressed",CompressedImage,self.callback)
        elif self.camera_name == "camera":
            self.image_sub = rospy.Subscriber(self.camera_name+"/color/image_raw",Image,self.callback)

        self.image_pub = rospy.Publisher(self.topic_name+"/compressed",CompressedImage,queue_size=10)
        self.bboxes_pub = rospy.Publisher(self.topic_name+"/data", Frame, queue_size = 10)

        # init infer engine
        # self.engine = SophonModel.BmodelEngine(bmodel_path, labels_path)
        self.engine = YOLOV5_BMCV_Detector2(bmodel_path = self.bmodel_path,
                                      tpu_id = 0,
                                      class_names_path= self.labels_path,
                                      confThreshold=0.5,
                                      nmsThreshold=0.5,
                                      objThreshold=0.1) 

    def callback(self,data):
        # get msg and convert to cv::mat
        try:
            if self.camera_name == "usb_cam":
                frame = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            elif self.camera_name == "camera":
                frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
                copy_frame = frame.copy()
        except CvBridgeError as e:
            print(e)
        # return result frame
        # t1 = time.time() 
        res_frame, res = self.engine.inference(frame)
        # t2 = time.time()
        # print("time: ", t2 -t1)
        
        FF = Frame()
        if res is not None:
            # publish msg
            frame_id, result_dict = res
            bboxes = Bboxes()
            bboxes.frame_id = frame_id
            bboxes.num_object=len(result_dict) if result_dict is not None else 0
            for i in range(bboxes.num_object):
                box = Bbox()
                box.frame_id = frame_id
                box.object_id = result_dict[i]["classId"]
                box.track_id = -1
                box.object_name = result_dict[i]["classId"]
                box.left_top_x = result_dict[i]["det_box"][0]
                box.left_top_y = result_dict[i]["det_box"][1]
                box.width = result_dict[i]["det_box"][2]
                box.height = result_dict[i]["det_box"][3]
                box.conf = result_dict[i]["conf"]
                bboxes.bboxes.append(box)
            FF.frame_id = frame_id
            if self.camera_name == "usb_cam":
                FF.img = data
            elif self.camera_name == "camera":
                FF.img = self.bridge.cv2_to_compressed_imgmsg(copy_frame)
            FF.bboxes = bboxes
        try:
            self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(res_frame))
            self.bboxes_pub.publish(FF)
        except CvBridgeError as e:
            print(e)

def main(args):
    rospy.init_node('object_detector', anonymous=True)
    ic = Object_Detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    # cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
