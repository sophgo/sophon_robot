#!/usr/bin/env python3

import os
import sys
import time
import argparse

import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

sys.path.append("../2_object_detect")
from yolov5_bmcv_pipeline_model import YOLOV5_BMCV_Detector2

from deep_sort import build_tracker
from config import get_config
from draw import draw_boxes

class Tracker_Publisher(object):
    def __init__(self, det_bmodel_path, det_labels_path):
        self.camera_name = rospy.get_param("~camera_name", "usb_cam")
        self.topic_name = rospy.get_param("~topic_name","Tracker")

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(self.camera_name+"/image_raw/compressed",CompressedImage, self.callback)
        self.tracker_img_pub = rospy.Publisher(self.topic_name + "/res_image/compressed", CompressedImage, queue_size = 10)

        self.engine = YOLOV5_BMCV_Detector2(bmodel_path = det_bmodel_path,
                                      tpu_id = 0,
                                      class_names_path= det_labels_path,
                                      confThreshold=0.5,
                                      nmsThreshold=0.5,
                                      objThreshold=0.1,
                                      draw_result = False)
        
        self.cfg = get_config()
        self.cfg.merge_from_file("/home/linaro/workspace/robot_ws/src/sophon_robot/scripts/cv/3_multi_object_track/deep_sort/configs/deep_sort.yaml")
        self.tracker = build_tracker(self.cfg, use_tpu = False)

    def callback(self,data):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        res_frame, res = self.engine.inference(frame)
        num_object = 0
        if res is not None:
            # publish msg
            frame_id, result_dict = res
            num_object=len(result_dict) if result_dict is not None else 0
        bbox_xywh = np.zeros((num_object,4))
        confidences = np.zeros(num_object)
        for i in range(num_object):
            bbox_xywh[i][0] = result_dict[i]["det_box"][0] + result_dict[i]["det_box"][2] / 2
            bbox_xywh[i][1] = result_dict[i]["det_box"][1] + result_dict[i]["det_box"][3] / 2 
            bbox_xywh[i][2] = result_dict[i]["det_box"][2]
            bbox_xywh[i][3] = result_dict[i]["det_box"][3]
            confidences[i]  = result_dict[i]["conf"]
        track_results = self.tracker.update(bbox_xywh, confidences)
        # draw boxes for visualization
        track_frame = frame
        if len(track_results) > 0:
            bbox_tlwh = []
            bbox_xyxy = track_results[:, :4]
            identities = track_results[:, -1]
            # bbox_xyxy = bbox_xywh[:]
            # bbox_xyxy[:,2] = bbox_xywh[:,0] + bbox_xywh[:,2]
            # bbox_xyxy[:,3] = bbox_xywh[:,1] + bbox_xywh[:,3]
            draw_boxes(track_frame, bbox_xyxy, identities)

        try:
            self.tracker_img_pub.publish(self.bridge.cv2_to_compressed_imgmsg(track_frame))
            # if res is not None:
            #     self.bboxes_pub.publish(bboxes)
        except CvBridgeError as e:
            print(e)


def main(args):
    rospy.init_node("Tracker", anonymous=True)
    ic = Tracker_Publisher(det_bmodel_path = "/home/linaro/workspace/robot_ws/src/sophon_robot/data/cv/2_object_detect/yolov5s_480x640_v6.1_1output_int8_1b.bmodel",
                            det_labels_path = "/home/linaro/workspace/robot_ws/src/sophon_robot/data/cv/2_object_detect/coco_labels.txt")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")



if __name__ == "__main__":
    main(sys.argv)
