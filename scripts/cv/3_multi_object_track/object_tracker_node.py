#!/usr/bin/env python3

import os
import sys
import time
import argparse

import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sophon_robot.msg import Bboxes

import cv2
import numpy as np

from deep_sort import build_tracker
from config import get_config

class Tracker_Publisher_Node(object):
    def __init__(self, det_bmodel_path, det_labels_path):
        self.detect_name = rospy.get_param("~detect_name", "object_detect")
        self.topic_name = rospy.get_param("~topic_name","Tracker")

        # create subscriber, subcriber listen msg from detect's bbox topic
        self.detect_sub = rospy.Subscriber(self.detect_name+"/bbox", Bboxes, self.callback)
        # create publisher, publisher puplish msg about track results.
        self.tracker_pub = rospy.Publisher(self.topic_name + "/tracker_res", Bboxes, queue_size = 10)
        
        self.cfg = get_config()
        self.cfg.merge_from_file("/home/linaro/workspace/robot_ws/src/sophon_robot/scripts/cv/3_multi_object_track/deep_sort/configs/deep_sort.yaml")
        self.tracker = build_tracker(self.cfg, use_tpu = False)

    def callback(self, det_res):
        #data is instance of Bboxes
        bbox_xywh = np.zeros((det_res.num_object,4))
        confidences = np.zeros((det_res.num_object))
        for i in range(det_res.num_object):
            bbox_xywh[i][0] = det_res.bboxes[i].left_top_x + det_res.bboxes[i].width / 2
            bbox_xywh[i][1] = det_res.bboxes[i].left_top_y + det_res.bboxes[i].height / 2
            bbox_xywh[i][2] = det_res.bboxes[i].width
            bbox_xywh[i][3] = det_res.bboxes[i].height
            confidences[i]    = det_res.bboxes[i].conf
        # [[x,y,x,y,id],...]
        print(bbox_xywh)
        track_results = self.tracker.update(bbox_xywh, confidences)



class ObjectTracker(object):
    def __init__(self, det_bmodel_path, det_labels_path = None):

        self.detector = SophonModel.BmodelEngine(bmodel_path, labels_path)
        self.class_names = self.detector.labels
        self.tracker = build_tracker(cfg)


    def run(self):
        results = []
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)

            # select person class
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


def main(args):
    rospy.init_node("Tracker", anonymous=True)
    ic = Tracker_Publisher_Node(det_bmodel_path = "",
                            det_labels_path = "/home/linaro/test_ws/src/SogoBot/data/voc_labels.txt")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")



if __name__ == "__main__":
    main(sys.argv)
