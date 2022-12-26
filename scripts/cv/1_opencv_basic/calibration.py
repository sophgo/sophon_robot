#!/usr/bin/env python3
import roslib
import sys
import rospy
import cv2 as cv
import numpy.matlib 
import numpy as np

import random
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage,Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):
    self.camera_name = rospy.get_param("~camera_name","usb_cam")
    self.topic_name = rospy.get_param("~topic_name","calibration_image")

    self.bridge = CvBridge()
    if self.camera_name == "usb_cam":
      self.image_sub = rospy.Subscriber(self.camera_name+"/image_raw/compressed",CompressedImage,self.callback)
      self.image_info_sub = rospy.Subscriber(self.camera_name+"/camera_info",CameraInfo,self.info_callback)
    elif self.camera_name == "camera":
      self.image_sub = rospy.Subscriber(self.camera_name+"/color/image_raw",Image,self.callback)
      self.image_info_sub = rospy.Subscriber(self.camera_name+"/color/camera_info",CameraInfo,self.info_callback)
    self.image_pub = rospy.Publisher(self.topic_name+"/compressed",CompressedImage,queue_size=10)

    self.mtx = None
    self.dist = None

  def info_callback(self,data):
    self.dist = np.array(data.D).reshape(1,5)
    self.mtx = np.array(data.K).reshape(3,3)

  def callback(self,data):
    try:
      if self.camera_name == "usb_cam":
        img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
      elif self.camera_name == "camera":
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    h,w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))

    # undistort
    dst = cv.undistort(img, self.mtx, self.dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # cv.imshow("Image window", img)
    # cv.imshow("Calibration Image window", dst)
    # cv.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(dst))
    except CvBridgeError as e:
      print(e)

def main(args):
  rospy.init_node('image_contours', anonymous=True)
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  # cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

