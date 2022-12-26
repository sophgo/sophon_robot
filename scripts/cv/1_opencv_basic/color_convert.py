#!/usr/bin/env python3
from __future__ import print_function

import roslib
import sys
import rospy
import cv2 as cv
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):
    self.camera_name = rospy.get_param("~camera_name","camera")
    self.topic_name = rospy.get_param("~topic_name","gray_convert")

    self.bridge = CvBridge()
    if(self.camera_name == "usb_cam"):
      self.image_sub = rospy.Subscriber(self.camera_name+"/image_raw/compressed",CompressedImage,self.callback)
    elif self.camera_name == "camera":
      self.image_sub = rospy.Subscriber(self.camera_name+"/color/image_raw",Image,self.callback)
    self.image_pub = rospy.Publisher(self.topic_name+"/compressed",CompressedImage,queue_size=10);

  def callback(self,data):
    try:
      if self.camera_name == "usb_cam":
        img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
      elif self.camera_name == "camera":
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # Our operations on the frame come here
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #cv.imshow("OpenCV Video window", gray)
    #cv.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(gray))
    except CvBridgeError as e:
      print(e)

def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)

