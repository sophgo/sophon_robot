#!/usr/bin/env python3
import roslib
import sys
import rospy
import cv2 as cv
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage,Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):
    self.camera_name = rospy.get_param("~camera_name","usb_cam")
    self.topic_name = rospy.get_param("~topic_name","face_detect")
    self.file_name = rospy.get_param("~file_name",
    "/home/linaro/robot_ws/src/sophon_robot/data/cv/1_opencv_basic/haarcascade_frontalface_alt.xml")

    self.bridge = CvBridge()
    if(self.camera_name == "usb_cam"):
      self.image_sub = rospy.Subscriber(self.camera_name+"/image_raw/compressed",CompressedImage,self.callback)
    elif self.camera_name == "camera":
      self.image_sub = rospy.Subscriber(self.camera_name+"/color/image_raw",Image,self.callback)
    self.image_pub = rospy.Publisher(self.topic_name+"/compressed",CompressedImage,queue_size=10)
    self.face_cascade = cv.CascadeClassifier(self.file_name)

  def callback(self,data):
    try:
        if self.camera_name == "usb_cam":
            frame = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        elif self.camera_name == "camera":
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    faces = self.face_cascade.detectMultiScale(frame_gray,1.2, 5,0,(50,50))
    for (x,y,w,h) in faces:
        frame = cv.rectangle(frame, (x, y),(x+w, y+h), ( 0, 255, 0), 2)

    # cv.imshow("Face detection", frame)
    # cv.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(frame))
    except CvBridgeError as e:
      print(e)

def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  # cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

