#!/usr/bin/env python3

import cv2
import argparse
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import random

class IMAGE_VIEW():
    def __init__(self):
        self.br = CvBridge()
        self.img = None

    def __del__(self):
        print("Shutting down...")

    def raw_callback(self,img_msg):
        self.img = self.br.imgmsg_to_cv2(img_msg)

    def compressed_callback(self,img_msg):
        self.img = self.br.compressed_imgmsg_to_cv2(img_msg)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Image_View',
        description='Display Camera Images in ROS...',
        epilog='Text at the bottom of help')
    parser.add_argument('-t', '--img_type',   type=str, help='Type of the ROS Image: raw or compressed')
    parser.add_argument('-r', '--rate',        type=int, help='Publish rate of the ROS Image')
    parser.add_argument('-n', '--topic_name', type=str, help='Topic name of the ROS Image')
    args = parser.parse_args()
    app = IMAGE_VIEW()
    rospy.init_node("image_view_"+str(random.randint(0,100)))
    rospy.loginfo("Start Displaying Images...")
    rospy.loginfo("Press q to quit the program...")
    if args.img_type == "compressed":
        rospy.Subscriber(args.topic_name,CompressedImage,app.compressed_callback)
    elif args.img_type == "raw":
        rospy.Subscriber(args.topic_name,Image,app.raw_callback)
    try:
        cv2.namedWindow(args.topic_name, cv2.WINDOW_NORMAL)
        r = rospy.Rate(args.rate)
        while not rospy.is_shutdown():
            if app.img is not None:
                cv2.imshow(args.topic_name, app.img)
                if cv2.waitKey(1) == ord('q'):
                    break
            r.sleep()
    except Exception as e:
        rospy.loginfo(e)
    finally:
        cv2.destroyAllWindows()
        del app