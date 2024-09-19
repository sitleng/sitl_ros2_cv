#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import rospy
import message_filters
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import random

class STEREO_VIEW():
    def __init__(self):
        self.br = CvBridge()
        self.stereo_img = None

    def __del__(self):
        print("Shutting down...")

    def raw_callback(self, left_img_msg, right_img_msg):
        left  = self.br.imgmsg_to_cv2(left_img_msg)
        right = self.br.imgmsg_to_cv2(right_img_msg)
        if left is not None and right is not None:
            self.stereo_img = np.concatenate((left, right), axis=1)

    def compressed_callback(self,left_img_msg, right_img_msg):
        left  = self.br.compressed_imgmsg_to_cv2(left_img_msg)
        right = self.br.compressed_imgmsg_to_cv2(right_img_msg)
        if left is not None and right is not None:
            self.stereo_img = np.concatenate((left, right), axis=1)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Image_View',
        description='Display Camera Images in ROS...',
        epilog='Text at the bottom of help')
    parser.add_argument('-t', '--img_type',    type=str, help='Type of the ROS Image: raw or compressed')
    parser.add_argument('-r', '--rate',        type=int, help='Publish rate of the ROS Image')
    parser.add_argument('-ln', '--left_name', type=str, help='Topic name of the Left Image')
    parser.add_argument('-rn', '--right_name', type=str, help='Topic name of the Left Image')
    args = parser.parse_args()
    app = STEREO_VIEW()
    rospy.init_node("stereo_view_"+str(random.randint(0,100)))
    rospy.loginfo("Start Displaying Images...")
    rospy.loginfo("Press q to quit the program...")
    if args.img_type == "compressed":
        left_img_msg = message_filters.Subscriber(args.left_name, CompressedImage)
        right_img_msg = message_filters.Subscriber(args.right_name, CompressedImage)
        ts = message_filters.ApproximateTimeSynchronizer([left_img_msg, right_img_msg], queue_size=10, slop=0.05)
        ts.registerCallback(app.compressed_callback)
    elif args.img_type == "raw":
        left_img_msg = message_filters.Subscriber(args.left_name, Image)
        right_img_msg = message_filters.Subscriber(args.right_name, Image)
        ts = message_filters.ApproximateTimeSynchronizer([left_img_msg, right_img_msg], queue_size=10, slop=0.05)
        ts.registerCallback(app.raw_callback)
    try:
        cv2.namedWindow("stereo_image", cv2.WINDOW_NORMAL)
        r = rospy.Rate(args.rate)
        while not rospy.is_shutdown():
            if app.stereo_img is not None:
                cv2.imshow("stereo_image", app.stereo_img)
                if cv2.waitKey(1) == ord('q'):
                    break
            r.sleep()
    except Exception as e:
        rospy.loginfo(e)
    finally:
        cv2.destroyAllWindows()
        del app