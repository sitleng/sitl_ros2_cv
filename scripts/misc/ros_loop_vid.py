#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

import cv2
import argparse
import traceback

class ROS_LOOP_VIDEO(object):
    def __init__(self, args):
        self.br  = CvBridge()
        self.cap = cv2.VideoCapture(args.path)
        self.R   = rospy.Rate(self.cap.get(cv2.CAP_PROP_FPS))
        self.vid_pub = rospy.Publisher("/loop_video/frame", CompressedImage, queue_size=10)

    def __del__(self):
        print("Shutting Down ROS_LOOP_VIDEO...")

    def run(self):
        rospy.loginfo("Start looping the video in ROS...")
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                msg = self.br.cv2_to_compressed_imgmsg(frame)
                msg.header.stamp = rospy.Time.now()
                self.vid_pub.publish(msg)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self.R.sleep()


if __name__ == "__main__":
    rospy.init_node("ros_loop_video")

    parser = argparse.ArgumentParser(
        prog='ROS_Loop_Video',
        description='Loop a video in ROS',
        epilog='Text at the bottom of help'
    )
    parser.add_argument("-p", "--path", type=str, help="Path of the video file...")
    args = parser.parse_args()

    app = ROS_LOOP_VIDEO(args)

    try:
        app.run()
    except Exception:
        traceback.print_exc()
    finally:
        app.cap.release()