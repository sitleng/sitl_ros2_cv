#!/usr/bin/env python3

import cv2
import rospy
import os
from sensor_msgs.msg import CompressedImage, CameraInfo
import traceback
from cv_bridge import CvBridge
import argparse

class REC_ENDO_VIDEO():
    def __init__(self,args):
        caminfo = rospy.wait_for_message("/ecm/left_rect/camera_info",CameraInfo)
        fullpath     = args.save_path + "/" + args.video_name
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        self.br = CvBridge()
        res = (caminfo.width, caminfo.height)
        fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
        # fourcc = -1
        self.frame  = None
        self.video  = cv2.VideoWriter(fullpath, fourcc, args.fps, res, True)

    def callback(self,frame_msg):
        self.frame = self.br.compressed_imgmsg_to_cv2(frame_msg)
        
if __name__ == "__main__":
    rospy.init_node("rec_endo_video")
    node_name = rospy.get_name()
    parser = argparse.ArgumentParser(
        prog='Record Video of Endoscopic camera',
        epilog='Text at the bottom of help')
    parser.add_argument('-sp', '--save_path',  type=str, help='path to save the video...')
    parser.add_argument('-vn', '--video_name', type=str, help='Name of the video (i.e. temp.mp4)')
    parser.add_argument('-fc', '--fourcc',     type=str, help='Fourcc codec of the video... (i.e. MP4V)')
    parser.add_argument('-r',  '--fps',        type=int, help='FPS of the video...')
    args = parser.parse_args()
    app = REC_ENDO_VIDEO(args)
    try:
        rospy.loginfo("Start recording ecm...")
        frame_msg = rospy.Subscriber('/ecm/left_rect/image_color', CompressedImage, app.callback)
        r = rospy.Rate(60)
        while not rospy.is_shutdown():
            if app.frame is not None:
                app.video.write(app.frame)
            r.sleep()
    except:
        traceback.print_exc()
    finally:
        app.video.release()
    