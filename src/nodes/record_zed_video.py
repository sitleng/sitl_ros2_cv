#!/usr/bin/env python3

import cv2
import rospy
import os
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge


class RECORD_ZED_VIDEO():
    def __init__(self,params):
        self.caminfo = rospy.wait_for_message("/zedm/zed_node/left/camera_info",CameraInfo)
        leftpath  = params["save_dir"]
        if not os.path.exists(leftpath):
            os.makedirs(leftpath)
        self.br = CvBridge()
        res = (self.caminfo.width,self.caminfo.height)
        fourcc = cv2.VideoWriter_fourcc(*params["fourcc"])
        self.left  = None
        self.left_vid  = cv2.VideoWriter(leftpath+"/left.mp4",fourcc,params["fps"],res,True)

    def callback(self,imgmsgL):
        self.left = self.br.compressed_imgmsg_to_cv2(imgmsgL)
        
if __name__ == "__main__":
    rospy.init_node("ecm_record_video")
    node_name = rospy.get_name()
    params = rospy.get_param(node_name)
    app = RECORD_ZED_VIDEO(params)
    try:
        rospy.loginfo("Start recording zed...")
        rospy.Subscriber("/zedm/zed_node/left/image_rect_color/compressed",CompressedImage,app.callback)
        r = rospy.Rate(30)
        while not rospy.is_shutdown():
            if app.left is not None:
                app.left_vid.write(app.left)
            r.sleep()
    except Exception as e:
        rospy.loginfo(e)
    finally:
        app.left_vid.release()
    