#!/usr/bin/env python3

import cv2
import rospy
import os
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge

class AUTO_DISSECT_RECORD_ENDO_VIDEO():
    def __init__(self,params):
        self.caminfo = rospy.wait_for_message("/ecm/left_rect/camera_info",CameraInfo)
        leftpath  = params["save_dir"]
        if not os.path.exists(leftpath):
            os.makedirs(leftpath)
        self.br = CvBridge()
        res = (self.caminfo.width,self.caminfo.height)
        fourcc = cv2.VideoWriter_fourcc(*params["fourcc"])
        self.left      = None
        self.left_vid  = cv2.VideoWriter(leftpath+"/audo_dissect_ecm.mp4",fourcc,params["fps"],res,True)

    def callback(self,imgmsgL):
        self.left = self.br.compressed_imgmsg_to_cv2(imgmsgL)
        
if __name__ == "__main__":
    rospy.init_node("auto_dis_rec_endo_vid")
    node_name = rospy.get_name()
    params = rospy.get_param(node_name)
    try:
        app = AUTO_DISSECT_RECORD_ENDO_VIDEO(params)
        rospy.loginfo("Start recording zed...")
        rospy.Subscriber("/ecm/left_rect/image_color",CompressedImage,app.callback)
        r = rospy.Rate(params["fps"])
        while not rospy.is_shutdown():
            if app.left is not None:
                app.left_vid.write(app.left)
            r.sleep()
    except Exception as e:
        rospy.loginfo(e)
    finally:
        app.left_vid.release()
    