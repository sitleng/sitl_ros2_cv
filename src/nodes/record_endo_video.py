#!/usr/bin/env python3

import cv2
import rospy
import os
from sensor_msgs.msg import CompressedImage, CameraInfo
import message_filters
from cv_bridge import CvBridge


class RECORD_ENDO_VIDEO():
    def __init__(self,params):
        self.caminfo = rospy.wait_for_message("/ecm/left/camera_info",CameraInfo)
        leftpath  = params["save_dir"] + "/left"
        rightpath = params["save_dir"] + "/right"
        if not os.path.exists(leftpath) or not os.path.exists(rightpath):
            os.makedirs(leftpath)
            os.makedirs(rightpath)
        self.br = CvBridge()
        self.bottom = int(0.1*self.caminfo.height)
        res = (self.caminfo.width,self.caminfo.height+self.bottom)
        fourcc = cv2.VideoWriter_fourcc(*params["fourcc"])
        self.left  = None
        self.right = None
        self.left_vid  = cv2.VideoWriter(leftpath+"/left.mp4",fourcc,params["fps"],res,True)
        self.right_vid = cv2.VideoWriter(rightpath+"/right.mp4",fourcc,params["fps"],res,True)

    def callback(self,imgmsgL,imgmsgR):
        imgL = self.br.compressed_imgmsg_to_cv2(imgmsgL)
        extL = cv2.copyMakeBorder(imgL, 0, self.bottom, 0, 0, cv2.BORDER_CONSTANT, None, [255,255,255])
        self.left = cv2.putText(extL, str(imgmsgL.header.stamp.to_sec()), (self.bottom,int(1.07*self.caminfo.height)),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 1, cv2.LINE_AA)
        imgR = self.br.compressed_imgmsg_to_cv2(imgmsgR)
        extR = cv2.copyMakeBorder(imgR, 0, self.bottom, 0, 0, cv2.BORDER_CONSTANT, None, [255,255,255])
        self.right = cv2.putText(extR, str(imgmsgR.header.stamp.to_sec()), (self.bottom,int(1.07*self.caminfo.height)),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 1, cv2.LINE_AA)
        
if __name__ == "__main__":
    rospy.init_node("ecm_record_video")
    node_name = rospy.get_name()
    params = rospy.get_param(node_name)
    app = RECORD_ENDO_VIDEO(params)
    try:
        rospy.loginfo("Start recording ecm...")
        imgmsgL = message_filters.Subscriber('/ecm/left_rect/image_color',CompressedImage)
        imgmsgR = message_filters.Subscriber('/ecm/right_rect/image_color',CompressedImage)
        ts = message_filters.ApproximateTimeSynchronizer([imgmsgL,imgmsgR], queue_size=10, slop=0.01)
        ts.registerCallback(app.callback)
        r = rospy.Rate(60)
        while not rospy.is_shutdown():
            if app.left is not None:
                app.left_vid.write(app.left)
            if app.right is not None:
                app.right_vid.write(app.right)
            r.sleep()
    except Exception as e:
        rospy.loginfo(e)
    finally:
        app.left_vid.release()
        app.right_vid.release()
    