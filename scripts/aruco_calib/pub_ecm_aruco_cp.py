#!/usr/bin/env python3

import cv2
import numpy as np

import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge

from utils import tf_utils, aruco_utils

class PUB_ECM_ARUCO_CP():
    def __init__(self):
        self.br = CvBridge()
        self.ecmtip_marker_id  = 1
        self.ecmtip_marker_len = 0.05
        self.aruco_detector    = aruco_utils.load_aruco_detector()
        self.g_ecm_aruco_dvrk  = aruco_utils.load_default_tfs("ECM")[0]
        zed_rect_caminfo  = rospy.wait_for_message("/zedm/zed_node/left/camera_info",CameraInfo)
        self.zed_mtx, self.zed_dist = aruco_utils.load_caminfo(zed_rect_caminfo)
        self.g_ecmbase_zed, self.g_ecmtip_ecmdvrk = self.preprocess_tf_data()
        self.zed_ecmtip_rvecs  = np.array([])
        self.zed_ecmtip_tvecs  = np.array([])
        self.N = 10
        self.pub_ecm_aruco_cp = rospy.Publisher("/ECM/ARUCO/setpoint_cp",TransformStamped,queue_size=10)

    def preprocess_tf_data(self):
        tf_data = aruco_utils.load_tf_data('/home/leo/aruco_data/base_tfs.yaml')
        g_ecmbase_zed = np.array(tf_data["g_ecmbase_zed"])
        g_ecmtip_ecmdvrk = np.array(tf_data["g_ecmtip_ecmdvrk"])
        return g_ecmbase_zed, g_ecmtip_ecmdvrk

    def callback(self,zed_rect_left_msg):
        if len(self.zed_ecmtip_rvecs) == self.N:
            zed_ecmtip_rvecs_avg = aruco_utils.avg_aruco_rvecs(self.zed_ecmtip_rvecs)
            zed_ecmtip_tvecs_avg = np.mean(self.zed_ecmtip_tvecs,axis=0)
            g_zed_ecmtip         = tf_utils.cv2vecs2g(zed_ecmtip_rvecs_avg,zed_ecmtip_tvecs_avg)
            g_ecmbase_ecmtip = self.g_ecmbase_zed.dot(g_zed_ecmtip).dot(self.g_ecmtip_ecmdvrk)
            ecmbase_ecmtip_zed = tf_utils.g2tfstamped(g_ecmbase_ecmtip,rospy.Time.now(),"ECM_base","ECM_ARUCO")
            self.pub_ecm_aruco_cp.publish(ecmbase_ecmtip_zed)
            self.zed_ecmtip_rvecs  = np.array([])
            self.zed_ecmtip_tvecs  = np.array([])
            try:
                rospy.wait_for_message("ecm_moved_flag",Bool,timeout=10)
            except:
                pass
        else:
            zed_rect_left = self.br.imgmsg_to_cv2(zed_rect_left_msg)[:,:,:3].copy()
            zed_corners, zed_ids, _ = self.aruco_detector.detectMarkers(zed_rect_left)
            if zed_ids is not None:
                for i,idx in enumerate(zed_ids):
                    if idx == self.ecmtip_marker_id:
                        zed_ecmtip_rvec, zed_ecmtip_tvec, _ = cv2.aruco.estimatePoseSingleMarkers(zed_corners[i],self.ecmtip_marker_len,self.zed_mtx,self.zed_dist)
                        # zed_ecmtip_rvec = cv2.Rodrigues(cv2.Rodrigues(zed_ecmtip_rvec)[0].dot(self.g_ecm_aruco_dvrk[:3,:3]))[0]
                        if not self.zed_ecmtip_rvecs.any():
                            self.zed_ecmtip_rvecs = np.hstack((self.zed_ecmtip_rvecs,zed_ecmtip_rvec.reshape(-1)))
                            self.zed_ecmtip_tvecs = np.hstack((self.zed_ecmtip_tvecs,zed_ecmtip_tvec.reshape(-1)))
                        else:
                            self.zed_ecmtip_rvecs = np.vstack((self.zed_ecmtip_rvecs,zed_ecmtip_rvec.reshape(-1)))
                            self.zed_ecmtip_tvecs = np.vstack((self.zed_ecmtip_tvecs,zed_ecmtip_tvec.reshape(-1)))

if __name__ == "__main__":
    rospy.init_node("pub_ecm_aruco_cp")
    rospy.loginfo("Start Publishing ECM Tooltip Aruco Pose...")
    app = PUB_ECM_ARUCO_CP()
    zed_rect_left_msg = rospy.Subscriber("/zedm/zed_node/left/image_rect_color",Image,app.callback)
    rospy.spin()
