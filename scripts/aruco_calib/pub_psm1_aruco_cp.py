#!/usr/bin/env python3

import cv2
import numpy as np

import rospy
import message_filters
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge

from utils import tf_utils, aruco_utils

class PUB_PSM1_ARUCO_CP():
    def __init__(self):
        self.br = CvBridge()
        self.psm1tip_marker_id  = 0
        self.psm1tip_marker_len = 0.01
        self.aruco_detector = aruco_utils.load_aruco_detector()
        self.g_ecm_dvrk_cv2, self.g_psm1tip_aruco_dvrk = aruco_utils.load_default_tfs("PSM1")
        zed_rect_caminfo  = rospy.wait_for_message("/zedm/zed_node/left/camera_info",CameraInfo)
        self.zed_mtx, self.zed_dist = aruco_utils.load_caminfo(zed_rect_caminfo)
        self.g_psm1base_zed = self.preprocess_tf_data()
        self.zed_psm1tip_rvecs  = np.array([])
        self.zed_psm1tip_tvecs  = np.array([])
        self.N = 10
        self.pub_psm1_aruco_cp = rospy.Publisher("/PSM1/ARUCO/setpoint_cp",TransformStamped,queue_size=10)

    def preprocess_tf_data(self):
        tf_data = aruco_utils.load_tf_data('/home/leo/aruco_data/base_tfs.yaml')
        g_psm1base_zed = np.array(tf_data["g_psm1base_zed"])
        return g_psm1base_zed

    def callback(self,zed_rect_left_msg,psm1_js):
        if len(self.zed_psm1tip_rvecs) == self.N:
            zed_psm1tip_rvecs_avg = aruco_utils.avg_aruco_rvecs(self.zed_psm1tip_rvecs)
            zed_psm1tip_tvecs_avg = np.mean(self.zed_psm1tip_tvecs,axis=0)
            g_zed_psm1tip         = tf_utils.cv2vecs2g(zed_psm1tip_rvecs_avg,zed_psm1tip_tvecs_avg)
            g_psm1base_psm1tip = self.g_psm1base_zed.dot(g_zed_psm1tip)
            psm1base_psm1tip_zed = tf_utils.g2tfstamped(g_psm1base_psm1tip,rospy.Time.now(),"PSM1_base","PSM1_ARUCO")
            self.pub_psm1_aruco_cp.publish(psm1base_psm1tip_zed)
            self.count = 0
            self.zed_psm1tip_rvecs  = np.array([])
            self.zed_psm1tip_tvecs  = np.array([])
            try:
                rospy.wait_for_message("psm1_moved_flag",Bool,timeout=10)
            except:
                pass
        else:
            psm1_th6 = np.array(psm1_js.position)[5]
            R_th6 = cv2.Rodrigues(np.array([1,0,0])*(-psm1_th6))[0]
            zed_rect_left = self.br.imgmsg_to_cv2(zed_rect_left_msg)[:,:,:3].copy()
            zed_corners, zed_ids, _ = self.aruco_detector.detectMarkers(zed_rect_left)
            if zed_ids is not None:
                for i,idx in enumerate(zed_ids):
                    if idx == self.psm1tip_marker_id:
                        zed_psm1tip_rvec, zed_psm1tip_tvec, _ = cv2.aruco.estimatePoseSingleMarkers(zed_corners[i],self.psm1tip_marker_len,self.zed_mtx,self.zed_dist)
                        zed_psm1tip_rvec = cv2.Rodrigues(cv2.Rodrigues(zed_psm1tip_rvec)[0].dot(self.g_psm1tip_aruco_dvrk[:3,:3]).dot(R_th6))[0]
                        if not self.zed_psm1tip_rvecs.any():
                            self.zed_psm1tip_rvecs = np.hstack((self.zed_psm1tip_rvecs,zed_psm1tip_rvec.reshape(-1)))
                            self.zed_psm1tip_tvecs = np.hstack((self.zed_psm1tip_tvecs,zed_psm1tip_tvec.reshape(-1)))
                        else:
                            self.zed_psm1tip_rvecs = np.vstack((self.zed_psm1tip_rvecs,zed_psm1tip_rvec.reshape(-1)))
                            self.zed_psm1tip_tvecs = np.vstack((self.zed_psm1tip_tvecs,zed_psm1tip_tvec.reshape(-1)))

if __name__ == "__main__":
    rospy.init_node("pub_psm1_aruco_cp")
    rospy.loginfo("Start Publishing PSM1 Tooltip Aruco Pose...")
    app = PUB_PSM1_ARUCO_CP()
    zed_rect_left_msg = message_filters.Subscriber("/zedm/zed_node/left/image_rect_color",Image)
    psm1_js = message_filters.Subscriber("/PSM1/setpoint_js",JointState)
    ts = message_filters.ApproximateTimeSynchronizer([zed_rect_left_msg,psm1_js], queue_size=10, slop=0.05)
    ts.registerCallback(app.callback)
    rospy.spin()
