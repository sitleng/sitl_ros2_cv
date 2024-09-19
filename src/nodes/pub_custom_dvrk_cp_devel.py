#!/usr/bin/env python3

import os
import numpy as np
import math

import rospy
import tf
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import message_filters

from utils import tf_utils, aruco_utils, ik_devel_utils



class PUB_CUSTOM_DVRK_CP():
    def __init__(self,node_name,params):
        self.node_name = node_name
        self.params    = params
        self.load_tfs()
        self.load_params()
        if self.params["cam_type"] == "30":
            self.g_ecm_dvrk  = tf_utils.cv2vecs2g(np.array([1,0,0])*math.radians(30),np.array([0,0,0]))
        elif self.params["cam_type"] == "0":
            self.g_ecm_dvrk  = np.eye(4)
        self.custom_psm1_cp = rospy.Publisher("/PSM1/custom/setpoint_cp",PoseStamped,queue_size=10)
        self.custom_psm2_cp = rospy.Publisher("/PSM2/custom/setpoint_cp",PoseStamped,queue_size=10)
        self.custom_psm1_jaw_cp = rospy.Publisher("/PSM1/custom/jaw/setpoint_cp",PoseStamped,queue_size=10)
        self.custom_psm2_jaw_cp = rospy.Publisher("/PSM2/custom/jaw/setpoint_cp",PoseStamped,queue_size=10)
        self.custom_ecm_cp = rospy.Publisher("/ECM/custom/setpoint_cp",PoseStamped,queue_size=10)
        self.custom_psm1_local_cp = rospy.Publisher("/PSM1/custom/local/setpoint_cp",PoseStamped,queue_size=10)
        self.custom_psm2_local_cp = rospy.Publisher("/PSM2/custom/local/setpoint_cp",PoseStamped,queue_size=10)
        self.custom_psm1_local_jaw_cp = rospy.Publisher("/PSM1/custom/local/jaw/setpoint_cp",PoseStamped,queue_size=10)
        self.custom_psm2_local_jaw_cp = rospy.Publisher("/PSM2/custom/local/jaw/setpoint_cp",PoseStamped,queue_size=10)
        self.custom_ecm_local_cp = rospy.Publisher("/ECM/custom/local/setpoint_cp",PoseStamped,queue_size=10)
    
    def __del__(self):
        print("Shutting down...")

    def load_tfs(self):
        tf_path = "/home/" + os.getlogin() + "/aruco_data/base_tfs.yaml"
        tf_data = aruco_utils.load_tf_data(tf_path)
        g_odom_psm1base = np.array(tf_data["g_odom_psm1base"])
        g_odom_psm2base = np.array(tf_data["g_odom_psm2base"])
        self.g_odom_ecmbase = np.array(tf_data["g_odom_ecmbase"])
        g_ecmbase_odom = tf_utils.ginv(self.g_odom_ecmbase)
        self.g_ecmbase_psm1base = g_ecmbase_odom.dot(g_odom_psm1base)
        self.g_ecmbase_psm2base = g_ecmbase_odom.dot(g_odom_psm2base)
        # Fix dataset with the correct transformation
        # self.g_psm1tip_psm1jaw = tf_utils.cv2vecs2g(np.array([0.0,0.0,0.0]),np.array([0.002,-0.0035,0.015]))
        # self.g_psm2tip_psm2jaw = tf_utils.cv2vecs2g(np.array([0.0,0.0,0.0]),np.array([0.002,-0.0035,0.015]))
        self.g_psm1tip_psm1jaw = tf_utils.cv2vecs2g(np.array([0.0,0.0,0.0]),np.array([-0.005,-0.0025,0.0147]))
        self.g_psm2tip_psm2jaw = tf_utils.cv2vecs2g(np.array([0.0,0.0,0.0]),np.array([-0.0039, 0.0, 0.02]))
    
    def load_params(self):
        self.psm1_params = ik_devel_utils.get_arm_calib_data(self.params["psm1_calib_fn"])
        self.psm2_params = ik_devel_utils.get_arm_calib_data(self.params["psm2_calib_fn"])
        self.ecm_params  = ik_devel_utils.get_arm_calib_data(self.params["ecm_calib_fn"])

    def callback(self,psm1_js,psm2_js,ecm_js):
        psm1_jp = np.array(psm1_js.position)
        g_psm1base_psm1tip = ik_devel_utils.get_tip_pose(psm1_jp,self.psm1_params)
        g_psm1base_psm1jaw = g_psm1base_psm1tip.dot(self.g_psm1tip_psm1jaw)
        psm2_jp = np.array(psm2_js.position)
        g_psm2base_psm2tip = ik_devel_utils.get_tip_pose(psm2_jp,self.psm2_params)
        g_psm2base_psm2jaw = g_psm2base_psm2tip.dot(self.g_psm2tip_psm2jaw)
        ecm_jp  = np.array(ecm_js.position)
        g_ecmbase_ecmtip = ik_devel_utils.get_tip_pose(ecm_jp,self.ecm_params).dot(self.g_ecm_dvrk)
        g_ecmtip_ecmbase = tf_utils.ginv(g_ecmbase_ecmtip)
        g_ecmtip_psm1tip = g_ecmtip_ecmbase.dot(self.g_ecmbase_psm1base).dot(g_psm1base_psm1tip)
        g_ecmtip_psm1jaw = g_ecmtip_psm1tip.dot(self.g_psm1tip_psm1jaw)
        g_ecmtip_psm2tip = g_ecmtip_ecmbase.dot(self.g_ecmbase_psm2base).dot(g_psm2base_psm2tip)
        g_ecmtip_psm2jaw = g_ecmtip_psm2tip.dot(self.g_psm2tip_psm2jaw)
        g_odom_ecmtip = self.g_odom_ecmbase.dot(g_ecmbase_ecmtip)
        t = rospy.Time.now()

        custom_ecm_cp_msg = tf_utils.g2posestamped(g_odom_ecmtip,t,"Cart")
        self.custom_ecm_cp.publish(custom_ecm_cp_msg)

        custom_psm1_cp_msg = tf_utils.g2posestamped(g_ecmtip_psm1tip,t,"ECM")
        self.custom_psm1_cp.publish(custom_psm1_cp_msg)

        custom_psm2_cp_msg = tf_utils.g2posestamped(g_ecmtip_psm2tip,t,"ECM")
        self.custom_psm2_cp.publish(custom_psm2_cp_msg)

        custom_psm1_jaw_cp_msg = tf_utils.g2posestamped(g_ecmtip_psm1jaw,t,"ECM")
        self.custom_psm1_jaw_cp.publish(custom_psm1_jaw_cp_msg)

        custom_psm2_jaw_cp_msg = tf_utils.g2posestamped(g_ecmtip_psm2jaw,t,"ECM")
        self.custom_psm2_jaw_cp.publish(custom_psm2_jaw_cp_msg)

        custom_psm1_local_cp_msg = tf_utils.g2posestamped(g_psm1base_psm1tip,t,"PSM1_base")
        self.custom_psm1_local_cp.publish(custom_psm1_local_cp_msg)

        custom_psm2_local_cp_msg = tf_utils.g2posestamped(g_psm2base_psm2tip,t,"PSM2_base")
        self.custom_psm2_local_cp.publish(custom_psm2_local_cp_msg)

        custom_ecm_local_cp_msg = tf_utils.g2posestamped(g_ecmbase_ecmtip,t,"ECM_base")
        self.custom_ecm_local_cp.publish(custom_ecm_local_cp_msg)

        custom_psm1_local_jaw_cp_msg = tf_utils.g2posestamped(g_psm1base_psm1jaw,t,"PSM1_base")
        self.custom_psm1_local_jaw_cp.publish(custom_psm1_local_jaw_cp_msg)

        custom_psm2_local_jaw_cp_msg = tf_utils.g2posestamped(g_psm2base_psm2jaw,t,"PSM2_base")
        self.custom_psm1_local_jaw_cp.publish(custom_psm2_local_jaw_cp_msg)

if __name__ == '__main__':
    rospy.init_node("pub_custom_dvrk_cp")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)

    app = PUB_CUSTOM_DVRK_CP(node_name,params)

    try:
        psm1_js = message_filters.Subscriber("/PSM1/setpoint_js",JointState)
        psm2_js = message_filters.Subscriber("/PSM2/setpoint_js",JointState)
        ecm_js  = message_filters.Subscriber("/ECM/setpoint_js",JointState)
        ts      = message_filters.ApproximateTimeSynchronizer([psm1_js,psm2_js,ecm_js],slop=0.01,queue_size=10)
        ts.registerCallback(app.callback)
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)
    finally:
        del app
