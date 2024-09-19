#!/usr/bin/env python3

import numpy as np
import math
import os
import tf
import rospy
from sensor_msgs.msg import JointState
from utils import tf_utils, aruco_utils, ik_utils

class PUB_CUSTOM_DVRK_TF():
    def __init__(self,node_name,params):
        self.node_name = node_name
        self.params    = params
        self.load_tfs()
        self.load_params()
        self.br = tf.TransformBroadcaster()

    def __del__(self):
        print("Shutting down...")

    def load_tfs(self):
        tf_path = "/home/" + os.getlogin() + "/aruco_data/base_tfs.yaml"
        tf_data = aruco_utils.load_tf_data(tf_path)
        g_odom_psm1base = np.array(tf_data["g_odom_psm1base"])
        g_odom_psm2base = np.array(tf_data["g_odom_psm2base"])
        g_odom_ecmbase  = np.array(tf_data["g_odom_ecmbase"])
        g_odom_zed      = np.array(tf_data["g_odom_zed"])
        g_map_odom      = tf_utils.cv2vecs2g(np.array([0,0,1])*math.radians(90),np.array([0,0,1])).dot(
            tf_utils.cv2vecs2g(np.array([1,0,0])*math.radians(90),np.array([0,0,0])))
        # g_psm1tip_psm1jaw = tf_utils.cv2vecs2g(np.array([0.0,0.0,0.0]),np.array([0.002,-0.0035,0.015]))
        g_psm1tip_psm1jaw = tf_utils.cv2vecs2g(np.array([0.0,0.0,0.0]),np.array([-0.005,-0.0025,0.0147]))
        g_psm2tip_psm2jaw = tf_utils.cv2vecs2g(np.array([0.0,0.0,0.0]),np.array([-0.004, 0.0, 0.019]))
        if self.params["cam_type"] == "30":
            self.g_ecm_dvrk  = tf_utils.cv2vecs2g(np.array([1,0,0])*math.radians(30),np.array([0,0,0]))
        elif self.params["cam_type"] == "0":
            self.g_ecm_dvrk  = tf_utils.cv2vecs2g(np.array([0.0,0.0,0.0]),np.array([0,0,0]))
        self.g_ecmdvrk_ecmopencv = np.array(tf_data["g_ecmdvrk_ecmopencv"])
        # self.g_ecmdvrk_ecmopencv = tf_utils.cv2vecs2g(np.array([0.0,0.0,1.0])*math.radians(180),np.array([0,0,0]))
        self.zed_test_tf         = tf_utils.g2tf(np.array(tf_data["g_zed_test"]))
        # self.g_test_ecmcv2       = tf_utils.g2tf(np.array(tf_data["g_test_ecmcv2"]))
        # self.psm1base_ecm_tf     = tf_utils.g2tf(np.array(tf_data["g_psm1base_ecm"]))
        # self.psm2base_ecm_tf     = tf_utils.g2tf(np.array(tf_data["g_psm2base_ecm"]))
        self.odom_psm1base_tf    = tf_utils.g2tf(g_odom_psm1base)
        self.odom_psm2base_tf    = tf_utils.g2tf(g_odom_psm2base)
        self.odom_ecmbase_tf     = tf_utils.g2tf(g_odom_ecmbase)
        self.odom_zed_tf         = tf_utils.g2tf(g_odom_zed)
        self.map_odom_tf         = tf_utils.g2tf(g_map_odom)
        self.psm1tip_psm1jaw_tf  = tf_utils.g2tf(g_psm1tip_psm1jaw)
        self.psm2tip_psm2jaw_tf  = tf_utils.g2tf(g_psm2tip_psm2jaw)
        # self.map_cart_tf         = tf_utils.g2tf(
        #     tf_utils.cv2vecs2g(np.array([0,0,1])*math.radians(90),np.array([0,0,0])).dot(
        #     tf_utils.cv2vecs2g(np.array([1,0,0])*math.radians(90),np.array([0,0,0])))
        # )
    
    def load_params(self):
        self.psm1_params = ik_utils.get_arm_calib_data(self.params["psm1_calib_fn"])
        self.psm2_params = ik_utils.get_arm_calib_data(self.params["psm2_calib_fn"])
        self.ecm_params  = ik_utils.get_arm_calib_data(self.params["ecm_calib_fn"])

    def psm1_callback(self,psm1_js):
        t = rospy.Time.now()
        self.br.sendTransform((self.odom_psm1base_tf.translation.x,self.odom_psm1base_tf.translation.y,self.odom_psm1base_tf.translation.z),
                              (self.odom_psm1base_tf.rotation.x,self.odom_psm1base_tf.rotation.y,self.odom_psm1base_tf.rotation.z,self.odom_psm1base_tf.rotation.w),
                              t,"psm1_base","odom")
        psm1_jp = np.array(psm1_js.position)
        g_psm1base_psm1tip = ik_utils.get_tip_pose(psm1_jp,self.psm1_params)
        psm1base_psm1tip_tf = tf_utils.g2tf(g_psm1base_psm1tip)
        self.br.sendTransform((psm1base_psm1tip_tf.translation.x,psm1base_psm1tip_tf.translation.y,psm1base_psm1tip_tf.translation.z),
                              (psm1base_psm1tip_tf.rotation.x,psm1base_psm1tip_tf.rotation.y,psm1base_psm1tip_tf.rotation.z,psm1base_psm1tip_tf.rotation.w),
                              t,"psm1_tip","psm1_base")
        self.br.sendTransform((self.psm1tip_psm1jaw_tf.translation.x,self.psm1tip_psm1jaw_tf.translation.y,self.psm1tip_psm1jaw_tf.translation.z),
                              (self.psm1tip_psm1jaw_tf.rotation.x,self.psm1tip_psm1jaw_tf.rotation.y,self.psm1tip_psm1jaw_tf.rotation.z,self.psm1tip_psm1jaw_tf.rotation.w),
                              t,"psm1_jaw","psm1_tip")
    
    def psm2_callback(self,psm2_js):
        t = rospy.Time.now()
        self.br.sendTransform((self.odom_psm2base_tf.translation.x,self.odom_psm2base_tf.translation.y,self.odom_psm2base_tf.translation.z),
                              (self.odom_psm2base_tf.rotation.x,self.odom_psm2base_tf.rotation.y,self.odom_psm2base_tf.rotation.z,self.odom_psm2base_tf.rotation.w),
                              t,"psm2_base","odom")
        psm2_jp = np.array(psm2_js.position)
        g_psm2base_psm2tip = ik_utils.get_tip_pose(psm2_jp,self.psm2_params)
        psm2base_psm2tip_tf = tf_utils.g2tf(g_psm2base_psm2tip)
        self.br.sendTransform((psm2base_psm2tip_tf.translation.x,psm2base_psm2tip_tf.translation.y,psm2base_psm2tip_tf.translation.z),
                              (psm2base_psm2tip_tf.rotation.x,psm2base_psm2tip_tf.rotation.y,psm2base_psm2tip_tf.rotation.z,psm2base_psm2tip_tf.rotation.w),
                              t,"psm2_tip","psm2_base")
        self.br.sendTransform((self.psm2tip_psm2jaw_tf.translation.x,self.psm2tip_psm2jaw_tf.translation.y,self.psm2tip_psm2jaw_tf.translation.z),
                              (self.psm2tip_psm2jaw_tf.rotation.x,self.psm2tip_psm2jaw_tf.rotation.y,self.psm2tip_psm2jaw_tf.rotation.z,self.psm2tip_psm2jaw_tf.rotation.w),
                              t,"psm2_jaw","psm2_tip")
        
    def ecm_callback(self,ecm_js):
        t = rospy.Time.now()
        self.br.sendTransform((self.map_odom_tf.translation.x,self.map_odom_tf.translation.y,self.map_odom_tf.translation.z),
                              (self.map_odom_tf.rotation.x,self.map_odom_tf.rotation.y,self.map_odom_tf.rotation.z,self.map_odom_tf.rotation.w),
                              t,"odom","map")
        # self.br.sendTransform((self.map_cart_tf.translation.x,self.map_cart_tf.translation.y,self.map_cart_tf.translation.z),
        #                       (self.map_cart_tf.rotation.x,self.map_cart_tf.rotation.y,self.map_cart_tf.rotation.z,self.map_cart_tf.rotation.w),
        #                       t,"Cart","map")
        self.br.sendTransform((self.odom_zed_tf.translation.x,self.odom_zed_tf.translation.y,self.odom_zed_tf.translation.z),
                              (self.odom_zed_tf.rotation.x,self.odom_zed_tf.rotation.y,self.odom_zed_tf.rotation.z,self.odom_zed_tf.rotation.w),
                              t,"base_link","odom")
        self.br.sendTransform((self.odom_ecmbase_tf.translation.x,self.odom_ecmbase_tf.translation.y,self.odom_ecmbase_tf.translation.z),
                              (self.odom_ecmbase_tf.rotation.x,self.odom_ecmbase_tf.rotation.y,self.odom_ecmbase_tf.rotation.z,self.odom_ecmbase_tf.rotation.w),
                              t,"ecm_base","odom")
        self.br.sendTransform((self.zed_test_tf.translation.x,self.zed_test_tf.translation.y,self.zed_test_tf.translation.z),
                              (self.zed_test_tf.rotation.x,self.zed_test_tf.rotation.y,self.zed_test_tf.rotation.z,self.zed_test_tf.rotation.w),
                              t,"test","base_link")
        # self.br.sendTransform((self.g_test_ecmcv2.translation.x,self.g_test_ecmcv2.translation.y,self.g_test_ecmcv2.translation.z),
        #                       (self.g_test_ecmcv2.rotation.x,self.g_test_ecmcv2.rotation.y,self.g_test_ecmcv2.rotation.z,self.g_test_ecmcv2.rotation.w),
        #                       t,"ecm_left","test")
        ecm_jp  = np.array(ecm_js.position)
        g_ecmbase_ecmdvrk    = ik_utils.get_tip_pose(ecm_jp,self.ecm_params).dot(self.g_ecm_dvrk)
        ecmbase_ecmdvrk_tf   = tf_utils.g2tf(g_ecmbase_ecmdvrk)
        ecmdvrk_ecmopencv_tf = tf_utils.g2tf(self.g_ecmdvrk_ecmopencv)
        self.br.sendTransform((ecmbase_ecmdvrk_tf.translation.x,ecmbase_ecmdvrk_tf.translation.y,ecmbase_ecmdvrk_tf.translation.z),
                              (ecmbase_ecmdvrk_tf.rotation.x,ecmbase_ecmdvrk_tf.rotation.y,ecmbase_ecmdvrk_tf.rotation.z,ecmbase_ecmdvrk_tf.rotation.w),
                              t,"ecm_tip","ecm_base")
        self.br.sendTransform((ecmdvrk_ecmopencv_tf.translation.x,ecmdvrk_ecmopencv_tf.translation.y,ecmdvrk_ecmopencv_tf.translation.z),
                              (ecmdvrk_ecmopencv_tf.rotation.x,ecmdvrk_ecmopencv_tf.rotation.y,ecmdvrk_ecmopencv_tf.rotation.z,ecmdvrk_ecmopencv_tf.rotation.w),
                              t,"ecm_left","ecm_tip")

if __name__ == '__main__':
    rospy.init_node("pub_custom_dvrk_tf")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)

    app = PUB_CUSTOM_DVRK_TF(node_name,params)

    try:
        psm1_js = rospy.Subscriber("/PSM1/setpoint_js",JointState,app.psm1_callback)
        psm2_js = rospy.Subscriber("/PSM2/setpoint_js",JointState,app.psm2_callback)
        ecm_js  = rospy.Subscriber("/ECM/setpoint_js",JointState,app.ecm_callback)
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)
    finally:
        del app
