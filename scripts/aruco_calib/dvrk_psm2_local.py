#!/usr/bin/env python3

import numpy as np

import rospy
from geometry_msgs.msg import TransformStamped

from utils import aruco_utils, tf_utils

class PUB_PSM2_LOCAL():
    def __init__(self):
        self.g_psm2base_ecmtip = self.preprocess_tf_data()
        self.pub_psm2_local = rospy.Publisher("/PSM2/local_setpoint_cp",TransformStamped,queue_size=10)

    def preprocess_tf_data(self):
        tf_data = aruco_utils.load_tf_data('/home/leo/aruco_data/base_tfs.yaml')
        g_psm2base_ecmtip    = np.array(tf_data["g_psm2base_ecmtip"])
        return g_psm2base_ecmtip

    def callback(self,psm2_cp):
        g_ecm_psm2tip = tf_utils.tfstamped2g(psm2_cp)
        new_psm2_cp   = tf_utils.g2tfstamped(self.g_psm2base_ecmtip.dot(g_ecm_psm2tip),
                                             psm2_cp.header.stamp,"PSM2_base","PSM2")
        self.pub_psm2_local.publish(new_psm2_cp)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node("dvrk_psm2_local")
    rospy.loginfo("Publish PSM2 local cp")
    app     = PUB_PSM2_LOCAL()
    psm2_cp = rospy.Subscriber("/PSM2/setpoint_cp",TransformStamped,app.callback)
    try:
        app.run()
    except Exception as e:
        print(e)
        rospy.loginfo("Shutdown...")
