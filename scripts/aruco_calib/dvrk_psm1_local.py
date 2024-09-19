#!/usr/bin/env python3

import numpy as np

import rospy
from geometry_msgs.msg import TransformStamped

from utils import aruco_utils, tf_utils

class PUB_PSM1_LOCAL():
    def __init__(self):
        self.g_psm1base_ecmtip = self.preprocess_tf_data()
        self.pub_psm1_local = rospy.Publisher("/PSM1/local_setpoint_cp",TransformStamped,queue_size=10)

    def preprocess_tf_data(self):
        tf_data = aruco_utils.load_tf_data('/home/leo/aruco_data/base_tfs.yaml')
        g_psm1base_ecmtip    = np.array(tf_data["g_psm1base_ecmtip"])
        return g_psm1base_ecmtip

    def callback(self,psm1_cp):
        g_ecm_psm1tip = tf_utils.tfstamped2g(psm1_cp)
        new_psm1_cp = tf_utils.g2tfstamped(self.g_psm1base_ecmtip.dot(g_ecm_psm1tip),
                                            psm1_cp.header.stamp,"PSM1_base","PSM1")
        self.pub_psm1_local.publish(new_psm1_cp)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node("dvrk_psm1_local")
    rospy.loginfo("Publish PSM1 local cp")
    app = PUB_PSM1_LOCAL()
    psm1_cp = rospy.Subscriber("/PSM1/setpoint_cp",TransformStamped,app.callback)
    try:
        app.run()
    except Exception as e:
        print(e)
        rospy.loginfo("Shutdown...")
