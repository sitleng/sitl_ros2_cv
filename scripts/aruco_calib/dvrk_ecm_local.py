#!/usr/bin/env python3

import numpy as np

import rospy
from geometry_msgs.msg import TransformStamped

from utils import dvrk_utils, aruco_utils, tf_utils

class PUB_ECM_LOCAL():
    def __init__(self):
        self.g_ecmbase_psm1tip = self.preprocess_tf_data()
        self.pub_ecm_local = rospy.Publisher("/ECM/local_setpoint_cp",TransformStamped,queue_size=10)

    def preprocess_tf_data(self):
        tf_data = aruco_utils.load_tf_data('/home/leo/aruco_data/base_tfs.yaml')
        g_psmtip_aruco_dvrk = aruco_utils.load_default_tfs("ECM")[1]
        g_ecmbase_psm1tip    = np.array(tf_data["g_ecmbase_psm1tip"]).dot(g_psmtip_aruco_dvrk)
        return g_ecmbase_psm1tip

    def callback(self,psm1_cp):
        g_psm1tip_ecm = tf_utils.tfstamped2ginv(psm1_cp)
        new_ecm_cp = tf_utils.g2tfstamped(self.g_ecmbase_psm1tip.dot(g_psm1tip_ecm),
                                            psm1_cp.header.stamp,"ECM_base","ECM")
        self.pub_ecm_local.publish(new_ecm_cp)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node("dvrk_ecm_local")
    rospy.loginfo("Publish ECM local cp")
    app = PUB_ECM_LOCAL()
    psm1_cp = rospy.Subscriber("/PSM1/setpoint_cp",TransformStamped,app.callback)
    try:
        app.run()
    except Exception as e:
        print(e)
        rospy.loginfo("Shutdown...")
        