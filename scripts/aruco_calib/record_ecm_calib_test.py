#!/usr/bin/env python3

import math
import numpy as np
import random

import rospy
import rosbag
import message_filters
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped

from utils import dvrk_utils, aruco_utils, tf_utils, ik_utils

class RECORD_ECM_CALIB_TEST():
    def __init__(self,robot_name, expected_interval):
        self.record_flag = False
        self.bag = rosbag.Bag("/home/leo/aruco_data/ecm_calib_test_temp.bag","w")
        self.ecm_app = dvrk_utils.DVRK_CTRL(robot_name,expected_interval)
        self.g_ecmbase_psm1tip = self.preprocess_tf_data()
        self.load_params()
        self.pub_move_flag = rospy.Publisher("ecm_moved_flag",Bool,queue_size=10)

    def preprocess_tf_data(self):
        tf_data = aruco_utils.load_tf_data('/home/leo/aruco_data/base_tfs.yaml')
        g_psmtip_aruco_dvrk = aruco_utils.load_default_tfs("ECM")[1]
        g_ecmbase_psm1tip = np.array(tf_data["g_ecmbase_psm1tip"])
        g_ecmbase_psm1tip = g_ecmbase_psm1tip.dot(g_psmtip_aruco_dvrk)
        return g_ecmbase_psm1tip
    
    def load_params(self):
        arm_calib_path   = "/home/leo/aruco_data/"
        self.ecm_params = ik_utils.get_arm_calib_data(arm_calib_path+"ecm_calib_results.mat")

    def callback(self,ecm_aruco_cp,psm1_cp,ecm_js):
        if self.record_flag is False:
            g_psm1tip_ecm = tf_utils.tfstamped2ginv(psm1_cp)
            new_ecm_cp = tf_utils.g2tfstamped(self.g_ecmbase_psm1tip.dot(g_psm1tip_ecm),
                                             psm1_cp.header.stamp,"ECM_base","ECM")
            g_ecmbase_ecmtip = ik_utils.get_tip_poe(np.array(ecm_js.position),self.ecm_params)
            ecm_custom_cp = tf_utils.g2tfstamped(g_ecmbase_ecmtip,psm1_cp.header.stamp,"ECM_base","ECM")
            self.bag.write("/ECM/ARUCO/setpoint_cp",ecm_aruco_cp)
            self.bag.write("/ECM/custom/setpoint_cp",ecm_custom_cp)
            self.bag.write("/ECM/setpoint_cp",new_ecm_cp)
            self.bag.write("/ECM/setpoint_js",ecm_js)
            self.record_flag = True

    def move_random_jp(self,duration):
        initial_jp = np.copy(self.ecm_app.get_jp())
        goal_jp    = np.copy(initial_jp)
        goal_jp[0] = goal_jp[0] + math.radians(random.randint(-40,40))
        goal_jp[1] = goal_jp[1] + math.radians(random.randint(-40,40))
        goal_jp[2] = goal_jp[2] + math.radians(random.randint(-5,5))
        goal_jp[3] = goal_jp[3] + math.radians(random.randint(-45,45))
        samples = duration / self.ecm_app.expected_interval
        amplitude = (goal_jp-initial_jp)/samples
        for k in range(int(samples)):
            while not rospy.is_shutdown():
                if self.record_flag is True:
                    cur_goal = initial_jp + k*amplitude
                    self.ecm_app.arm.servo_jp(cur_goal)
                    rospy.sleep(self.ecm_app.expected_interval)
                    self.record_flag = False
                    self.pub_move_flag.publish(True)
                    break

    def run(self):
        N = 5
        self.ecm_app.home_test_position(5)
        for i in range(N):
            self.move_random_jp(1)
            dvrk_utils.print_id("Finished {}/{}".format(i+1,N))
            self.ecm_app.home_test_position(5)
            self.pub_move_flag.publish(True)
        self.bag.close()

if __name__ == "__main__":
    rospy.init_node("record_ecm_calib",disable_signals=True)
    rospy.loginfo("Start Publishing ECM Tooltip Aruco Pose...")
    app = RECORD_ECM_CALIB_TEST("ECM",0.01)
    ecm_aruco_cp = message_filters.Subscriber("/ECM/ARUCO/setpoint_cp",TransformStamped)
    psm1_cp  = message_filters.Subscriber("/PSM1/setpoint_cp",TransformStamped)
    ecm_js = message_filters.Subscriber("/ECM/setpoint_js",JointState)
    ts = message_filters.ApproximateTimeSynchronizer([ecm_aruco_cp,psm1_cp,ecm_js], queue_size=10, slop=0.05)
    ts.registerCallback(app.callback)
    try:
        app.run()
    except Exception as e:
        print(e)
        dvrk_utils.print_id("Shutdown...")
        app.bag.close()
        