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

class RECORD_PSM1_CALIB_TEST():
    def __init__(self,robot_name, expected_interval):
        self.record_flag = False
        self.bag = rosbag.Bag("/home/leo/aruco_data/psm1_calib_test_temp.bag","w")
        self.psm1_app = dvrk_utils.DVRK_CTRL(robot_name,expected_interval)
        self.g_ecm_dvrk_cv2 = aruco_utils.load_default_tfs("PSM1")[0]
        self.g_psm1base_ecmtip = self.preprocess_tf_data()
        self.load_params()
        self.pub_move_flag = rospy.Publisher("psm1_moved_flag",Bool,queue_size=10)

    def preprocess_tf_data(self):
        tf_data = aruco_utils.load_tf_data('/home/leo/aruco_data/base_tfs.yaml')
        # Deprecated...
        # g_ecm_odom      = np.array(tf_data["g_ecm_odom"])
        # g_odom_psm1base = np.array(tf_data["g_odom_psm1base"])
        # g_ecm_psm1base  = self.g_ecm_dvrk_cv2.dot(g_ecm_odom).dot(g_odom_psm1base)
        # g_psm1base_ecm  = tf_utils.ginv(g_ecm_psm1base)
        g_psm1base_ecmtip    = np.array(tf_data["g_psm1base_ecmtip"])
        return g_psm1base_ecmtip
    
    def load_params(self):
        arm_calib_path   = "/home/leo/aruco_data/"
        self.psm1_params = ik_utils.get_arm_calib_data(arm_calib_path+"psm1_calib_results.mat")

    def callback(self,psm1_aruco_cp,psm1_cp,psm1_js):
        if self.record_flag is False:
            g_ecm_psm1tip = tf_utils.tfstamped2g(psm1_cp)
            new_psm1_cp = tf_utils.g2tfstamped(self.g_psm1base_ecmtip.dot(g_ecm_psm1tip),
                                               psm1_cp.header.stamp,"PSM1_base","PSM1")
            g_psm1base_psm1tip = ik_utils.get_tip_poe(np.array(psm1_js.position),self.psm1_params)
            psm1_custom_cp = tf_utils.g2tfstamped(g_psm1base_psm1tip,psm1_cp.header.stamp,"PSM1_base","PSM1")
            self.bag.write("/PSM1/ARUCO/setpoint_cp",psm1_aruco_cp)
            self.bag.write("/PSM1/custom/setpoint_cp",psm1_custom_cp)
            self.bag.write("/PSM1/setpoint_cp",new_psm1_cp)
            self.bag.write("/PSM1/setpoint_js",psm1_js)
            self.record_flag = True

    def move_random_jp(self,duration):
        initial_jp = np.copy(self.psm1_app.get_jp())
        goal_jp    = np.copy(initial_jp)
        goal_jp[0] = goal_jp[0] + math.radians(random.randint(-40,40))
        goal_jp[1] = goal_jp[1] + math.radians(random.randint(-40,40))
        goal_jp[2] = goal_jp[2] + math.radians(random.randint(-5,5))
        goal_jp[3] = goal_jp[3] + math.radians(random.randint(-30,30))
        if goal_jp[1] > 0:
            goal_jp[4] = goal_jp[4] + math.radians(random.randint(-30,0))
        else:
            goal_jp[4] = goal_jp[4] + math.radians(random.randint(0,30))
        goal_jp[5] = goal_jp[5] + math.radians(random.randint(-30,30))
        samples = duration / self.psm1_app.expected_interval
        amplitude = (goal_jp-initial_jp)/samples
        for k in range(int(samples)):
            while not rospy.is_shutdown():
                if self.record_flag is True:
                    cur_goal = initial_jp + k*amplitude
                    self.psm1_app.arm.servo_jp(cur_goal)
                    rospy.sleep(self.psm1_app.expected_interval)
                    self.record_flag = False
                    self.pub_move_flag.publish(True)
                    break

    def run(self):
        N = 5
        self.psm1_app.home_test_position(5)
        for i in range(N):
            self.move_random_jp(1)
            dvrk_utils.print_id("Finished {}/{}".format(i+1,N))
            self.psm1_app.home_test_position(5)
            self.pub_move_flag.publish(True)
        self.bag.close()

if __name__ == "__main__":
    rospy.init_node("record_psm1_calib_test",disable_signals=True)
    rospy.loginfo("Start Recording PSM1 Tooltip Aruco Pose...")
    app = RECORD_PSM1_CALIB_TEST("PSM1",0.01)
    psm1_aruco_cp = message_filters.Subscriber("/PSM1/ARUCO/setpoint_cp",TransformStamped)
    psm1_cp       = message_filters.Subscriber("/PSM1/setpoint_cp",TransformStamped)
    psm1_js       = message_filters.Subscriber("/PSM1/setpoint_js",JointState)
    ts            = message_filters.ApproximateTimeSynchronizer([psm1_aruco_cp,psm1_cp,psm1_js], queue_size=10, slop=0.05)
    ts.registerCallback(app.callback)
    try:
        app.run()
    except Exception as e:
        print(e)
        dvrk_utils.print_id("Shutdown...")
        app.bag.close()
