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

class RECORD_PSM2_CALIB_TEST():
    def __init__(self,robot_name, expected_interval):
        self.bag = rosbag.Bag("/home/leo/aruco_data/psm2_calib_test_temp.bag","w")
        self.psm2_app = dvrk_utils.DVRK_CTRL(robot_name,expected_interval)
        self.g_ecm_dvrk_cv2 = aruco_utils.load_default_tfs("PSM2")[0]
        self.record_flag = False
        self.g_psm2base_ecmtip = self.preprocess_tf_data()
        self.load_params()
        self.pub_move_flag = rospy.Publisher("psm2_moved_flag",Bool,queue_size=10)

    def preprocess_tf_data(self):
        tf_data = aruco_utils.load_tf_data('/home/leo/aruco_data/base_tfs.yaml')
        # Deprecated...
        # g_ecm_odom      = np.array(tf_data["g_ecm_odom"])
        # g_odom_psm2base = np.array(tf_data["g_odom_psm2base"])
        # g_ecm_psm2base  = self.g_ecm_dvrk_cv2.dot(g_ecm_odom).dot(g_odom_psm2base)
        # g_psm2base_ecm  = tf_utils.ginv(g_ecm_psm2base)
        g_psm2base_ecmtip    = np.array(tf_data["g_psm2base_ecmtip"])
        return g_psm2base_ecmtip
    
    def load_params(self):
        arm_calib_path   = "/home/leo/aruco_data/"
        self.psm2_params = ik_utils.get_arm_calib_data(arm_calib_path+"psm2_calib_results.mat")

    def callback(self,psm2_aruco_cp,psm2_cp,psm2_js):
        if self.record_flag is False:
            g_ecm_psm2tip = tf_utils.tfstamped2g(psm2_cp)
            new_psm2_cp = tf_utils.g2tfstamped(self.g_psm2base_ecmtip.dot(g_ecm_psm2tip),
                                               psm2_cp.header.stamp,"PSM2_base","PSM2")
            g_psm2base_psm2tip = ik_utils.get_tip_poe(np.array(psm2_js.position),self.psm2_params)
            psm2_custom_cp = tf_utils.g2tfstamped(g_psm2base_psm2tip,psm2_cp.header.stamp,"PSM2_base","PSM2")
            self.bag.write("/PSM2/ARUCO/setpoint_cp",psm2_aruco_cp)
            self.bag.write("/PSM2/custom/setpoint_cp",psm2_custom_cp)
            self.bag.write("/PSM2/setpoint_cp",new_psm2_cp)
            self.bag.write("/PSM2/setpoint_js",psm2_js)
            self.record_flag = True

    def move_random_jp(self,duration):
        initial_jp = np.copy(self.psm2_app.get_jp())
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
        samples = duration / self.psm2_app.expected_interval
        amplitude = (goal_jp-initial_jp)/samples
        for k in range(int(samples)):
            while not rospy.is_shutdown():
                if self.record_flag is True:
                    cur_goal = initial_jp + k*amplitude
                    self.psm2_app.arm.servo_jp(cur_goal)
                    rospy.sleep(self.psm2_app.expected_interval)
                    self.record_flag = False
                    self.pub_move_flag.publish(True)
                    break

    def run(self):
        N = 5
        self.psm2_app.home_test_position(5)
        for i in range(N):
            self.move_random_jp(1)
            dvrk_utils.print_id("Finished {}/{}".format(i+1,N))
            self.psm2_app.home_test_position(5)
            self.pub_move_flag.publish(True)
        self.bag.close()

if __name__ == "__main__":
    rospy.init_node("record_psm2_calib_test",disable_signals=True)
    rospy.loginfo("Start Recording PSM2 Tooltip Aruco Pose...")
    app = RECORD_PSM2_CALIB_TEST("PSM2",0.01)
    psm2_aruco_cp = message_filters.Subscriber("/PSM2/ARUCO/setpoint_cp",TransformStamped)
    psm2_cp       = message_filters.Subscriber("/PSM2/setpoint_cp",TransformStamped)
    psm2_js       = message_filters.Subscriber("/PSM2/setpoint_js",JointState)
    ts            = message_filters.ApproximateTimeSynchronizer([psm2_aruco_cp,psm2_cp,psm2_js], queue_size=10, slop=0.05)
    ts.registerCallback(app.callback)
    try:
        app.run()
    except Exception as e:
        print(e)
        dvrk_utils.print_id("Shutdown...")
        app.bag.close()
