#!/usr/bin/env python3

import math
import numpy as np

import rospy
import rosbag
import message_filters
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped

from utils import dvrk_utils, aruco_utils, tf_utils

class RECORD_PSM2_CALIB():
    def __init__(self,robot_name, expected_interval):
        self.bag = rosbag.Bag("/home/leo/aruco_data/psm2_calib_data_new.bag","w")
        self.psm2_app = dvrk_utils.DVRK_CTRL(robot_name,expected_interval)
        self.g_ecm_dvrk_cv2 = aruco_utils.load_default_tfs("PSM2")[0]
        self.record_flag = False
        self.g_psm2base_ecmtip = self.preprocess_tf_data()
        self.goals = np.array([[math.radians(90),math.radians(-90)],
                               [math.radians(45),math.radians(-45)],
                               [           0.240,                0],
                               [math.radians(60),math.radians(-60)],
                               [math.radians(50),math.radians(-30)],
                               [math.radians(80),math.radians(-80)]])
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

    def callback(self,psm2_aruco_cp,psm2_cp,psm2_js):
        if self.record_flag is False:
            g_ecm_psm2tip = tf_utils.tfstamped2g(psm2_cp)
            new_psm2_cp = tf_utils.g2tfstamped(self.g_psm2base_ecmtip.dot(g_ecm_psm2tip),
                                               psm2_cp.header.stamp,"PSM2_base","PSM2")
            self.bag.write("/PSM2/ARUCO/setpoint_cp",psm2_aruco_cp)
            self.bag.write("/PSM2/setpoint_cp",new_psm2_cp)
            self.bag.write("/PSM2/setpoint_js",psm2_js)
            self.record_flag = True

    def move_joint(self,joint,joint_goal,duration):
        initial_jp = np.copy(self.psm2_app.get_jp())
        goal_jp    = np.zeros_like(initial_jp)
        goal_jp[joint] = joint_goal
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
        self.psm2_app.home_zero_position(5)
        for joint, goal in enumerate(self.goals):
            if joint == 2:
                max_duration = 1
                min_duration = 1
            elif joint < 2:
                max_duration = 1
                min_duration = 2
            else:
                max_duration = 0.5
                min_duration = 1
            max_goal = goal[0]
            min_goal = goal[1]
            # Move to max goal
            self.move_joint(joint,max_goal,max_duration)
            dvrk_utils.print_id("Finished joint {}, {}".format(joint,"max"))
            # Move to min goal
            self.move_joint(joint,min_goal,min_duration)
            dvrk_utils.print_id("Finished joint {}, {}".format(joint,"min"))
            if joint != 2:
                # Move to zero position
                self.move_joint(joint,0,min_duration)
                dvrk_utils.print_id("Going back to zero position")
            self.pub_move_flag.publish(True)
        self.bag.close()

if __name__ == "__main__":
    rospy.init_node("record_psm2_calib",disable_signals=True)
    rospy.loginfo("Start Publishing PSM2 Tooltip Aruco Pose...")
    app = RECORD_PSM2_CALIB("PSM2",0.01)
    psm2_aruco_cp = message_filters.Subscriber("/PSM2/ARUCO/setpoint_cp",TransformStamped)
    psm2_cp  = message_filters.Subscriber("/PSM2/setpoint_cp",TransformStamped)
    psm2_js = message_filters.Subscriber("/PSM2/setpoint_js",JointState)
    ts = message_filters.ApproximateTimeSynchronizer([psm2_aruco_cp,psm2_cp,psm2_js], queue_size=10, slop=0.05)
    ts.registerCallback(app.callback)
    try:
        app.run()
    except Exception as e:
        print(e)
        dvrk_utils.print_id("Shutdown...")
        app.bag.close()
