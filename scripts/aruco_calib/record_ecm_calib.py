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

class RECORD_ECM_CALIB():
    def __init__(self,robot_name, expected_interval):
        self.bag = rosbag.Bag("/home/leo/aruco_data/ecm_calib_data_new.bag","w")
        self.ecm_app = dvrk_utils.DVRK_CTRL(robot_name,expected_interval)
        self.record_flag = False
        self.g_ecmbase_psm1tip = self.preprocess_tf_data()
        self.goals = np.array([[math.radians(70),math.radians(-70)],
                               [math.radians(40),math.radians(-45)],
                               [            0.23,            0.03],
                               [math.radians(85),math.radians(-85)]])
        self.pub_move_flag = rospy.Publisher("ecm_moved_flag",Bool,queue_size=10)

    def preprocess_tf_data(self):
        tf_data = aruco_utils.load_tf_data('/home/leo/aruco_data/base_tfs.yaml')
        g_psmtip_aruco_dvrk = aruco_utils.load_default_tfs("ECM")[1]
        g_ecmbase_psm1tip = np.array(tf_data["g_ecmbase_psm1tip"])
        g_ecmbase_psm1tip = g_ecmbase_psm1tip.dot(g_psmtip_aruco_dvrk)
        return g_ecmbase_psm1tip

    def callback(self,ecm_aruco_cp,psm1_cp,ecm_js):
        if self.record_flag is False:
            g_psm1tip_ecm = tf_utils.tfstamped2ginv(psm1_cp)
            new_ecm_cp = tf_utils.g2tfstamped(self.g_ecmbase_psm1tip.dot(g_psm1tip_ecm),
                                              psm1_cp.header.stamp,"ECM_base","ECM")
            self.bag.write("/ECM/ARUCO/setpoint_cp",ecm_aruco_cp)
            self.bag.write("/ECM/setpoint_cp",new_ecm_cp)
            self.bag.write("/ECM/setpoint_js",ecm_js)
            # self.bag.write("/ECM/io/analog_input_pos_si",ecm_potentiometer_js)
            # self.bag.write("/ECM/io/joint_measured_js"  ,ecm_encoder_js)
            self.record_flag = True

    def move_joint(self,joint,joint_goal,duration):
        initial_jp = np.copy(self.ecm_app.get_jp())
        goal_jp    = np.zeros_like(initial_jp)
        if joint != 2:
            goal_jp[2] = 0.03
        goal_jp[joint] = joint_goal
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
        self.ecm_app.home_zero_position(10)
        for joint, goal in enumerate(self.goals):
            if joint == 2:
                max_duration = 1
                min_duration = 1
            else:
                max_duration = 1
                min_duration = 2
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
    rospy.init_node("record_ecm_calib",disable_signals=True)
    rospy.loginfo("Start Publishing ECM Tooltip Aruco Pose...")
    app = RECORD_ECM_CALIB("ECM",0.01)
    # ecm_potentiometer_js = message_filters.Subscriber("/ECM/io/analog_input_pos_si",JointState)
    # ecm_encoder_js       = message_filters.Subscriber("/ECM/io/joint_measured_js",  JointState)
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
