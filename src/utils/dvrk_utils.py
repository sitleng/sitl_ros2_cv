#!/usr/bin/env python3

import sys
import dvrk
import rospy
import numpy as np
np.seterr(all="ignore")
import math
from geometry_msgs.msg import TransformStamped
from utils import tf_utils
import tf_conversions.posemath as pm

# print with node id
def print_id(message):
    print('%s -> %s' % (rospy.get_caller_id(), message))

# example of application using arm.py
class DVRK_CTRL():

    # configuration
    def __init__(self, arm_name, expected_interval):
        print_id('configuring dvrk_arm_test for %s' % arm_name)
        self.expected_interval = expected_interval
        self.arm_cp_tn = "/{}/custom/setpoint_cp".format(arm_name)
        self.arm_local_cp_tn = "/{}/custom/local/setpoint_cp".format(arm_name)
        if "PSM" in arm_name:
            self.jaw_cp_tn = "/{}/custom/jaw/setpoint_cp".format(arm_name)
            self.jaw_local_cp_tn = "/{}/custom/local/jaw/setpoint_cp".format(arm_name)
            self.arm = dvrk.psm(arm_name = arm_name,
                                expected_interval = expected_interval)
        else:
            self.arm = dvrk.arm(arm_name = arm_name,
                                expected_interval = expected_interval)
        self.init_jp = np.copy(self.get_jp())
            
    def __del__(self):
        print("Destructing Class DVRK_CTRL...")
        
    def home_zero_position(self,duration):
        print_id('starting enable')
        if not self.arm.enable(10):
            sys.exit('failed to enable within 10 seconds')
        print_id('starting home')
        if not self.arm.home(10):
            sys.exit('failed to home within 10 seconds')
        # get current joints just to set size
        print_id('move to zero position')
        zero_jp = np.copy(self.get_jp())
        # go to zero position, for PSM and ECM make sure 3rd joint is past cannula
        zero_jp.fill(0)
        if "PSM" in self.arm.name():
            self.arm.jaw.open(angle=math.radians(0)).wait()
        else:
            zero_jp[2] = 0.03
        self.run_servo_jp(zero_jp,duration)
        print_id('moving to zero position complete')

    def home_init_position(self,duration):
        # get current joints just to set size
        print_id('move to init position')
        self.run_servo_jp(self.init_jp,duration)
        print_id('moving to init position complete')

    def home_test_position(self,duration):
        print_id('starting enable')
        if not self.arm.enable(10):
            sys.exit('failed to enable within 10 seconds')
        print_id('starting home')
        if not self.arm.home(10):
            sys.exit('failed to home within 10 secsetpoint_jponds')
        # get current joints just to set size
        print_id('move to test position')
        zero_jp = np.copy(self.get_jp())
        # go to initial position
        zero_jp.fill(0)
        if "PSM" in self.arm.name():
            self.arm.jaw.open(angle=math.radians(0)).wait()
        zero_jp[2] = 0.12
        self.run_servo_jp(zero_jp,duration)
        print_id('moving to test position complete')

    def update_init_jp(self):
        self.init_jp = np.copy(self.get_jp())

    # direct joint control example
    def run_servo_jp(self,goal,duration=5):
        initial_joint_position = np.copy(self.get_jp())
        samples = duration / self.expected_interval
        amplitude = (goal-initial_joint_position)/samples
        for i in range(int(samples)):
            cur_goal = initial_joint_position + i*amplitude
            self.arm.servo_jp(cur_goal)
            rospy.sleep(self.expected_interval)

    def run_servo_jp_with_jaw(self,arm_goal,jaw_goal=math.radians(0), duration=5):
        initial_joint_position = np.copy(self.get_jp())
        samples = duration / self.expected_interval
        arm_amplitude = (arm_goal-initial_joint_position)/samples

        # self.arm.jaw.servo_jp(np.array(math.radians(jaw_goal)))

        for i in range(int(samples)):
            cur_arm_goal = initial_joint_position + i*arm_amplitude
            self.arm.servo_jp(cur_arm_goal)
            rospy.sleep(self.expected_interval)

    def get_jp(self):
        while True:
            try:
                jp = self.arm.setpoint_jp()
                return jp
            except:
                continue

    def get_jaw_jp(self):
        while True:
            try:
                jp = self.arm.jaw.setpoint_jp()
                return jp
            except:
                continue

    def get_cp(self):
        while True:
            try:
                # cp = pm.toMatrix(self.arm.setpoint_cp())
                cp = tf_utils.tfstamped2g(rospy.wait_for_message(self.arm_cp_tn,TransformStamped))
                return cp
            except:
                continue

    def get_local_cp(self):
        while True:
            try:
                # cp = pm.toMatrix(self.arm.setpoint_cp())
                cp = tf_utils.tfstamped2g(rospy.wait_for_message(self.arm_local_cp_tn,TransformStamped))
                return cp
            except:
                continue

    def get_jaw_cp(self):
        while True:
            try:
                # cp = pm.toMatrix(self.arm.setpoint_cp())
                cp = tf_utils.tfstamped2g(rospy.wait_for_message(self.jaw_cp_tn,TransformStamped))
                return cp
            except:
                continue

    def get_local_jaw_cp(self):
        while True:
            try:
                # cp = pm.toMatrix(self.arm.setpoint_cp())
                cp = tf_utils.tfstamped2g(rospy.wait_for_message(self.jaw_local_cp_tn,TransformStamped))
                return cp
            except:
                continue