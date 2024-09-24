#!/usr/bin/env python3

import rospy
import numpy as np
np.set_printoptions(suppress=True)

from utils import dvrk_devel_utils

import numpy as np

arm_name = "PSM2"
expected_interval = 0.001
node_name = "PSM2_homing"

# rospy.init_node("PSM2_hand")

psm2_ik_test = dvrk_devel_utils.DVRK_CTRL("PSM2", 0.001, node_name)
init_jp = psm2_ik_test.get_jp()
goal_jp = np.zeros_like(init_jp)
goal_jp[2] = 0.05

init_jaw = psm2_ik_test.get_jaw_jp()
goal_jaw = np.zeros_like(init_jaw)

duration = 5

if __name__=="__main__":

    psm2_ik_test.run_full_servo_jp(goal_jp, goal_jaw, duration)