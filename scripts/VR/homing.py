#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from utils import dvrk_utils
from scipy.spatial.transform import Rotation as R
from tqdm.notebook import tqdm
import numpy as np
np.set_printoptions(suppress=True)
import math
import traceback

from utils import tf_utils, ik_devel_utils, dvrk_devel_utils

from geometry_msgs.msg import TransformStamped,Vector3,Quaternion,Transform
from sitl_dvrk.msg import param_3D

from sitl_manus.msg import glove

import math
import numpy as np

from pyquaternion import Quaternion
import tf      

br = tf.TransformBroadcaster()  
angle = None
params = {

            "calib_fn": "/home/phd-leonardo-sitl/aruco_data/psm2_calib_results_final_new_v2.mat",
            "arm_name": "PSM2",
            "expected_interval": 0.001,
            
            "joint1_min": -30,
            "joint2_min": -30,
            "joint3_min":  0.01,
            "joint4_min": -200,
            "joint5_min": -65,
            "joint6_min": -80,
            "wT_min"    :  1e-5,
            "wR_min"    :  3e-7,
            "joint1_max":  30,
            "joint2_max":  30,
            "joint3_max":  0.2,
            "joint4_max":  200,
            "joint5_max":  65,
            "joint6_max":  80,
            "wT_max"    :  1e1,
            "wR_max"    :  1e-3,
            "TransWeight": 0.99,
            "RotWeight": 0.01
        }

rospy.init_node("PSM2_hand")
psm2_ik_test = dvrk_devel_utils.DVRK_CTRL("PSM2", 0.001, rospy.get_name())
init_jp = psm2_ik_test.get_jp()
goal_jp = np.zeros_like(init_jp)
goal_jp[2] = 0.05
print(init_jp)
print(goal_jp)

init_jaw = psm2_ik_test.get_jaw_jp()
goal_jaw = np.zeros_like(init_jaw)
print(init_jaw)
print(goal_jaw)

if __name__=="__main__":

    psm2_ik_test.run_jaw_servo_jp(goal_jaw, 3)
    psm2_ik_test.run_arm_servo_jp( goal_jp, 5)