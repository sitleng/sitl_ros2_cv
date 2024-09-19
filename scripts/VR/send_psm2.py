
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
#!/usr/bin/env python3

import rospy
import tf
from geometry_msgs.msg import TransformStamped

from scipy.spatial.transform import Rotation as R
from tqdm.notebook import tqdm
import numpy as np
np.set_printoptions(suppress=True)
import math
import sys
sys.path.append("/home/leo/catkin_ws/src/sitl_dvrk_custom_ik/scripts/")
from utils import tf_utils, aruco_utils, ik_utils, dvrk_utils
from geometry_msgs.msg import Quaternion
from filterpy.kalman import ExtendedKalmanFilter
from numpy.random import normal

rospy.init_node("dvrk_custom_ik_vr_test")

br = tf.TransformBroadcaster()

#################################### EXTENDED KALMAN FILTER ####################################

# def Hx(state):
#     return state

# def HJ(state):
#     return np.eye(4)

# # Create a Kalman filter with appropriate dimensions
# kf = ExtendedKalmanFilter(dim_x=4, dim_z=4)

# # Define initial state and covariance matrix
# initial_state = r1.as_quat()  # Initial state is your rotation_applied_psm2
# initial_covariance = np.eye(4) * 10   # Initial covariance matrix (adjust as needed)

# kf.x = initial_state
# kf.P = initial_covariance

# # Define transition matrix (state transition model)
# kf.F = np.eye(4)

# # Define measurement matrix (observation model)
# kf.H = np.eye(4)

# # Define process noise covariance matrix
# #Generate a random white Gaussian noise matrix with the desired standard deviation
# white_noise = normal(loc=0, scale=0.5, size=(4, 4))

# # Set the process noise covariance matrix with the generated white noise
# kf.Q = white_noise @ white_noise.T

# # Define measurement noise covariance matrix
# # You'll need to tune this based on the noise characteristics of your measurements
# loaded_covariance_matrix = np.loadtxt("covariance_matrix.txt")

# kf.R = loaded_covariance_matrix # build a covariance matrix by taking 1000 measurements

######################################################################################################################

params = {

    "calib_fn": "/home/phd-leonardo-sitl/aruco_data/psm2_calib_results_final_new_v2.mat",
    "arm_name": "PSM2",
    "expected_interval": 0.01,
    
    "joint1_min": -30,
    "joint2_min": -30,
    "joint3_min":  0.02,
    "joint4_min": -100,
    "joint5_min": -65,
    "joint6_min": -65,
    "wT_min"    :  1e-3,
    "wR_min"    :  3e-4,
    "joint1_max":  30,
    "joint2_max":  30,
    "joint3_max":  0.08,
    "joint4_max":  100,
    "joint5_max":  65,
    "joint6_max":  65,
    "wT_max"    :  1e1,
    "wR_max"    :  1e1,
    "TransWeight": 1e-6,
    "RotWeight": 1e-3
}

psm2_ik_test = ik_utils.dvrk_custom_ik(params)

rate = rospy.Rate(30)
init_jp = np.zeros_like(psm2_ik_test.arm.get_jp())
init_jp[2] = 0.05
psm2_ik_test.arm.run_servo_jp(init_jp,5)



g_psm2base_psm2tip = tf_utils.tfstamped2g(rospy.wait_for_message("/PSM2/custom/local/setpoint_cp",TransformStamped))
psm2init_tf = tf_utils.g2tf(g_psm2base_psm2tip)


def rescale_value(original_value, max_old, min_old,min_new,max_new):

    rescaled_value = ((original_value - min_old) * (max_new - min_new)) / (max_old - min_old) + min_new

    if rescaled_value > max_new:
        rescaled_value = max_new

    elif rescaled_value < min_new:
        rescaled_value = min_new

    return rescaled_value

psm2_app = dvrk_utils.DVRK_CTRL("PSM2",0.01)


def callback(quat_msg):

    t = rospy.Time.now()
    r1 = R.from_quat(np.array([quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w]).flatten())
    r1_euler=r1.as_euler('xyz',degrees=True)
    
    print(" \n r1 \n ",r1.as_euler('xyz'))
    rroll = rescale_value(r1_euler[0],0, 360,params["joint4_min"],params["joint4_max"])
    rpitch = rescale_value(r1_euler[1],0,360,params["joint5_min"],params["joint5_max"])
    ryaw = rescale_value(r1_euler[2],0,360,params["joint6_min"],params["joint6_max"])

    print("Eurler angles",rroll,rpitch,ryaw)
    
    goal_jp = psm2_app.get_jp()
    goal_jp[3] = math.radians(rroll)
    goal_jp[4] = math.radians(rpitch)

    #goal_jp[5] = math.radians(ryaw)
    
    psm2_app.run_servo_jp(goal_jp,duration=1)

    '''
    T_psm2 = tf_utils.gen_g(r1.as_matrix(),np.array([0,0,0]).reshape(3,1))
    
    br.sendTransform(
        (psm2init_tf.translation.x,psm2init_tf.translation.y,psm2init_tf.translation.z),
        (psm2init_tf.rotation.x,psm2init_tf.rotation.y,psm2init_tf.rotation.z,psm2init_tf.rotation.w),
        t,"psm2_init","psm2_base"
    )
    br.sendTransform(
         (np.array([0,0,0])),
         (r1.as_quat()),
         t,"psm2_command","psm2_init"
    )
    
    psm2_ik_test.target = g_psm2base_psm2tip.dot(T_psm2)

    psm2_ik_test.move_to_goal(0.5)
    '''



while not rospy.is_shutdown():
    
    rospy.Subscriber("hand_3D/left/R_psm2", Quaternion, callback)

    rate.sleep()