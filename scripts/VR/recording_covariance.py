from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
#!/usr/bin/env python3

import rospy
import tf
from ros_numpy.point_cloud2 import pointcloud2_to_array
from sensor_msgs.msg import JointState, PointCloud2
from geometry_msgs.msg import TransformStamped, PointStamped
from scipy.spatial.transform import Rotation as R
import os
import cv2
from tqdm.notebook import tqdm
import math
import numpy as np
np.set_printoptions(suppress=True)
import random
import scipy.io as sio
from scipy.optimize import least_squares
import sys
sys.path.append("/home/leo/catkin_ws/src/sitl_dvrk_custom_ik/scripts/")
from utils import tf_utils, dvrk_utils, pcl_utils, aruco_utils, ik_utils
from geometry_msgs.msg import Quaternion
from filterpy.kalman import ExtendedKalmanFilter
from numpy.random import normal

rospy.init_node("recording_covariance")
# rate = rospy.Rate(100)

# Create an empty list to store R_matrix values
R_matrix_values = []

# Loop for 10k iterations
for i in range(1000):

    t = rospy.Time.now()
    g_psm2_rotation = rospy.wait_for_message("hand_3D/left/R_psm2", Quaternion)

    r1 = R.from_quat(np.array([g_psm2_rotation.x, g_psm2_rotation.y, g_psm2_rotation.z, g_psm2_rotation.w]).flatten()) 

    quat = r1.as_quat()

    # Append the current R_matrix value to the list
    R_matrix_values.append(quat)  # Corrected variable name

# Stack all R_matrix values into a single array
quat_matrix_stack = np.stack(R_matrix_values)
# Compute the covariance matrix of all R_matrix values
covariance_matrix = np.cov(quat_matrix_stack, rowvar=False)

print("Covariance Matrix:")
print(covariance_matrix)

# Save the covariance matrix to a text file
np.savetxt("covariance_matrix.txt", covariance_matrix, fmt='%.8f')

