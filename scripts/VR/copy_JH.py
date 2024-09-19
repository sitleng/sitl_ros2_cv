#!/usr/bin/env python3

# Import necessary libraries and modules
import rospy
from std_msgs.msg import Float64
from utils import dvrk_utils
from scipy.spatial.transform import Rotation as R
from tqdm.notebook import tqdm
import numpy as np
np.set_printoptions(suppress=True)
import math
import traceback
from utils import tf_utils, ik_utils, dvrk_utils
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion, Transform
from sensor_msgs.msg import JointState
from sitl_dvrk.msg import param_3D
from sitl_manus.msg import glove
import math
import numpy as np
from pyquaternion import Quaternion
import tf
from std_msgs.msg import Float64

# Define a class named PSM_YAW_CTRL
class PSM_YAW_CTRL():
    def __init__(self, expected_interval):
        self.angle = None
        self.psm_app = dvrk_utils.DVRK_CTRL("PSM2", expected_interval)
        self.bool_angle = False

    def __del__(self):
        print("Shutting down...")

    def defineangle(self, angle):
        # Define a method to change the 'angle' value based on a condition
        if angle == 20:
            angle = 0
        else:
            angle = 20
        return angle

    def callback(self, angle_msg):
        
        self.angle = angle_msg.data


if __name__ == "__main__":
    rospy.init_node("psm_yaw_ctrl")


    rate = rospy.Rate(1000)
    expected_interval = 1 / 1200

    hand_app = PSM_YAW_CTRL(expected_interval)
    pub_jaw_servo_jp = rospy.Publisher("/PSM2/jaw/servo_jp", JointState, queue_size=10)

    start_angle = math.radians(50.0)
    amplitude = math.radians(30.0)
    hand_app.psm_app.arm.jaw.open(angle = start_angle).wait()

    duration = 1  # seconds
    samples = int(duration /expected_interval)
    

    try:
        for i in range(samples * 4):
            goal = start_angle + amplitude * (math.cos(i * math.radians(360.0) / samples) - 1.0)
            hand_app.psm_app.arm.jaw.servo_jp(np.array(goal))
            rospy.sleep(expected_interval)

    except Exception as e:

        traceback.print_exc()

    finally:
        del hand_app
       





       
        
       
       
        