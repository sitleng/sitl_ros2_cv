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
from utils import tf_utils, ik_utils, dvrk_devel_utils
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion, Transform
from sensor_msgs.msg import JointState
from sitl_dvrk.msg import param_3D
import math
import numpy as np
from pyquaternion import Quaternion
import tf
from std_msgs.msg import Float64

# Define a class named PSM_YAW_CTRL
class PSM_YAW_CTRL():
    def __init__(self, expected_interval):
        self.angle = None
        self.psm_app = dvrk_devel_utils.DVRK_CTRL("PSM2", expected_interval)
        self.bool_angle = False

    def __del__(self):
        print("Shutting down...")

    def defineangle(self, angle):
        # Define a method to change the 'angle' value based on a condition
        if angle == 50:
            angle = 0

        else:
            
            angle = 50
        return angle

    def callback(self, angle_msg):
        
        self.angle = angle_msg.data


if __name__ == "__main__":

    expected_interval = 1 / 10000

    hand_app = PSM_YAW_CTRL(expected_interval)
    pub_jaw_servo_jp = rospy.Publisher("/PSM2/jaw/servo_jp", JointState, queue_size=10)


    msg = JointState()
    msg.name = "jaw"
    goal_angle = 0

    try:
        while not rospy.is_shutdown():
            # Get the current angle of the jaw
            start_angle = hand_app.psm_app.get_jaw_jp()
            
            # Set the duration and the number of samples
            duration = 0.2
            samples = int(duration / expected_interval)
            goal_angle = hand_app.defineangle(goal_angle)

            # Calculate the amplitude of change in angle
            amplitude = (math.radians(goal_angle) - start_angle) / samples
            print("start angle: ", start_angle, " samples: ", samples, " amplitude: ", amplitude)

            for i in range(samples):
                # Calculate the goal angle for the jaw servo
                goal = start_angle + amplitude * i

                # Set the position in the JointState message
                msg.position = goal
                msg.header.stamp = rospy.Time.now()
                # print("goal", goal)

                # Move the jaw servo to the goal angle
                hand_app.psm_app.arm.jaw.servo_jp(goal)


                rospy.sleep(expected_interval)

    except Exception as e:

        traceback.print_exc()

    finally:
        del hand_app
