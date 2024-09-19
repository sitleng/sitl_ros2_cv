#!/usr/bin/env python3

import rospy
import random
from std_msgs.msg import Float64
from utils import dvrk_utils
from scipy.spatial.transform import Rotation as R
from tqdm.notebook import tqdm
import numpy as np
np.set_printoptions(suppress=True)
import math
import traceback
from utils import tf_utils, ik_utils, dvrk_utils
from geometry_msgs.msg import TransformStamped,Vector3,Quaternion,Transform
from sensor_msgs.msg import JointState
from sitl_dvrk.msg import param_3D
from sitl_manus.msg import glove
import math
import numpy as np
from pyquaternion import Quaternion
import tf
from std_msgs.msg import Float64

class PSM_YAW_CTRL():
    def __init__(self, expected_interval):
        self.angle = None
        self.psm_app = dvrk_utils.DVRK_CTRL("PSM2",expected_interval)

    def __del__(self):
        print("Shutting down...")
    
    def defineangle(self,angle):
        
        if angle == 30:
            angle = 0

        else:
            angle = 30

        return angle
    


    def callback(self,angle_msg):
        self.angle = angle_msg.data


if __name__=="__main__":

    rospy.init_node("psm_yaw_ctrl") 
    
    rate = rospy.Rate(1000)
    expected_interval = 1/1000

    hand_app = PSM_YAW_CTRL(expected_interval)
    rospy.Subscriber("glove/left/angle",Float64,hand_app.callback)
    pub_jaw_servo_jp = rospy.Publisher("/PSM2/jaw/servo_jp", JointState, queue_size=10)
    
    start_angle = np.copy(hand_app.psm_app.get_jaw_jp())
    test = hand_app.psm_app.get_jp()
    test = np.append(test,start_angle)
    msg = JointState()
    msg.position = test#.tolist()
    msg.header.stamp = rospy.Time.now()
    pub_jaw_servo_jp.publish(msg)
    goal_angle = 0

    arm_init = hand_app.psm_app.get_jp()
    jaw_init = hand_app.psm_app.get_jaw_jp()

    try:
        while not rospy.is_shutdown():
 
            # start_angle = np.copy(hand_app.psm_app.get_jaw_jp())           
            duration = 1  # seconds
            samples = int(duration / expected_interval)
            # goal_angle = hand_app.defineangle(goal_angle)            
            
            arm_goal = np.copy(arm_init)
            for i,_ in enumerate(arm_goal):
                arm_goal[i] += math.radians(random.randint(-5,5))
            arm_amp = (arm_goal-arm_init)/samples
            
            jaw_goal = np.copy(jaw_init)
            jaw_goal += math.radians(random.randint(-10,10))
            jaw_amp = (jaw_goal-jaw_init)/samples
            # jaw_amplitude = (math.radians(goal_angle)-start_angle)/samples

            test = np.append(test,0)
            for i in range(samples):
                # goal = start_angle + amplitude*i #[start_angle + amplitude*i]
                # test[-1] = goal
                # print(test)
                # msg.position = test
                # msg.header.stamp = rospy.Time.now()
                # pub_jaw_servo_jp.publish(msg)
                # hand_app.psm_app.arm.jaw.servo_jp(test)

                # arm_goal = arm_init + arm_amp*i
                jaw_goal = jaw_init + jaw_amp*i
                # hand_app.psm_app.arm.servo_jp(arm_goal)
                hand_app.psm_app.arm.jaw.servo_jp(jaw_goal)

                rospy.sleep(expected_interval)


    except Exception as e:
        traceback.print_exc()

    finally:
        del hand_app