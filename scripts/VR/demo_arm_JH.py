#!/usr/bin/env python3

import rospy
import random
from std_msgs.msg import Float64
import numpy as np
np.set_printoptions(suppress=True)
import math
import traceback
from utils import dvrk_devel_utils
import math
import numpy as np
import tf
from std_msgs.msg import Float64

class PSM_YAW_CTRL():
    def __init__(self, expected_interval):
        self.angle = None
        self.psm_app = dvrk_devel_utils.DVRK_CTRL("PSM2",expected_interval)
        self.bool_angle =  False

    def __del__(self):
        print("Shutting down...")
    
    def defineangle_arm(self,angle):
        if angle == 10:
            angle = -10

        else:
            angle = 35

        return angle

    def check_angle(self,angle):

        if angle < 0.2617:

            self.bool_angle =  True

        else:

            self.bool_angle = False


    def callback(self,angle_msg):
        self.angle = angle_msg.data


if __name__=="__main__":

    expected_interval = 1/1000

    hand_app = PSM_YAW_CTRL(expected_interval)
    rospy.Subscriber("glove/left/angle",Float64,hand_app.callback)
    
    init_jp = hand_app.psm_app.get_jp()

    try:
        while not rospy.is_shutdown():
            arm_init = hand_app.psm_app.get_jp()
 
            # start_angle = np.copy(hand_app.psm_app.get_jaw_jp())           
            duration = 1  # seconds
            samples = int(duration / expected_interval)
            # goal_angle = hand_app.defineangle(goal_angle)            
            
            arm_goal = np.copy(init_jp)
            for i,_ in enumerate(arm_goal):
                if i == 2:
                    arm_goal[i] += math.radians(random.randint(-1,1))
                else:
                    arm_goal[i] += math.radians(random.randint(-5,5))
            arm_amp = (arm_goal-arm_init)/samples
            
            for i in range(samples): 
                cur_arm_goal = arm_init + arm_amp*i
                hand_app.psm_app.arm.servo_jp(cur_arm_goal)
            
                rospy.sleep(expected_interval)
                
    except Exception as e:
        traceback.print_exc()

    finally:
        del hand_app