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
        self.bool_angle =  False

    def __del__(self):
        print("Shutting down...")
    
    def defineangle(self,angle):
        
        if angle == 20:
            angle = 0

        else:
            angle = 20

        return angle
    
    def defineangle_arm(self,angle):
        if angle == 35:
            angle = -35

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

    rospy.init_node("psm_yaw_ctrl") 
    
    rate = rospy.Rate(1000)
    expected_interval = 1/1000

    hand_app = PSM_YAW_CTRL(expected_interval)
    rospy.Subscriber("glove/left/angle",Float64,hand_app.callback)
    pub_jaw_servo_jp = rospy.Publisher("/PSM2/jaw/move_jp", JointState, queue_size=10)
    
    start_angle = np.copy(hand_app.psm_app.get_jaw_jp())
    msg = JointState()
    msg.name = "jaw"
    msg.position = start_angle#.tolist()
    msg.header.stamp = rospy.Time.now()
    pub_jaw_servo_jp.publish(msg)
    goal_angle = 0
    goal_angle_arm = 0
    

    try:
        while not rospy.is_shutdown():
 
            start_angle = np.copy(hand_app.psm_app.get_jaw_jp())
            start_angle_arm = hand_app.psm_app.get_jp()
            duration = 0.2
            samples = int(duration / expected_interval)

            goal_angle = hand_app.defineangle(goal_angle)
            goal_angle_arm = hand_app.defineangle_arm(goal_angle_arm)

            print(goal_angle_arm)

            amplitude = (math.radians(goal_angle)-start_angle)/samples
            amplitude_arm =(math.radians(goal_angle_arm)-start_angle_arm[5])/samples
            
            for i in range(samples):

                goal = start_angle + amplitude*i 
                goal_arm = hand_app.psm_app.get_jp()
                goal_arm[5]= start_angle_arm[5]+amplitude_arm*i
                              
                # print(test)
                msg.position = goal
                msg.header.stamp = rospy.Time.now()
                
                #pub_jaw_servo_jp.publish(msg)
                # hand_app.psm_app.arm.servo_jp(goal_arm)
                hand_app.psm_app.arm.jaw.servo_jp(goal)

                hand_app.check_angle(hand_app.psm_app.get_jaw_jp())
                print(hand_app.bool_angle)
                rospy.sleep(expected_interval)
                
    except Exception as e:
        traceback.print_exc()

    finally:
        del hand_app