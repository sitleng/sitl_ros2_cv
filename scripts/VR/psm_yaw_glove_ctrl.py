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

from utils import  dvrk_devel_utils, ik_devel_utils



import math
import numpy as np

from std_msgs.msg import Float64

from std_msgs.msg import Bool


class PSM_YAW_CTRL():
    def __init__(self, expected_interval):
        print("1")
        self.angle = None
        self.psm_app = dvrk_devel_utils.DVRK_CTRL("PSM2",expected_interval)
        self.bool_angle = False 
        print("2")
        
    def __del__(self):
        print("Shutting down...")

    def callback(self,angle_msg):
        self.angle = angle_msg.data
    
    def check_angle(self,angle):
        if angle < 0.2617:
            self.bool_angle =  False
        else:
            self.bool_angle = True
            
if __name__=="__main__":

    # rospy.init_node("psm_yaw_ctrl") 
    
    # rate = rospy.Rate(1000)
    expected_interval = 1/1000

    hand_app = PSM_YAW_CTRL(expected_interval)
    pub_angle_bool = rospy.Publisher('glove/left/angle_bool', Bool, queue_size=10)
    pub_angle_bool.publish(False)

    rospy.Subscriber("glove/left/angle",Float64,hand_app.callback)

    # print(start_angle)

    try:
        while not rospy.is_shutdown():
            if hand_app.angle is not None:
                start_angle = np.copy(hand_app.psm_app.get_jaw_jp())
                hand_app.psm_app.run_jaw_servo_jp(start_angle,math.radians(hand_app.angle), duration=0.1)

 
############## to control the yaw #######################
            # try:    
            #     hand_app.psm_app.arm.jaw.move_jp(np.array(math.radians(hand_app.angle))).wait()
            # except:
            #     pass

            #SERVO JP TRY
            # if hand_app.angle is not None: 
                
            #     print("help")
            #     hand_app.psm_app.arm.jaw.servo_jp(math.radians(hand_app.angle)).wait()
            #     print("help")
            #     hand_app.check_angle(hand_app.psm_app.get_jaw_jp())
            #     pub_angle_bool.publish(hand_app.bool_angle)

               
            #     rospy.sleep(expected_interval)


    except Exception as e:
        traceback.print_exc()

    finally:
        del hand_app