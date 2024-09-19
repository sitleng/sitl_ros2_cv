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
import time

from utils import tf_utils, ik_utils, dvrk_utils

from geometry_msgs.msg import TransformStamped,Vector3,Quaternion,Transform
from sitl_dvrk.msg import param_3D

from sitl_manus.msg import glove

import math
import numpy as np

from pyquaternion import Quaternion
import tf


class PSM_HAND_CTRL():
    def __init__(self):
        

        self.br = tf.TransformBroadcaster()  
        self.angle = None
        self.params = {

            "calib_fn": "/home/phd-leonardo-sitl/aruco_data/psm2_calib_results_final_new_v2.mat",
            "arm_name": "PSM2",
            "expected_interval": 0.01,
            
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
            "TransWeight": 0.9,
            "RotWeight": 0.1
        }

        
        self.psm2_ik_test = ik_utils.dvrk_custom_ik(self.params)
        
        
        self.init_jp = np.zeros_like(self.psm2_ik_test.arm.get_jp())
        
        self.init_jp[2] = 0.05
        
        #self.psm2_ik_test.arm.run_servo_jp(self.init_jp,5)

        self.g_psm2base_psm2tip = tf_utils.tfstamped2g(rospy.wait_for_message("/PSM2/custom/local/setpoint_cp",TransformStamped)) #DOUBLE CHECK IF THIS IS THE INITILA POS OR THE BASE FRAME
        self.psm2init_tf = tf_utils.g2tf(self.g_psm2base_psm2tip)
        
        self.pos = None

        self.r1_euler = None
        self.index_T= np.array([1,1,1])


        self.R_tracker = None
        
        self.x = None
        self.y = None
        self.z = None

        self.rx = None
        self.ry = None
        self.rz = None
        self.rw = None

        self.brx = None
        self.bry = None
        self.brz = None
        self.brw = None

        self.next_rotation = TransformStamped()
        self.next_rotation.child_frame_id = "next_rotation"
        self.next_rotation.header.frame_id = "psm2_base"
        self.next_rotation.transform.translation.x=0
        self.next_rotation.transform.translation.y=0
        self.next_rotation.transform.translation.z=0

        self.Qbase=None
        self.Qtracker=None

        self.rate = rospy.Rate(100)

    def __del__(self):
        print("Shutting down...")

    def callback(self,angle_msg):
        self.angle = angle_msg.data

    def rescale_value(self,original_value, max_old, min_old,min_new,max_new):

        rescaled_value = ((original_value - min_old) * (max_new - min_new)) / (max_old - min_old) + min_new

        if rescaled_value > max_new:
            rescaled_value = max_new

        elif rescaled_value < min_new:
            rescaled_value = min_new

        return rescaled_value
    
  
    
    def rotate_quat(self,q1,q2):
        resulting_quat = q1 * q2 
       

        return resulting_quat.x, resulting_quat.y, resulting_quat.z, resulting_quat.w



    def callbackTracker_frame(self, tracker_transform):
        self.x = -tracker_transform.transform.translation.z
        self.y = tracker_transform.transform.translation.y
        self.z = tracker_transform.transform.translation.x



    def callbackTracker_rot(self, tracker_transform_rot):
        self.rx = tracker_transform_rot.transform.rotation.x
        self.ry = tracker_transform_rot.transform.rotation.y
        self.rz = tracker_transform_rot.transform.rotation.z
        self.rw = tracker_transform_rot.transform.rotation.w

        self.Qtracker=Quaternion(w=self.rw,x=self.rx,y=self.ry,z=self.rz)

        
    def callbackTracker_base(self, base_rot):
        self.brx = base_rot.transform.rotation.x
        self.bry = base_rot.transform.rotation.y
        self.brz = base_rot.transform.rotation.z
        self.brw = base_rot.transform.rotation.w
        self.Qbase=Quaternion(w=self.brw,x=self.brx,y=self.bry,z=self.brz)
    
    def listener(self):
        rospy.Subscriber("/GLOVE_INPUT/T", TransformStamped, self.callbackTracker_frame)
        rospy.Subscriber("vive_tracker_transform", TransformStamped, self.callbackTracker_rot)
        rospy.Subscriber("base_tracker", TransformStamped, self.callbackTracker_base)

    def compute_rel(self, q1,q2):
        if (q1 and q2) is not None: 

            # Compute the relative rotation quaternion
            q_rel = q1.inverse * q2
            # Normalize the relative rotation quaternion to ensure it's a unit quaternion
            q_rel = q_rel.normalised
        
            return q_rel
        else:
            qrel= Quaternion(0, 0, 0, 1) 
            return qrel
                        
if __name__=="__main__":

    rospy.init_node("psm_hand_ctrl") 
   
    hand_app = PSM_HAND_CTRL()
    hand_app.listener()

    g_psm2_actualpos = tf_utils.tfstamped2g(rospy.wait_for_message("/PSM2/custom/local/setpoint_cp",TransformStamped))
    R=g_psm2_actualpos[0:3,0:3]
    q_psm2_actualpos = Quaternion(matrix=R)

    count = 0
    q_rotation_z_neg_90 = Quaternion(w=0.7071, x=0, y=0, z=-0.7071)
    q_rotation_test = Quaternion(w=0.7071, x=0, y=0, z=0.7071)
    first =  True
    second = True 
    cpt=0
    try:
        while not rospy.is_shutdown():
            try:
                
                t = rospy.Time.now()

                hand_app.next_rotation.header.stamp = t
                q_psm2_actualpos_rotated =  (q_psm2_actualpos*hand_app.Qtracker).normalised
                
                q=Quaternion(w=q_psm2_actualpos_rotated.w,x=q_psm2_actualpos_rotated.x,y=q_psm2_actualpos_rotated.y,z=q_psm2_actualpos_rotated.z)
                qxinv=(q*q_rotation_z_neg_90).normalised
                qxinv=(qxinv*q_rotation_test).normalised
                hand_app.next_rotation.transform.rotation.w=qxinv.w
                hand_app.next_rotation.transform.rotation.x=qxinv.x
                hand_app.next_rotation.transform.rotation.y=qxinv.y
                hand_app.next_rotation.transform.rotation.z=qxinv.z

                hand_app.next_rotation.transform.translation.x=hand_app.x
                hand_app.next_rotation.transform.translation.y=hand_app.y
                hand_app.next_rotation.transform.translation.z=hand_app.z

                print("translation: ",hand_app.x,hand_app.y,hand_app.z)
                hand_app.br.sendTransformMessage(hand_app.next_rotation)

                

                rotation_matrix = q_psm2_actualpos_rotated.rotation_matrix
                T_psm2 = tf_utils.gen_g(rotation_matrix,np.array([hand_app.x, hand_app.y, hand_app.z]).reshape(3,1))

                

                hand_app.psm2_ik_test.target = hand_app.g_psm2base_psm2tip.dot(T_psm2)
                hand_app.psm2_ik_test.target = T_psm2
                
                goal_jp = hand_app.psm2_ik_test.get_goal_jp()
                

                # yaw, pitch, roll = hand_app.Qtracker.yaw_pitch_roll

                

                # # Convert radians to degrees if desired
                # yaw_degrees = np.degrees(yaw)
                # pitch_degrees = np.degrees(pitch)
                # roll_degrees = np.degrees(roll)

                # rroll = hand_app.rescale_value(roll_degrees,-180, 180,hand_app.params["joint4_min"],hand_app.params["joint4_max"])
                # rpitch = hand_app.rescale_value(pitch_degrees,-180,180,hand_app.params["joint5_min"],hand_app.params["joint5_max"])
                # ryaw = hand_app.rescale_value(roll_degrees,-180,180,hand_app.params["joint6_min"],hand_app.params["joint6_max"])
                
                

                hand_app.rate.sleep()

            except:
                pass

    except Exception as e:
        traceback.print_exc()

    finally:
        del hand_app