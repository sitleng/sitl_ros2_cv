#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64,Bool
from utils import dvrk_utils
from scipy.spatial.transform import Rotation as R
from tqdm.notebook import tqdm
import numpy as np
np.set_printoptions(suppress=True)
import math
import traceback

from utils import tf_utils, ik_devel_utils, dvrk_devel_utils

from geometry_msgs.msg import TransformStamped,Vector3,Quaternion,Transform,PoseStamped

from sitl_dvrk.msg import param_3D,BoolStamped, Float64Stamped

from sitl_manus.msg import glove

import time

import math
import numpy as np

from pyquaternion import Quaternion
import tf
from sensor_msgs.msg import JointState

def sig(x):
    return 1/(1 + np.exp(-5e-4*x))

class PSM_HAND_CTRL():

    def __init__(self, params):
        
        self.br = tf.TransformBroadcaster()  
        self.angle = None
        # self.params = {

        #     "calib_fn": "/home/phd-leonardo-sitl/aruco_data/psm2_calib_results_final_new_v2.mat",
        #     "arm_name": "PSM2",
        #     "expected_interval": 0.001,
            
        #     # "joint1_min": -40,
        #     # "joint2_min": -40,
        #     # "joint3_min":  0.01,
        #     # "joint4_min": -190,
        #     # "joint5_min": -75,
        #     # "joint6_min": -85,
        #     # "wT_min"    :  1e-5,
        #     # "wR_min"    :  3e-7,
        #     # "joint1_max":  40,
        #     # "joint2_max":  40,
        #     # "joint3_max":  0.24,
        #     # "joint4_max":  190,
        #     # "joint5_max":  75,
        #     # "joint6_max":  85,
        #     # "wT_max"    :  1e1,
        #     # "wR_max"    :  1e-3,
        #     # "TransWeight": 10,
        #     # "RotWeight": 2.05
        #     "TransWeight": 0.8,
        #     "RotWeight": 0.2
        # }

        self.params = params

        self.arm = dvrk_devel_utils.DVRK_CTRL("PSM2", 0.001, "PSM2_Hand_Ctrl")
        
        self.init_jp = np.zeros_like(self.arm.get_jp())
        self.init_jp[2] = 0.05

        self.arm_ik = ik_devel_utils.dvrk_custom_ik(
            self.params["calib_fn"],
            self.params["Tweight"],
            self.params["Rweight"],
            self.init_jp,
            self.params["Joffsets"]
        )

        self.g_psm2base_psm2tip = self.arm.get_local_cp() #DOUBLE CHECK IF THIS IS THE INITILA POS OR THE BASE FRAME
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

        self.angle = None

        self.flag = False
        self.flag_on_off = False
        self.bool_dist_real = False

        self.expected_interval = 1/100



    def __del__(self):
        print("Shutting down...")

    def callback(self,angle_msg):
        self.angle = angle_msg.data

    def calculate_distance(self,x1, y1, z1, x2, y2, z2):
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return distance
    
    def rescale_value(self,original_value, max_old, min_old,min_new,max_new):

        rescaled_value = ((original_value - min_old) * (max_new - min_new)) / (max_old - min_old) + min_new



        return rescaled_value
    
  
    def rotate_quat(self,q1,q2):
        
        resulting_quat = q1 * q2 
        return resulting_quat.x, resulting_quat.y, resulting_quat.z, resulting_quat.w


    def callback2(self,quat_msg):

        t = rospy.Time.now()
        r1 = R.from_quat(np.array([quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w]).flatten())
        self.r1_euler=r1.as_euler('xyz',degrees=True)
        self.r1_euler[0]=-self.r1_euler[0]-100
        

    def callback3(self,pos_msg):
        
        self.pos=np.array([pos_msg.x,pos_msg.y,pos_msg.z])
        self.pos[0] = self.rescale_value(self.pos[0],0.51,0.21,-0.3, 0.3)
        self.pos[1] = self.rescale_value(self.pos[1],1,0.75,-0.45,0.45)
        self.pos[2] = self.rescale_value(self.pos[2],0,-0.2,-1.33,0.025)


    def callback4(self, joint_pos):

        joint_values = joint_pos.joint
        index=joint_values[8]
        
        self.index_T=[ index.x,index.y,index.z ]
        #print(self.index_T)
        self.index_T[0] = self.rescale_value(self.index_T[0],0.3,-0.3,-0.4, -0.1)
        self.index_T[1] = self.rescale_value(self.index_T[1],0.18,-0.18,-0.1,0.1)
        self.index_T[2] = self.rescale_value(self.index_T[2],1,0,-1.33,0.025)


    def callbackTracker_frame(self, tracker_transform):

        self.x = tracker_transform.transform.translation.x      #-tracker_transform.transform.translation.z
        self.y = tracker_transform.transform.translation.y
        self.z = tracker_transform.transform.translation.z      #tracker_transform.transform.translation.x

    def callbackTracker_rot(self, tracker_transform_rot):
        if tracker_transform_rot is None:
            return

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
    
    def callback_fist(self,flag_fist):
        
        self.flag = flag_fist.data

    def callback_on_off(self,flag_on_off):

        self.flag_on_off = flag_on_off.data
        
    def callback_real(self,bool_dist_real):
        self.bool_dist_real = bool_dist_real.data

    def listener(self):
        rospy.Subscriber("glove/left/angle",Float64Stamped,hand_app.callback)
        rospy.Subscriber("/GLOVE_INPUT/T", TransformStamped, self.callbackTracker_frame)
        rospy.Subscriber("tracker_current_pos_tf", TransformStamped, self.callbackTracker_rot)
        rospy.Subscriber("base_tracker", TransformStamped, self.callbackTracker_base)
        rospy.Subscriber("glove/left/fist", BoolStamped, self.callback_fist)
        rospy.Subscriber("glove/left/on_off", BoolStamped, self.callback_on_off)
        rospy.Subscriber("glove/left/real", BoolStamped, self.callback_real)



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
                        
    


    # def set_duration(self, rot_distance, distance):

    #         if rot_distance > 0.5:
    #             rot_distance = 0.5
    #         duration = (2e-1*rot_distance + 2*distance)/3 + 0.0015

    #         print("duration: ",duration)
    #         return duration 
        
    def set_duration(self, cur_jp, goal_jp):
        tr_dist0 = np.linalg.norm(cur_jp[0] - goal_jp[0])
        tr_dist1 = np.linalg.norm(cur_jp[1] - goal_jp[1])
        tr_dist2 = np.linalg.norm(cur_jp[2] - goal_jp[2])
        ang_dist0 = np.linalg.norm(cur_jp[3] - goal_jp[3])
        ang_dist1 = np.linalg.norm(cur_jp[4] - goal_jp[4])
        ang_dist2 = np.linalg.norm(cur_jp[5] - goal_jp[5])
        # print("tr: ", tr_dist0," ",tr_dist1," ",tr_dist2 , " ang: ", ang_dist0," ",ang_dist1," ",ang_dist2)
        duration = tr_dist0/100+tr_dist1/50+tr_dist2/40 + ang_dist0/1000 +ang_dist1/1000+ang_dist2/1000+ 0.0018
        # print("duration: ",duration)
        return duration


if __name__=="__main__":

    rospy.init_node("psm_hand_ef")

    node_name = rospy.get_name()
    params    = rospy.get_param(node_name)
    
    hand_app = PSM_HAND_CTRL(params)
    hand_app.listener()

    
    g_psm2_actualpos = hand_app.arm.get_local_cp()

    pub_jaw_servo_jp = rospy.Publisher("/PSM2/jaw/servo_jp", JointState, queue_size=10)
    
    R = g_psm2_actualpos[0:3,0:3]
    q_psm2_actualpos = Quaternion(matrix=R)

    count = 0
    q_rotation_z_neg_90 = Quaternion(w=0.7071, x=0, y=0, z=-0.7071)
    q_rotation_test = Quaternion(w=0.7071, x=0, y=0, z=0.7071)
    first = True
    cpt=0
    precedent_rotation =Quaternion(w=1,x=0,y=0,z=0)
    precedent_translation =  np.zeros(3)
    
    # TO DELETE 
    # hand_app.flag_on_off =  True
    try:
        while not rospy.is_shutdown():

            q_psm2_actualpos_rotated =  hand_app.Qtracker

            if hand_app.flag_on_off == True:  
                
                rotation_matrix = q_psm2_actualpos_rotated.rotation_matrix
                T_psm2 = tf_utils.gen_g(rotation_matrix,np.array([hand_app.x, hand_app.y, hand_app.z]).reshape(3,1))

                hand_app.arm_ik.target = hand_app.g_psm2base_psm2tip.dot(T_psm2) 
                hand_app.arm_ik.target = T_psm2

                s = time.time()
                cur_jp = hand_app.arm.get_jp()
                goal_jp = hand_app.arm_ik.get_goal_jp(cur_jp)
                # print("IK Time Elapsed: {}".format(time.time()-s))
                
                if (first) or (hand_app.bool_dist_real == True) :
                    hand_app.arm.run_arm_servo_jp(  
                        goal_jp,
                        duration = 3
                    )
                    # precedent_rotation= q_psm2_actualpos_rotated
                    # precedent_translation = np.array([hand_app.x, hand_app.y, hand_app.z])
                    first = False
            
                else:
                    # rot_distance = Quaternion.absolute_distance(precedent_rotation, q_psm2_actualpos_rotated)

                    # distance = np.linalg.norm(np.array([hand_app.x, hand_app.y, hand_app.z]) - precedent_translation)
                    # duration_movement = hand_app.set_duration(rot_distance, distance)
                    duration_movement = hand_app.set_duration(cur_jp, goal_jp)
                    s = time.time()
                    hand_app.arm.run_arm_servo_jp(
                        goal_jp,
                        duration =duration_movement
                    )
                    # print("Move Arm Time Elapsed: {}".format(time.time()-s))
            else:
                first = True
                
            precedent_rotation = q_psm2_actualpos_rotated
            precedent_translation = np.array([hand_app.x, hand_app.y, hand_app.z])

            # print("ON_OFF status: ",hand_app.flag_on_off)



    except Exception as e:
        traceback.print_exc()

    finally:
        del hand_app

