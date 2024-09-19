#!/usr/bin/env python

import cv2
import rospy
import tf
from geometry_msgs.msg import Vector3, PointStamped
import numpy as np
import math
import open3d as o3d

import time
from utils_display import DisplayCamera
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Quaternion
from sitl_dvrk.msg import param_3D
from utils import tf_utils
from utils import dvrk_utils
import tf_conversions.posemath as pm
from scipy.linalg import expm
from filterpy.kalman import KalmanFilter


class visualiser_R():


    def __init__(self):

        self.orientation = None
        self.rotation = None
        self.wrist = None

        self.br = tf.TransformBroadcaster()
        self.pub_t = rospy.Publisher('hand_3D/left/t_psm2', Vector3, queue_size=10)
        self.pub_R = rospy.Publisher('hand_3D/left/R_psm2', Quaternion, queue_size=10)
        

    def listener(self):

        
        rospy.Subscriber('hand_3D/left', param_3D, sub.callback3)
        rospy.Subscriber('controller/2/R', Quaternion, sub.callback4)
        

    def callback2(self, data):

        # Assuming you have a Quaternion message named "quaternion_msg"
        quaternion = [data.x, data.y, data.z, data.w]

        # Convert the quaternion to a rotation matrix
        try:
            self.rotation = R.from_quat(quaternion)
            self.rotation = self.rotation.as_matrix()
        except:
            self.rotation = np.eye(3)
        

    def callback3(self, data):
        
        joint_values = data.joint
        self.wrist=joint_values[0]
        self.wrist=np.array([self.wrist.x,self.wrist.y,self.wrist.z])

    def callback4(self, data):
        
        rotation_controller = [data.x, data.y, data.z, data.w]
        self.rotation = R.from_quat(rotation_controller)
        self.rotation = self.rotation.as_matrix()
        
        print(self.rotation)



        


    



if __name__ == '__main__':

    sub = visualiser_R()
    psm_app = dvrk_utils.DVRK_CTRL("PSM2",0.01)
    
    g = psm_app.get_cp()
    sub.listener()

    rotation_matrix = np.eye(3)
    memory_rotation = np.eye(3)
    arrow_transform = np.eye(4)
    translation = np.array([0,0,0])
    h_coord = np.array([0,0,0,1])
    psm_t=g[0:3,3]


    cam_tf = tf_utils.g2tf(
        tf_utils.cv2vecs2g(
            np.array([1,0,0])*math.radians(-90),np.array([0,0,0])
        ).dot(
            tf_utils.cv2vecs2g(
                np.array([0,1,0])*math.radians(90),np.array([0,-1,0])
            )
        )
    )

    hand_ref_tf = tf_utils.g2tf(
        np.eye(4)
    )
    
    
    try:

        rate = rospy.Rate(30)  # 10 Hz

        while not rospy.is_shutdown():

            if sub.rotation is not None:
                rotation_matrix = sub.rotation
            else:
                rotation_matrix = np.eye(3)
        
            # Create a transformation matrix for the coordinate frame
            hand_transform = np.eye(4)

            hand_transform[:3, :3] = rotation_matrix.dot(
                tf_utils.cv2vecs2g(
                    np.array([1,0,0])*math.radians(0),np.array([0,0,0])
                )[:3,:3]
            )

            hand_transform[:3, 3] = sub.wrist
            hand_tf = tf_utils.g2tf(hand_transform)
            hand_quaternion = np.array([hand_tf.rotation.x,hand_tf.rotation.y,hand_tf.rotation.z,hand_tf.rotation.w])
            print(hand_quaternion)
            t = rospy.Time.now()
            
            sub.br.sendTransform((cam_tf.translation.x,cam_tf.translation.y,cam_tf.translation.z),
                              (cam_tf.rotation.x,cam_tf.rotation.y,cam_tf.rotation.z,cam_tf.rotation.w),
                              t,"camera","map")
            
            sub.br.sendTransform((psm_t[0],psm_t[1],psm_t[2]),
                                 (hand_ref_tf.rotation.x,hand_ref_tf.rotation.y,hand_ref_tf.rotation.z,hand_ref_tf.rotation.w),
                               t,"hand_ref","camera")
            
            sub.br.sendTransform((0,0,0),
                             (hand_tf.rotation.x,hand_tf.rotation.y,hand_tf.rotation.z,hand_tf.rotation.w),
                              t,"hand_rotated","hand_ref")
            
            # Create rotation objects from rotation vectors
            r1 = R.from_quat(np.array([hand_ref_tf.rotation.x, hand_ref_tf.rotation.y, hand_ref_tf.rotation.z, hand_ref_tf.rotation.w]).flatten())
            r2 = R.from_quat(np.array([hand_tf.rotation.x,hand_tf.rotation.y,hand_tf.rotation.z,hand_tf.rotation.w]).flatten())

            # Compute the relative rotation
            rotation_applied = r2 * r1.inv()
            rotation_applied_psm2 = rotation_applied.as_quat()
        
            sub.br.sendTransform((0,0,0),
                             rotation_applied_psm2,
                              t,"psm2_sent","psm2_tip")
            
            sub.pub_t.publish(0,0,0)
            sub.pub_R.publish(rotation_applied_psm2[0],rotation_applied_psm2[1],rotation_applied_psm2[2],rotation_applied_psm2[3])




            rate.sleep()    

    except rospy.ROSInterruptException:
        pass

    finally:
       
        cv2.destroyAllWindows()