#!/usr/bin/env python

import rospy
from sitl_dvrk.msg import param_3D
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Float64
import cv2
from geometry_msgs.msg import Vector3
import math
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Quaternion


class sub_3DHand():

    def __init__(self):

        self.angle_norm = None
        self.angle_norm_yaw = None
        self.image = None 

        self.orientation_palm = None
        self.orientation_grasp = None
        self.rotation = None
        
        self.ref_vec=np.array([0,1,0])
        
        self.pub_angle = rospy.Publisher('hand_3D/left/angle', Float64, queue_size=10)
        self.rotation_matrix = rospy.Publisher('hand_3D/left/R', Quaternion, queue_size=10)
        
        self.ref_frame=np.array([[ 0.00614022, -0.09101295, -0.03463632], 
                                 [ 0.02463714, -0.0810517,  -0.04615099]])
        
        self.v3=self.find_normal(self.ref_frame[0,:],self.ref_frame[1,:])
        self.ref_frame=np.concatenate((self.ref_frame,self.v3[None,:]),axis=0)
        


    def compute_rotation_matrix(self, f1, f2):

        # Compute the cross-correlation matrix
        H = np.dot(f2.T, f1)

        # Perform Singular Value Decomposition
        U, _, Vt = np.linalg.svd(H)

        # Compute the rotation matrix
        rotation = np.dot(Vt.T, U.T)
        
        return rotation


    def mean_rotation_vectors(self,v1,v2,v3):
        
        #key point more to the left of the map => first
        rotation1=self.find_rotation_vector(v1,v2)
        rotation2=self.find_rotation_vector(v3,v1)
        rotation3=self.find_rotation_vector(v3,v2)
        rotation_matrix=np.array([rotation1,rotation2,rotation3])
        rotation_mean = np.mean(rotation_matrix,axis=0)

        return rotation_mean
    

    def find_normal(self,v1, v2):

        # Normalize input vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        # Calculate rotation axis (cross product)
        v3 = np.cross(v1_norm, v2_norm)
        
        return v3
        

    def find_rotation_vector(self,v1, v2):

        # Normalize input vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        # Calculate rotation axis (cross product)
        axis = np.cross(v1_norm, v2_norm)
        
        # Calculate rotation angle (dot product)
        angle = np.arccos(np.dot(v1_norm, v2_norm))
        #rotation = R.from_rotvec(angle * axis)
        rotation = R.from_rotvec(axis)

        return rotation.as_quat()
    


    def calculate_distance(self,x1, y1, z1, x2, y2, z2):

        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

        return distance



    def compute_angle_orientation(self,joint_values):
        
        wrist=joint_values[0]
        thumb=joint_values[4]
        index=joint_values[8]

        index_mcp=joint_values[5]
        palm_left=joint_values[9]
        palm_right=joint_values[13]

        thumb=[ thumb.x,thumb.y,thumb.z ]
        index=[ index.x,index.y,index.z ]
        wrist=[ wrist.x,wrist.y,wrist.z ]

        palm_left=[palm_left.x,palm_left.y,palm_left.z]
        palm_right=[palm_right.x,palm_right.y,palm_right.z]
        index_mcp=[index_mcp.x,index_mcp.y,index_mcp.z]
        
        vector1 = np.array( palm_left ) - np.array( wrist )
        vector2 = np.array( palm_right ) - np.array( wrist )


        ############################################ Distance ###################################################

        distance=self.calculate_distance(thumb[0],thumb[1],thumb[2],index[0],index[1],index[2])
        angle_in_degrees = distance

        ############################################ Rotation Hand (obsolete) ##################################

        #self.rotation=self.mean_rotation_vectors(vector1,vector2,vector3)
        frame=np.array([vector1,vector2])
        v3_frame=self.find_normal(frame[0,:],frame[1,:])
        frame=np.concatenate((frame,v3_frame[None,:]),axis=0)
        
        self.rotation=self.compute_rotation_matrix( self.ref_frame,frame )
        self.rotation=R.from_matrix(self.rotation)
        self.rotation=self.rotation.as_quat()
        
        return angle_in_degrees,self.rotation


    def rescale_value(self, original_value, max_old, min_old,min_new,max_new):

            rescaled_value = ((original_value - min_old) * (max_new - min_new)) / (max_old - min_old) + min_new

            if rescaled_value > max_new:
                rescaled_value = max_new

            elif rescaled_value < min_new:
                rescaled_value = min_new

            return rescaled_value


    def listener(self):
         
        rospy.init_node('joint_listener', anonymous=True)
        rospy.Subscriber('hand_3D/left', param_3D, sub.callback)
        rospy.spin()


    def callback(self,data):

        joint_values = data.joint
            
        angle,self.rotation=self.compute_angle_orientation(joint_values)
        self.angle_norm = self.rescale_value(angle, 0.16, 0.2,-5,55)
        
        if self.angle_norm is not None:
            self.pub_angle.publish(self.angle_norm)

        if self.rotation is not None:
            self.rotation_matrix.publish(self.rotation[0],self.rotation[1],self.rotation[2],self.rotation[3])

if __name__ == '__main__':

    sub=sub_3DHand()
    sub.listener()
    


        

    