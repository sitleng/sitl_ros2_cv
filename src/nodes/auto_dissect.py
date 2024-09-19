#!/usr/bin/env python3

import rospy
import tf
import message_filters
from ros_numpy.point_cloud2 import pointcloud2_to_array
from sensor_msgs.msg import PointCloud2
from sitl_dvrk.msg import BoolStamped
from geometry_msgs.msg import TransformStamped

from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import os
from tqdm.auto import tqdm
import time
import numpy as np
from utils import tf_utils, pcl_utils, misc_utils, ik_utils, aruco_utils


class DVRK_AUTO_DISSECTION(object):
    def __init__(self,params):
        self.arm_ik = ik_utils.dvrk_custom_ik(params)
        # self.kf     = self.load_kf(np.asarray(params["pv_cov"]),dt=1/params["rate"])
        # self.kf     = self.load_kf(dt=1/params["rate"])
        self.kf     = self.load_kf_vel()
        self.br     = tf.TransformBroadcaster()
        # self.kf     = self.load_kf_pos()
        self.goals  = self.get_goals()
        self.g_ecmopencv_ecmdvrk, self.g_armbase_ecmopencv = self.load_tf(params["arm_name"])
        self.pedal_flag = params["pedal_flag"]
        self.goal_idx = 0
        self.min_dist = params["min_dist"]
        self.dist_eps = params["dist_eps"]
        self.rate = rospy.Rate(params["rate"])
        self.pbar = tqdm(total=self.goals.shape[0])
        self.g_dt2_instjaw      = tf_utils.tfstamped2g(rospy.wait_for_message("/dt2/PCH/instjaw",TransformStamped))
        self.g_ecmopencv_armjaw = self.g_ecmopencv_ecmdvrk.dot(self.arm_ik.arm.get_jaw_cp())
        self.g_armbase_armjaw   = self.arm_ik.arm.get_local_jaw_cp()
        self.pedal_mp_pub = rospy.Publisher("/pedals/write/monopolar",BoolStamped,queue_size=10)
        self.kf_dt2_instjaw_pub = rospy.Publisher("/kf/dt2/PCH/instjaw",TransformStamped,queue_size=10)

    def __del__(self):
        print("Destructing class DVRK_AUTO_DISSECTION...")

    def load_tf(self, arm_name):
        tf_path = "/home/" + os.getlogin() + "/aruco_data/base_tfs.yaml"
        tf_data = aruco_utils.load_tf_data(tf_path)
        g_odom_ecmbase = np.array(tf_data["g_odom_ecmbase"])
        g_psm1base_odom = tf_utils.ginv(np.array(tf_data["g_odom_psm1base"]))
        g_psm2base_odom = tf_utils.ginv(np.array(tf_data["g_odom_psm2base"]))
        g_ecmdvrk_ecmopencv = np.array(tf_data["g_ecmdvrk_ecmopencv"])
        if arm_name == "PSM1":
            g_armbase_ecmbase = g_psm1base_odom.dot(g_odom_ecmbase)
        elif arm_name == "PSM2":
            g_armbase_ecmbase = g_psm2base_odom.dot(g_odom_ecmbase)
        g_ecmbase_ecmtip = tf_utils.tfstamped2g(rospy.wait_for_message("/ECM/custom/local/setpoint_cp",TransformStamped))
        g_armbase_ecmopencv = g_armbase_ecmbase.dot(g_ecmbase_ecmtip).dot(g_ecmdvrk_ecmopencv)
        return tf_utils.ginv(g_ecmdvrk_ecmopencv), g_armbase_ecmopencv

    # def load_kf(self,pv_cov,dt=0.2):
    def load_kf_vel(self,dt=1):
        kf = KalmanFilter (dim_x=6, dim_z=3)
        kf.F = np.array([[1, dt, 0,  0,  0,  0],
                         [0,  1, 0,  0,  0,  0],
                         [0,  0, 1, dt,  0,  0],
                         [0,  0, 0,  1,  0,  0],
                         [0,  0, 0,  0,  1, dt],
                         [0,  0, 0,  0,  0,  1]])
        q = Q_discrete_white_noise(dim=2, dt=dt, var=0.005)
        kf.Q = block_diag(q, q, q)
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0]])
        kf.x = np.zeros((6,))
        kf.x[::2] = np.array([-0.00417268, -0.01104909,  0.0216614 ])
        kf.R = np.array([[ 0.00001569,  0.0000023 ,  0.00000504],
                         [ 0.0000023 ,  0.00000138, -0.00000025],
                         [ 0.00000504, -0.00000025,  0.00000658]])
        kf.P = np.eye(6)*100
        return kf
    
    def load_kf_pos(self):
        kf = KalmanFilter (dim_x=3, dim_z=3)
        kf.F = np.eye(3)
        kf.Q = np.eye(3)*0.005
        kf.H = np.eye(3)
        kf.x = np.array([-0.00417268, -0.01104909,  0.0216614 ])
        kf.R = np.array([[ 0.00001569,  0.0000023 ,  0.00000504],
                         [ 0.0000023 ,  0.00000138, -0.00000025],
                         [ 0.00000504, -0.00000025,  0.00000658]])
        kf.P = np.eye(3)*100
        return kf

    def get_goals(self):
        goals_msg = rospy.wait_for_message("/dt2/goals",PointCloud2)
        goals_xyz = pointcloud2_to_array(goals_msg)
        return pcl_utils.xyzarr_to_nparr(goals_xyz)
    
    def get_dt2_jaw(self):
        return tf_utils.tfstamped2g(
            rospy.wait_for_message("/dt2/PCH/instjaw",TransformStamped)
        )
    
    def hook(self):
        self.pbar.close()
        self.arm_ik.arm.home_init_position(5)
        rospy.loginfo("Finished auto dissection, moving to initial position... ")

    def dt2_cb(self,dt2_instjaw_msg):
        self.g_dt2_instjaw = tf_utils.tfstamped2g(dt2_instjaw_msg)

    def kin_instjaw_cb(self,kin_instjaw_msg):
        self.g_ecmopencv_armjaw = self.g_ecmopencv_ecmdvrk.dot(
            tf_utils.tfstamped2g(kin_instjaw_msg)
        )

    def kin_local_instjaw_cb(self,kin_local_instjaw_msg):
        self.g_armbase_armjaw = tf_utils.tfstamped2g(kin_local_instjaw_msg)

    def ts_callback(self,dt2_instjaw_msg,kin_instjaw_msg,kin_local_instjaw_msg):
        self.t = dt2_instjaw_msg.header.stamp
        self.g_dt2_instjaw = tf_utils.tfstamped2g(dt2_instjaw_msg)
        self.g_ecmopencv_armjaw = self.g_ecmopencv_ecmdvrk.dot(
            tf_utils.tfstamped2g(kin_instjaw_msg)
        )
        self.g_armbase_armjaw = tf_utils.tfstamped2g(kin_local_instjaw_msg)

    def init(self):
        g_dt2_instjaw = self.g_dt2_instjaw
        g_ecmopencv_armjaw = self.g_ecmopencv_armjaw
        g_armbase_armjaw = self.g_armbase_armjaw
        # g_dt2_instjaw = self.get_dt2_jaw()
        # g_ecmopencv_armjaw = self.g_ecmopencv_ecmdvrk.dot(self.arm_ik.arm.get_jaw_cp())
        # g_armbase_armjaw = self.arm_ik.arm.get_local_jaw_cp()
        goal_g = np.copy(g_ecmopencv_armjaw)
        goal_g[:3,3] = g_ecmopencv_armjaw[:3,3] - g_dt2_instjaw[:3,3] + self.goals[self.goal_idx]
        self.arm_ik.target = self.g_armbase_ecmopencv.dot(goal_g)
        self.arm_ik.move_to_goal_jaw(5)
        time.sleep(1)

    def run_kf(self):
        pedal_mp_msg = BoolStamped()
        while not rospy.is_shutdown() and self.goal_idx < self.goals.shape[0]:
            t = self.t
            g_dt2_instjaw = self.g_dt2_instjaw
            g_ecmopencv_armjaw = self.g_ecmopencv_armjaw
            g_armbase_armjaw = self.g_armbase_armjaw
            # g_dt2_instjaw = self.get_dt2_jaw()
            # g_ecmopencv_armjaw = self.g_ecmopencv_ecmdvrk.dot(self.arm_ik.arm.get_jaw_cp())
            # g_armbase_armjaw = self.arm_ik.arm.get_local_jaw_cp()
            self.kf.predict()
            self.kf.update(g_ecmopencv_armjaw[:3,3] - g_dt2_instjaw[:3,3])
            if self.kf.x.shape[0] == 6:
                cur_x = self.kf.x[::2]
            else:
                cur_x = self.kf.x
            kf_dt2_instjaw_msg = tf_utils.g2tfstamped(
                tf_utils.gen_g(g_dt2_instjaw[:3,:3],(g_ecmopencv_armjaw[:3,3]-cur_x).reshape(3,1)),
                t, "ecm_opencv", "kf_dt2_jaw"
            )
            self.kf_dt2_instjaw_pub.publish(kf_dt2_instjaw_msg)
            self.br.sendTransformMessage(kf_dt2_instjaw_msg)
            goal_g = np.copy(g_ecmopencv_armjaw)
            goal_g[:3,3] = cur_x + self.goals[self.goal_idx]
            self.arm_ik.target = self.g_armbase_ecmopencv.dot(goal_g)
            cur_dist = np.linalg.norm(self.arm_ik.target[:3,3]-g_armbase_armjaw[:3,3])
            rospy.loginfo("Current distance to goal: {} (mm)".format(cur_dist*1000))
            if cur_dist > self.min_dist:
                self.arm_ik.target[:3,3] = misc_utils.prop_pt(g_armbase_armjaw[:3,3],self.arm_ik.target[:3,3],self.min_dist)
                self.arm_ik.move_to_goal_jaw(1)
            elif cur_dist < self.dist_eps:
                self.goal_idx += 1
                self.pbar.update()
                rospy.loginfo("Reached Goal {}/{} with distance {} (mm)".format(self.goal_idx,self.goals.shape[0],cur_dist*1000))
            else:
                self.arm_ik.move_to_goal_jaw(1)
            if self.goal_idx > 1 and self.pedal_flag:
                pedal_mp_msg.header.stamp = rospy.Time.now()
                pedal_mp_msg.data = True
                self.pedal_mp_pub.publish(pedal_mp_msg)
            self.rate.sleep()

    def run_no_kf(self):
        while not rospy.is_shutdown() and self.goal_idx < self.goals.shape[0]:
            g_dt2_instjaw = self.g_dt2_instjaw
            g_ecmopencv_armjaw = self.g_ecmopencv_armjaw
            g_armbase_armjaw = self.g_armbase_armjaw
            # g_dt2_instjaw = self.get_dt2_jaw()
            # g_ecmopencv_armjaw = self.g_ecmopencv_ecmdvrk.dot(self.arm_ik.arm.get_jaw_cp())
            # g_armbase_armjaw = self.arm_ik.arm.get_local_jaw_cp()
            goal_g = np.copy(g_ecmopencv_armjaw)
            goal_g[:3,3] = g_ecmopencv_armjaw[:3,3] - g_dt2_instjaw[:3,3] + self.goals[self.goal_idx]
            self.arm_ik.target = self.g_armbase_ecmopencv.dot(goal_g)
            cur_dist = np.linalg.norm(self.arm_ik.target[:3,3]-g_armbase_armjaw[:3,3])
            rospy.loginfo("Current distance to goal: {} (mm)".format(cur_dist*1000))
            if cur_dist > self.min_dist:
                self.arm_ik.target[:3,3] = misc_utils.prop_pt(g_armbase_armjaw[:3,3],self.arm_ik.target[:3,3],self.min_dist)
                self.arm_ik.move_to_goal_jaw(1)
            elif cur_dist < self.dist_eps:
                self.goal_idx += 1
                self.pbar.update()
                rospy.loginfo("Reached Goal {}/{} with distance {} (mm)".format(self.goal_idx,self.goals.shape[0],cur_dist*1000))
            else:
                self.arm_ik.move_to_goal_jaw(1)
            # self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node("auto_dissection")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)

    app = DVRK_AUTO_DISSECTION(params)

    try:
        rospy.loginfo("Start auto dissection...")
        dt2_instjaw = message_filters.Subscriber("/dt2/PCH/instjaw",TransformStamped)
        if params["arm_name"] == "PSM1":
            kin_instjaw = message_filters.Subscriber("/PSM1/custom/jaw/setpoint_cp",TransformStamped)
            kin_local_instjaw = message_filters.Subscriber("/PSM1/custom/local/jaw/setpoint_cp",TransformStamped)
        if params["arm_name"] == "PSM2":
            kin_instjaw = message_filters.Subscriber("/PSM2/custom/jaw/setpoint_cp",TransformStamped)
            kin_local_instjaw = message_filters.Subscriber("/PSM2/custom/local/jaw/setpoint_cp",TransformStamped)
        ts = message_filters.ApproximateTimeSynchronizer(
            [dt2_instjaw,kin_instjaw,kin_local_instjaw],slop=0.01,queue_size=10
        )
        ts.registerCallback(app.ts_callback)
        app.init()
        app.run_kf()
    # try:
        # rospy.loginfo("Start auto dissection...")
        # dt2_instjaw = rospy.Subscriber("/dt2/PCH/instjaw",TransformStamped,app.dt2_cb)
        # if params["arm_name"] == "PSM1":
        #     kin_instjaw = rospy.Subscriber("/PSM1/custom/jaw/setpoint_cp",TransformStamped,app.kin_instjaw_cb)
        #     kin_local_instjaw = rospy.Subscriber("/PSM1/custom/local/jaw/setpoint_cp",TransformStamped,app.kin_local_instjaw_cb)
        # if params["arm_name"] == "PSM2":
        #     kin_instjaw = rospy.Subscriber("/PSM2/custom/jaw/setpoint_cp",TransformStamped,app.kin_instjaw_cb)
        #     kin_local_instjaw = rospy.Subscriber("/PSM2/custom/local/jaw/setpoint_cp",TransformStamped,app.kin_local_instjaw_cb)
        # app.init()
        # app.run()
    except Exception as e:
        app.hook()
        rospy.logerr(e)
    finally:
        app.hook()
        del app