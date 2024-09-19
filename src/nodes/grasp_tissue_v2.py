#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped, PointStamped
from ros_numpy.point_cloud2 import pointcloud2_to_array

import math
import numpy as np
from scipy.optimize import minimize

from sitl_dvrk.msg import StringStamped
from utils import tf_utils, pcl_utils, misc_utils, aruco_utils, ik_devel_utils, dvrk_devel_utils

class GRASP_TISSUE(object):
    def __init__(self, node_name, params):
        self.key         = None
        self.ctrd_3d = None
        self.bnd_3d      = None
        self.cnt_3d      = None
        self.g_fbf       = None
        self.g_fbfjaw    = None
        self.grasp_dir   = None
        self.jp_align    = None
        self.jp_grasp    = None
        self.jp_pull     = None
        self.jp_release  = None
        self.update_flag = False
        self.tissue_norm = None
        self.arm = dvrk_devel_utils.DVRK_CTRL(params["arm_name"], params["expected_interval"], node_name)
        self.init_jp = self.arm.get_jp()
        self.arm_ik = ik_devel_utils.dvrk_custom_ik(
            params["calib_fn"], params["Tweight"], params["Rweight"], self.init_jp,
            params["Joffsets"]
        )
        self.g_ecmopencv_ecmdvrk = self.load_tf(params["tf_path"])
        self.grasp_offset = params["grasp_offset"]
        self.align_ratio  = params["align_ratio"]
        self.min_pull_dist = params["min_pull_dist"]
        self.jaw_open_angle = params["jaw_open_angle"]

    def reset(self):
        self.grasp_dir   = None
        self.jp_align    = None
        self.jp_grasp    = None
        self.jp_pull     = None
        self.jp_release  = None
        self.tissue_norm = None

    def load_tf(self, tf_path):
        tf_data = aruco_utils.load_tf_data(tf_path)
        g_ecmdvrk_ecmopencv = np.array(tf_data["g_ecmdvrk_ecmopencv"])
        return tf_utils.ginv(g_ecmdvrk_ecmopencv)
    
    def armtip_offset(self, g_ecmtip_armtip, g_armtip_dt2):
        g_ecmopencv_armtip = self.g_ecmopencv_ecmdvrk.dot(g_ecmtip_armtip)
        return g_armtip_dt2[:3,3] - g_ecmopencv_armtip[:3,3]
    
    def get_grasp_dir(self, cnt_3d, ctrd_3d):
        pca_comps = misc_utils.cnt_axes_3d(cnt_3d)
        tissue_norm = misc_utils.unit_vector(cnt_3d.mean(axis=0) - ctrd_3d)
        pca_x = np.argmax([np.dot(np.array([1,0,0]), pca_comp) for pca_comp in pca_comps])
        grasp_dir = misc_utils.unit_vector(pca_x*self.align_ratio[0] + tissue_norm*self.align_ratio[1])
        return grasp_dir, tissue_norm
    
    def get_tf(self, dir_vec, des_loc):
        z_tf = dir_vec
        y_tf = misc_utils.unit_vector(np.cross(dir_vec, np.array([0, 0, -1])))
        x_tf = misc_utils.unit_vector(np.cross(y_tf, dir_vec))
        # x_tf = misc_utils.unit_vector(np.cross(dir_vec, np.array([0, -1, 0])))
        # y_tf = misc_utils.unit_vector(np.cross(dir_vec, x_tf))
        R_tf = np.vstack((x_tf, y_tf, z_tf)).T
        return tf_utils.gen_g(R_tf, des_loc)
    
    def proj_curve_to_line_obj_func(self, r, curve, pt, min_dist):
        # Ensure r is within bounds
        r = np.clip(r, 0, 1)
        proj_line = misc_utils.proj_curve_to_line(r, curve, pt)

        # Calculate distances between original curve points and projected line points
        distances = np.linalg.norm(curve - proj_line, axis=1)

        # Penalty for distances below the threshold
        penalty_factor = 1000
        penalties = np.where(distances < min_dist, penalty_factor * (min_dist - distances), 0)
        
        # Objective is to minimize the total distance, with penalties for violations
        return np.sum(distances) + np.sum(penalties)
    
    def optimize_ratio(self, init_r, curve, pt, min_dist):
        result = minimize(self.proj_curve_to_line_obj_func, init_r, args=(curve, pt, min_dist),
                        bounds=[(0, 1)], method='L-BFGS-B')
        optimal_r = result.x[0]
        return optimal_r
    
    def get_pull_dir_mag(self, ctrd_3d, bnd_3d, min_pull_dist):
        optimal_r = self.optimize_ratio(0.5, bnd_3d, ctrd_3d, min_pull_dist)
        proj_line = misc_utils.proj_curve_to_line(optimal_r, bnd_3d, ctrd_3d)
        pull_dirs = proj_line - bnd_3d
        pull_dirs_norm = pull_dirs / np.linalg.norm(pull_dirs, axis=1)[:, np.newaxis]
        avg_pull_dir = np.mean(pull_dirs_norm, axis=0)
        avg_pull_dir /= np.linalg.norm(avg_pull_dir)  # Ensure it's a unit vector
        pull_mags = np.linalg.norm(pull_dirs, axis=1)
        avg_pull_mag = np.mean(pull_mags)
        return avg_pull_dir, avg_pull_mag
    
    def tf_align(self, g_ecmtip_armtip, g_armbase_armtip, g_fbf, ctrd_3d, cnt_3d, align_dist):
        grasp_dir, self.tissue_norm = self.get_grasp_dir(cnt_3d, ctrd_3d)
        # self.tip_offset = self.armtip_offset(g_ecmtip_armtip, g_fbf)
        g_ecmopencv_armtip = self.g_ecmopencv_ecmdvrk.dot(g_ecmtip_armtip)
        g_fbf = self.g_ecmopencv_ecmdvrk.dot(g_fbf)
        g_armbase_ecmopencv = g_armbase_armtip.dot(tf_utils.ginv(g_ecmopencv_armtip))
        # T_align = ctrd_3d - tissue_norm_vec*align_dist - self.tip_offset
        # g_align_cv = self.get_tf(tissue_norm_vec, T_align)
        g_align_temp = self.get_tf(grasp_dir, ctrd_3d - grasp_dir*align_dist)
        g_offset = tf_utils.ginv(g_fbf).dot(g_align_temp)
        # self.tip_offset = tf_utils.gdotv(g_align_cv, self.tip_offset)
        return grasp_dir, g_armbase_ecmopencv.dot(g_ecmopencv_armtip.dot(g_offset))
    
    def tf_grasp(self, g_ecmtip_armtip, g_armbase_armtip, g_fbf, grasp_3d, grasp_offset):
        g_ecmopencv_armtip = self.g_ecmopencv_ecmdvrk.dot(g_ecmtip_armtip)
        g_fbf = self.g_ecmopencv_ecmdvrk.dot(g_fbf)
        g_armbase_ecmopencv = g_armbase_armtip.dot(tf_utils.ginv(g_ecmopencv_armtip))
        # self.tip_offset = self.armtip_offset(g_ecmtip_armtip, g_fbf)
        g_grasp_cv = np.copy(g_ecmopencv_armtip)
        g_grasp_cv[:3,3] = grasp_3d + self.grasp_dir*grasp_offset
        g_offset = tf_utils.ginv(g_fbf).dot(g_grasp_cv)
        g_grasp = g_armbase_ecmopencv.dot(g_ecmopencv_armtip.dot(g_offset))
        return g_grasp, g_armbase_ecmopencv.dot(g_ecmopencv_armtip.dot(g_offset))
    
    def tf_pull(self, g_ecmtip_armtip, g_armbase_armtip, ctrd_3d, bnd_3d):
        g_armbase_ecmopencv = g_armbase_armtip.dot(
            tf_utils.ginv(self.g_ecmopencv_ecmdvrk.dot(g_ecmtip_armtip))
        )
        pull_dir, pull_dist = self.get_pull_dir_mag(ctrd_3d, bnd_3d, self.min_pull_dist)
        g_pull = ik_devel_utils.get_tip_pose_jaw(self.jp_grasp, self.arm_ik.arm_calib_data)
        g_pull[:3,3] += tf_utils.gdotv(g_armbase_ecmopencv, pull_dir*pull_dist)
        return g_pull
    
    def key_cb(self, key_msg):
        self.key = key_msg.data
    
    def data_cb(self, cnt_3d_msg, bnd_3d_msg, ctrd_3d_msg, grasp_3d_msg, fbf_cp_msg, fbf_jaw_cp_msg):
        self.cnt_3d      = pcl_utils.xyzarr_to_nparr(pointcloud2_to_array(cnt_3d_msg))
        self.bnd_3d      = pcl_utils.xyzarr_to_nparr(pointcloud2_to_array(bnd_3d_msg))
        self.ctrd_3d     = tf_utils.ptstamped2pt3d(ctrd_3d_msg)
        self.grasp_3d    = tf_utils.ptstamped2pt3d(grasp_3d_msg)
        self.g_fbf       = tf_utils.tfstamped2g(fbf_cp_msg)
        self.g_fbfjaw    = tf_utils.tfstamped2g(fbf_jaw_cp_msg)
        self.update_flag = True

if __name__ == "__main__":
    rospy.init_node("grasp_tissue")
    node_name = rospy.get_name()
    params    = rospy.get_param(node_name)
    try:
        app = GRASP_TISSUE(node_name, params)
        rospy.Subscriber("/keyboard/grasp", StringStamped, app.key_cb)
        topic_list = []
        topic_list.append(message_filters.Subscriber(params["cnt_msg"],     PointCloud2))
        topic_list.append(message_filters.Subscriber(params["bnd_msg"],     PointCloud2))
        topic_list.append(message_filters.Subscriber(params["ctrd_msg"],    PointStamped))
        topic_list.append(message_filters.Subscriber(params["grasp_msg"],   PointStamped))
        topic_list.append(message_filters.Subscriber("/kpt/fbf/psm2_cp",    TransformStamped))
        topic_list.append(message_filters.Subscriber("/kpt/fbf/psm2jaw_cp", TransformStamped))
        ts = message_filters.ApproximateTimeSynchronizer(topic_list, queue_size=10, slop=0.5)
        ts.registerCallback(app.data_cb)
        while not rospy.is_shutdown():
            if app.key == "a":
                if not app.update_flag:
                    rospy.loginfo("Messages not updated yet!")
                    continue
                if app.ctrd_3d is None or app.cnt_3d is None:
                    rospy.loginfo("Tissue has not been detected yet!")
                    continue
                rospy.loginfo("Aligning the forceps before grasping...")
                app.grasp_dir, g_align = app.tf_align(
                    app.arm.get_cp(), app.arm.get_local_cp(), app.g_fbf,
                    app.ctrd_3d, app.cnt_3d, params["align_dist"]
                )
                print(g_align)
                app.arm_ik.target = g_align
                app.jp_align = app.arm_ik.get_goal_jp_jaw(app.arm.get_jp())
                # print(ik_devel_utils.get_tip_pose_jaw(app.jp_align, app.arm_ik.arm_calib_data))
                app.arm.run_jaw_servo_jp(math.radians(0), 1)
                app.arm.run_arm_servo_jp(app.jp_align, 5)
                rospy.loginfo("Finished aligning the forceps!")
            elif app.key == "g":
                if not app.update_flag:
                    continue
                if app.jp_align is None:
                    rospy.loginfo("Please align the forceps before grasping!")
                    continue
                rospy.loginfo("Start grasping the tissue...")
                g_release, g_grasp = app.tf_grasp(
                    app.arm.get_cp(), app.arm.get_local_cp(), app.g_fbf, 
                    app.grasp_3d, app.grasp_offset
                )
                # print(g_grasp)
                app.arm_ik.target = g_grasp
                app.jp_grasp = app.arm_ik.get_goal_jp_jaw(app.arm.get_jp())
                # print(ik_devel_utils.get_tip_pose_jaw(app.jp_grasp, app.arm_ik.arm_calib_data))
                app.arm.run_jaw_servo_jp(math.radians(app.jaw_open_angle), 2)
                app.arm.run_arm_servo_jp(app.jp_grasp, 5)
                app.arm.run_jaw_servo_jp(math.radians(-10), 2)
                app.arm_ik.target = g_release
                app.jp_release = app.arm_ik.get_goal_jp_jaw(app.arm.get_jp())
                rospy.loginfo("Finished grasping the tissue!")
            elif app.key == "p":
                if not app.update_flag:
                    continue
                if app.ctrd_3d is None or app.bnd_3d is None:
                    rospy.loginfo("Tissue has not been detected yet!")
                    continue
                rospy.loginfo("Pulling the tissue...")
                g_pull = app.tf_pull(
                    app.arm.get_cp(), app.arm.get_local_cp(),
                    app.ctrd_3d, app.bnd_3d
                )
                # print(g_pull[:3,3])
                app.arm_ik.target = g_pull
                app.jp_pull = app.arm_ik.get_goal_jp_jaw(app.arm.get_jp())
                # print(ik_devel_utils.get_tip_pose_jaw(app.jp_pull, app.arm_ik.arm_calib_data)[:3,3])
                app.arm.run_arm_servo_jp(app.jp_pull, 5)
                rospy.loginfo("Finished pulling the tissue!")
            elif app.key == "r":
                if app.jp_grasp is None:
                    rospy.loginfo("The arm didn't grasp the tissue yet!")
                    continue
                rospy.loginfo("Start releasing the tissue...")
                app.arm.run_arm_servo_jp(app.jp_release, 5)
                app.arm.run_jaw_servo_jp(math.radians(app.jaw_open_angle), 2)
                app.arm.run_arm_servo_jp(app.jp_align, 5)
                app.arm.run_jaw_servo_jp(math.radians(-10), 2)
                app.reset()
                rospy.loginfo("Finished releasing the tissue!")
            elif app.key == "o":
                rospy.loginfo("Opening the jaw...")
                app.arm.run_jaw_servo_jp(math.radians(app.jaw_open_angle), 2)
                rospy.loginfo("The jaws are opened!")
            elif app.key == "c":
                rospy.loginfo("Closing the jaw...")
                app.arm.run_jaw_servo_jp(math.radians(-10), 2)
                rospy.loginfo("The jaws are closed!")
            elif app.key == "i":
                rospy.loginfo("Moving to its initial position...")
                app.arm.run_arm_servo_jp(app.init_jp, 5)
                rospy.loginfo("Moved to the initial position!")
            app.update_flag = False
            app.key = None
    except Exception as e:
        rospy.logerr(e)
    finally:
        del app