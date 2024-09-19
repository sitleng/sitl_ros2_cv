#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped, PointStamped
from ros_numpy.point_cloud2 import pointcloud2_to_array

import traceback
import math
import numpy as np
from scipy.spatial import KDTree

from sitl_dvrk.msg import StringStamped, Dt2KptState, BoolStamped
from utils import tf_utils, pcl_utils, misc_utils, aruco_utils, ik_devel_utils, dvrk_devel_utils, ma_utils

class FOLLOW_TISSUE_BND(object):
    def __init__(self, node_name, params):
        self.key      = None
        self.bnd_3d   = None
        self.kpt_nms  = None
        self.pch_2d   = None
        self.pch_3d   = None
        self.g_pchjaw = None
        self.pch_update_flag = False
        self.align_jp = None
        self.pch_angle_offset = misc_utils.law_of_cos_ang(0.0049, 0.0036, 0.0018)
        self.arm = dvrk_devel_utils.DVRK_CTRL(params["arm_name"], params["expected_interval"], node_name)
        self.init_jp = self.arm.get_jp()
        self.arm_ik = ik_devel_utils.dvrk_custom_ik(
            params["calib_fn"], params["Tweight"], params["Rweight"], self.init_jp,
            params["Joffsets"]
        )
        self.pedal_flag = params["pedal_flag"]
        self.pedal_mp_msg = BoolStamped()
        self.pedal_mp_pub = rospy.Publisher("/pedals/write/monopolar", BoolStamped, queue_size=10)
        self.g_ecmopencv_ecmdvrk = self.load_tf(params["tf_path"])
        self.pub_cur_traj = rospy.Publisher("/cur_traj", PointStamped, queue_size=10)
        self.cur_idx = 0
        self.success = 0

    def reset(self):
        self.bnd_3d   = None
        self.kpt_nms  = None
        self.pch_2d   = None
        self.pch_3d   = None
        self.g_pchjaw = None
        self.align_jp = None

    def load_tf(self, tf_path):
        tf_data = aruco_utils.load_tf_data(tf_path)
        g_ecmdvrk_ecmopencv = np.array(tf_data["g_ecmdvrk_ecmopencv"])
        return tf_utils.ginv(g_ecmdvrk_ecmopencv)
    
    def armtip_offset(self, g_ecmtip_armtip, g_armtip_dt2):
        g_ecmopencv_armtip = self.g_ecmopencv_ecmdvrk.dot(g_ecmtip_armtip)
        return g_ecmopencv_armtip[:3,3] - g_armtip_dt2[:3,3]
    
    def tf_align_pch(self, g_ecmtip_armtip, g_armbase_armtip, kpt_nms, pch_3d, bnd_3d):
        g_armtip_ecmopencv = tf_utils.ginv(self.g_ecmopencv_ecmdvrk.dot(g_ecmtip_armtip))
        if kpt_nms is None or "TipHook" not in kpt_nms or "CentralHook" not in kpt_nms:
            print("Missing Keypoints...")
            return None
        v_ch_th = pch_3d[kpt_nms.index("TipHook")] - pch_3d[kpt_nms.index("CentralHook")]
        v_bs_bf = bnd_3d[-1] - bnd_3d[0]
        angle = misc_utils.angle_btw_vecs(v_ch_th, v_bs_bf)
        w_rot = misc_utils.unit_vector(np.cross(v_ch_th, v_bs_bf))
        w_rot = tf_utils.gdotv(g_armtip_ecmopencv, w_rot)
        if w_rot[0] > 0:
            angle -= self.pch_angle_offset
        else:
            angle += self.pch_angle_offset
        return g_armbase_armtip.dot(tf_utils.cv2vecs2g(w_rot*angle, np.array([0,0,0])))
    
    def align_bnd(self, kpt_nms, pch_3d, bnd_3d):
        if kpt_nms is None or "TipHook" not in kpt_nms or "CentralHook" not in kpt_nms:
            return bnd_3d
        v_ch_th = pch_3d[kpt_nms.index("TipHook")] - pch_3d[kpt_nms.index("CentralHook")]
        v_bs_bf = bnd_3d[-1] - bnd_3d[0]
        angle = misc_utils.angle_btw_vecs(v_ch_th, v_bs_bf)
        if angle > math.pi/2:
            bnd_3d = bnd_3d[::-1]
            v_bs_bf = bnd_3d[-1] - bnd_3d[0]
        return bnd_3d
    
    def get_cur_traj_pt(self, prev_traj_pt, prev_idx, kpt_nms, bnd_3d, pch_3d, d_thr=7e-3, ang_thr=30):
        cur_traj_pt = prev_traj_pt
        th = pch_3d[kpt_nms.index("TipHook")]
        max_idx = 0
        max_dist = 0
        traj_tree = KDTree(bnd_3d)
        matching_indices = traj_tree.query_ball_point(th, d_thr)
        if matching_indices:
            for idx in matching_indices:
                dist = np.linalg.norm(bnd_3d[idx] - th)
                angle = math.degrees(misc_utils.angle_btw_vecs(bnd_3d[-1]-bnd_3d[0], bnd_3d[idx]-th))
                if dist > max_dist and abs(angle) < ang_thr:
                    max_dist = dist
                    max_idx = idx
                    cur_traj_pt = bnd_3d[idx]
        if max_idx == 0:
            max_idx = prev_idx
        return max_idx, cur_traj_pt
    
    def key_cb(self, key_msg):
        self.key = key_msg.data

    def bnd_cb(self, bnd_3d_msg):
        # self.bnd_3d   = self.align_bnd(
        #     self.kpt_nms, self.pch_3d,
        #     pcl_utils.xyzarr_to_nparr(pointcloud2_to_array(bnd_3d_msg))
        # )
        self.bnd_3d = pcl_utils.xyzarr_to_nparr(pointcloud2_to_array(bnd_3d_msg))
    
    def pch_cb(self, pch_raw_msg, pch_jaw_cp_msg):
        self.kpt_nms  = pch_raw_msg.name
        self.pch_2d   = ma_utils.ma2arr(pch_raw_msg.kpts2d)
        self.pch_3d   = ma_utils.ma2arr(pch_raw_msg.kpts3d)
        self.g_pchjaw = tf_utils.tfstamped2g(pch_jaw_cp_msg)
        self.pch_update_flag = True

if __name__ == "__main__":
    rospy.init_node("follow_tissue_bnd")
    node_name = rospy.get_name()
    params    = rospy.get_param(node_name)
    try:
        app = FOLLOW_TISSUE_BND(node_name, params)
        rospy.Subscriber("/keyboard/follow", StringStamped, app.key_cb)
        rospy.Subscriber(params["bnd_msg"], PointCloud2, app.bnd_cb)
        topic_list = []
        topic_list.append(message_filters.Subscriber("/kpt/pch/raw", Dt2KptState))
        topic_list.append(message_filters.Subscriber("/kpt/pch/psm1jaw_cp", TransformStamped))
        ts = message_filters.ApproximateTimeSynchronizer(topic_list, queue_size=10, slop=0.5)
        ts.registerCallback(app.pch_cb)
        while not rospy.is_shutdown():
            if app.key == "a":
                rospy.loginfo("Aligning the arm before following boundary...")
                if app.bnd_3d is None:
                    rospy.loginfo("Tissue has not been detected yet!")
                    continue
                app.arm_ik.target = app.tf_align_pch(
                    app.arm.get_cp(), app.arm.get_local_cp(), app.kpt_nms, app.pch_3d, app.bnd_3d
                )
                print(app.arm.get_jp())
                app.align_jp = app.arm_ik.get_goal_jp(app.arm.get_jp())
                print(app.align_jp)
                app.arm.run_arm_servo_jp(app.align_jp, 3)
                rospy.loginfo("Finished aligning the hook!")
            elif app.key == "f":
                cur_traj_pt = None
                cur_goal_pt = None
                # if app.align_jp is None:
                #     rospy.loginfo("Align the arm before following the tissue boundary!")
                #     continue
                rospy.loginfo("Start following the tissue boundary...")
                rospy.loginfo("Press b if you want to terminate...")
                while True:
                    if app.key == "b":
                        app.cur_idx = 0

                        break
                    if not app.pch_update_flag or app.bnd_3d is None or "TipHook" not in app.kpt_nms:
                        continue
                    if cur_traj_pt is None:
                        cur_traj_pt = app.bnd_3d[app.cur_idx]
                        cur_goal_pt = app.bnd_3d[-1]
                    # print("dist: {}".format(dist))
                    if np.linalg.norm(cur_traj_pt - app.g_pchjaw[:3,3]) < params["d_eps"]:
                        if np.linalg.norm(cur_goal_pt - app.g_pchjaw[:3,3]) < params["d_eps"]:
                            app.cur_idx = 0
                            cur_traj_pt = app.bnd_3d[app.cur_idx]
                            if app.align_jp is None:
                                app.arm.run_arm_servo_jp(app.init_jp, 3)
                            else:
                                app.arm.run_arm_servo_jp(app.align_jp, 3)
                            break
                        else:
                            app.cur_idx, cur_traj_pt = app.get_cur_traj_pt(
                                cur_traj_pt, app.cur_idx, app.kpt_nms, app.bnd_3d, app.pch_3d
                            )
                            cur_goal_pt = app.bnd_3d[-1]
                            app.success += 1
                            # rospy.loginfo("Current traj index: {}".format(app.cur_idx))
                    cur_traj_msg = tf_utils.pt3d2ptstamped(
                        cur_traj_pt, rospy.Time.now(), "ecm_left"
                    )
                    app.pub_cur_traj.publish(cur_traj_msg)
                    g_armbase_ecmopencv = app.arm.get_local_cp().dot(
                        tf_utils.ginv(app.g_ecmopencv_ecmdvrk.dot(app.arm.get_cp()))
                    )
                    app.arm_ik.target = app.arm.get_local_cp()
                    app.arm_ik.target[:3,3] += tf_utils.gdotv(
                        g_armbase_ecmopencv, 
                        misc_utils.unit_vector(
                            cur_traj_pt - app.g_pchjaw[:3,3]
                        )*params["move_dist"]
                    )
                    if app.cur_idx > 0 and app.pedal_flag:
                        app.pedal_mp_msg.header.stamp = rospy.Time.now()
                        app.pedal_mp_msg.data = True
                        app.pedal_mp_pub.publish(app.pedal_mp_msg)
                    app.arm.run_arm_servo_jp(
                        app.arm_ik.get_goal_jp(app.arm.get_jp()), 0.5
                    )
                    if app.cur_idx > 0 and app.pedal_flag:
                        app.pedal_mp_msg.header.stamp = rospy.Time.now()
                        app.pedal_mp_msg.data = False
                        app.pedal_mp_pub.publish(app.pedal_mp_msg)
                    app.pch_update_flag = False
                rospy.loginfo("Completed following the boundary!")
            elif app.key == "r":
                rospy.loginfo("Moving to the position before following the boundary...")
                app.arm.run_arm_servo_jp(app.align_jp, 5)
            elif app.key == "i":
                rospy.loginfo("Moving to its initial position...")
                app.arm.run_arm_servo_jp(app.init_jp, 5)
            app.key = None
    except Exception as e:
        traceback.print_exc()
    finally:
        del app