#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped, PointStamped
from ros_numpy.point_cloud2 import pointcloud2_to_array

import math
import numpy as np

from sitl_dvrk.msg import StringStamped
from utils import tf_utils, pcl_utils, misc_utils, aruco_utils, ik_devel_utils, dvrk_devel_utils

class GRASP_TISSUE(object):
    def __init__(self, node_name, params):
        self.key         = None
        self.tissue_c_3d = None
        self.bnd_3d      = None
        self.g_fbf       = None
        self.g_fbfjaw    = None
        self.w           = None
        self.jp_bf_grasp = None
        self.jp_af_grasp = None
        self.jp_grasp    = None
        self.jp_release  = None
        self.update_flag = False
        self.arm = dvrk_devel_utils.DVRK_CTRL(params["arm_name"], params["expected_interval"], node_name)
        self.init_jp = self.arm.get_jp()
        self.arm_ik = ik_devel_utils.dvrk_custom_ik(
            params["calib_fn"], params["Tweight"], params["Rweight"], self.init_jp,
            params["Joffsets"]
        )
        self.g_ecmopencv_ecmdvrk = self.load_tf(params["tf_path"])
        self.dist2bfgrasp = params["dist2bfgrasp"]
        self.dist2grasp = params["dist2grasp"]

    def reset(self):
        self.w           = None
        self.jp_bf_grasp = None
        self.jp_af_grasp = None
        self.jp_grasp    = None
        self.jp_release  = None

    def load_tf(self, tf_path):
        tf_data = aruco_utils.load_tf_data(tf_path)
        g_ecmdvrk_ecmopencv = np.array(tf_data["g_ecmdvrk_ecmopencv"])
        return tf_utils.ginv(g_ecmdvrk_ecmopencv)
    
    def armtip_offset(self, g_ecmtip_armtip, g_armtip_dt2):
        g_ecmopencv_armtip = self.g_ecmopencv_ecmdvrk.dot(g_ecmtip_armtip)
        return g_ecmopencv_armtip[:3,3] - g_armtip_dt2[:3,3]

    def tf_bf_af_grasp(self, g_ecmtip_armtip, g_armbase_armtip, g_fbf, g_fbfjaw, w, 
                         tip_offset, tissue_c_3d, dist2bfgrasp, dist2afgrasp):
        g_armtip_ecmopencv = tf_utils.ginv(self.g_ecmopencv_ecmdvrk.dot(g_ecmtip_armtip))
        v_fbf = g_fbf[:3,3] - g_fbfjaw[:3,3]
        angle = misc_utils.angle_btw_vecs(v_fbf, w)
        w_rot = tf_utils.gdotv(g_armtip_ecmopencv, misc_utils.unit_vector(np.cross(v_fbf, w)))
        g_bf  = g_armbase_armtip.dot(tf_utils.cv2vecs2g(w_rot*angle, np.array([0,0,0])))
        g_af  = np.copy(g_bf)
        g_armbase_ecmopencv = g_armbase_armtip.dot(g_armtip_ecmopencv)
        t_bf = dist2bfgrasp/np.linalg.norm(w)
        t_af = dist2afgrasp/np.linalg.norm(w)
        g_bf[:3,3] = tf_utils.gdotp(g_armbase_ecmopencv, tissue_c_3d + t_bf*w + tip_offset)
        g_af[:3,3] = tf_utils.gdotp(g_armbase_ecmopencv, tissue_c_3d + t_af*w + tip_offset)
        return g_bf, g_af
    
    def tf_grasp(self, g_ecmtip_armtip, g_armbase_armtip, w, tip_offset, tissue_c_3d, dist2grasp):
        g_armbase_ecmopencv = g_armbase_armtip.dot(
            tf_utils.ginv(self.g_ecmopencv_ecmdvrk.dot(g_ecmtip_armtip))
        )
        # Regarding the length of the forceps jaws.
        p2 = tf_utils.gdotp(g_armbase_ecmopencv, tissue_c_3d + (dist2grasp/np.linalg.norm(w))*w + tip_offset)
        g_release = np.copy(g_armbase_armtip)
        g_release[:3,3] = p2 + tf_utils.gdotv(g_armbase_armtip, misc_utils.unit_vector(np.array([1,0,-1]))*0.005)
        g_grasp = np.copy(g_armbase_armtip)
        g_grasp[:3,3] = p2 + tf_utils.gdotv(g_armbase_armtip, misc_utils.unit_vector(np.array([-1,0,1]))*0.005)
        return g_release, g_grasp
    
    def key_cb(self, key_msg):
        self.key = key_msg.data
    
    def data_cb(self, tissue_c_3d_msg, bnd_3d_msg, fbf_cp_msg, fbf_jaw_cp_msg):
        self.tissue_c_3d = tf_utils.ptstamped2pt3d(tissue_c_3d_msg)
        self.bnd_3d      = pcl_utils.xyzarr_to_nparr(pointcloud2_to_array(bnd_3d_msg))
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
        topic_list.append(message_filters.Subscriber(params["ctrd_msg"],    PointStamped))
        topic_list.append(message_filters.Subscriber(params["bnd_msg"],     PointCloud2))
        topic_list.append(message_filters.Subscriber("/kpt/fbf/psm2_cp",    TransformStamped))
        topic_list.append(message_filters.Subscriber("/kpt/fbf/psm2jaw_cp", TransformStamped))
        ts = message_filters.ApproximateTimeSynchronizer(topic_list, queue_size=10, slop=0.02)
        ts.registerCallback(app.data_cb)
        while not rospy.is_shutdown():
            if app.key == "a":
                if app.tissue_c_3d is None or app.bnd_3d is None:
                    rospy.loginfo("Tissue has not been detected yet!")
                    continue
                rospy.loginfo("Aligning the forceps before grasping...")
                app.w = misc_utils.orthogonal_vector(app.tissue_c_3d, app.bnd_3d)
                tip_offset = app.armtip_offset(app.arm.get_cp(), app.g_fbf)
                g_bf_grasp, g_af_grasp = app.tf_bf_af_grasp(
                    app.arm.get_cp(), app.arm.get_local_cp(), app.g_fbf, app.g_fbfjaw, app.w,
                    tip_offset, app.tissue_c_3d, params["dist2bfgrasp"], params["dist2afgrasp"]
                )
                app.arm_ik.target = g_bf_grasp
                app.jp_bf_grasp = app.arm_ik.get_goal_jp(app.arm.get_jp())
                app.arm.run_jaw_servo_jp(math.radians(0), 1)
                app.arm.run_arm_servo_jp(app.jp_bf_grasp, 5)
                app.arm_ik.target = g_af_grasp
                app.jp_af_grasp = app.arm_ik.get_goal_jp(app.arm.get_jp())
                rospy.loginfo("Finished aligning the forceps!")
            elif app.key == "g":
                if app.jp_bf_grasp is None:
                    rospy.loginfo("Pre-locate the forceps before grasping!")
                    continue
                if not app.update_flag:
                    continue
                rospy.loginfo("Start grasping the tissue...")
                tip_offset = app.armtip_offset(app.arm.get_cp(), app.g_fbf)
                g_release, g_grasp = app.tf_grasp(
                    app.arm.get_cp(), app.arm.get_local_cp(), app.w, tip_offset, 
                    app.tissue_c_3d, app.dist2grasp
                )
                app.arm_ik.target = g_grasp
                app.jp_grasp = app.arm_ik.get_goal_jp(app.arm.get_jp())
                app.arm.run_jaw_servo_jp(math.radians(80), 2)
                app.arm.run_arm_servo_jp(app.jp_grasp, 5)
                app.arm.run_jaw_servo_jp(math.radians(-10), 2)
                app.arm.run_arm_servo_jp(app.jp_af_grasp, 5)
                app.arm_ik.target = g_release
                app.jp_release = app.arm_ik.get_goal_jp(app.arm.get_jp())
                rospy.loginfo("Finished grasping the tissue!")
            elif app.key == "r":
                if app.jp_grasp is None:
                    rospy.loginfo("The arm didn't grasp the tissue yet!")
                    continue
                rospy.loginfo("Start releasing the tissue...")
                app.arm.run_arm_servo_jp(app.jp_release, 5)
                app.arm.run_jaw_servo_jp(math.radians(80), 2)
                app.arm.run_arm_servo_jp(app.jp_bf_grasp, 5)
                app.arm.run_jaw_servo_jp(math.radians(-10), 2)
                app.reset()
                rospy.loginfo("Finished releasing the tissue!")
            elif app.key == "o":
                rospy.loginfo("Opening the jaw...")
                app.arm.run_jaw_servo_jp(math.radians(80), 2)
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