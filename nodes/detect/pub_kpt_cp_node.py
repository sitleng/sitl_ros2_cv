#!/usr/bin/env python3

# Import open source libraries
import copy
import numpy as np

# Import ROS libraries
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, PoseStamped
import message_filters

# Import custom libraries
from sitl_ros2_interfaces.msg import Dt2KptState
from utils import kpt_utils, ma_utils, ros2_utils, tf_utils

class PUB_KPT_CP(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        self.tf_br = TransformBroadcaster(self)
        self.load_params(params)
        self.g_ecmopencv_ecmdvrk = self.load_tf_data(params["tf_path"])
        qos_profile = ros2_utils.custom_qos_profile(params["queue_size"])
        self.pub_kpt_psm_cp    = self.create_publisher(TransformStamped, "psm_cp"   , qos_profile)
        self.pub_kpt_psmjaw_cp = self.create_publisher(TransformStamped, "psmjaw_cp", qos_profile)
        ts = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(self, PoseStamped, params["psm_topic"], qos_profile=qos_profile),
                message_filters.Subscriber(self, Dt2KptState, "raw", qos_profile=qos_profile)
            ],
            queue_size=params["queue_size"], slop=params["slop"]
        )
        ts.registerCallback(self.callback)

    def load_params(self, params):
        self.inst_name = params["inst_name"]
        self.ct_kpt_nm = params["ct_kpt_nm"]
        self.frame_id = params["frame_id"]
        self.tip_child_frame_id = params["tip_child_frame_id"]
        self.jaw_child_frame_id = params["jaw_child_frame_id"]

    def load_tf_data(self,tf_path):
        tf_data = tf_utils.load_tf_data(tf_path)
        g_ecmopencv_ecmdvrk = tf_utils.ginv(np.array(tf_data["g_ecmdvrk_ecmopencv"]))
        return g_ecmopencv_ecmdvrk

    def pub_psm1tip(self, g_psm1_cp):
        kpt_insttip_msg = tf_utils.g2tfstamped(
            g_psm1_cp,
            ros2_utils.now(self),
            self.frame_id,
            self.tip_child_frame_id
        )
        self.pub_kpt_psm_cp.publish(kpt_insttip_msg)
        self.tf_br.sendTransform(kpt_insttip_msg)

    def pub_psm1jaw(self, g_psm1jaw_cp):
        kpt_instjaw_msg = tf_utils.g2tfstamped(
            g_psm1jaw_cp,
            ros2_utils.now(self),
            self.frame_id,
            self.jaw_child_frame_id
        )
        self.pub_kpt_psmjaw_cp.publish(kpt_instjaw_msg)
        self.tf_br.sendTransform(kpt_instjaw_msg)

    def callback(self, psm_msg, kpt_msg):
        g_psmtip = self.g_ecmopencv_ecmdvrk.dot(tf_utils.posestamped2g(psm_msg))
        g_psmjaw = copy.deepcopy(g_psmtip)
        kpt_nms = kpt_msg.name
        kpts_3d = ma_utils.ma2arr(kpt_msg.kpts3d)
        g_psmtip[:3,3] = kpts_3d[kpt_nms.index(self.ct_kpt_nm)]
        self.pub_psm1tip(g_psmtip)
        if self.inst_name == "PCH":
            g_psmjaw = kpt_utils.pch_g_pcmjaw(
                kpt_nms,
                kpts_3d,
                g_psmtip,
                g_psmjaw,
                tf_utils.g_psm1tip_psm1jaw
            )
        elif self.inst_name == "FBF":
            g_psmjaw = g_psmtip.dot(tf_utils.g_psm2tip_psm2jaw)
        self.pub_psm1jaw(g_psmjaw)
