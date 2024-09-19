#!/usr/bin/env python3

# Import open source libraries
import copy
import numpy as np

# Import ROS libraries
import rospy
import tf
from geometry_msgs.msg import TransformStamped, PoseStamped
import message_filters

# Import custom libraries
from sitl_dvrk.msg import Dt2KptState
from utils import aruco_utils, tf_utils, ma_utils

class PUB_KPT_PCH_CP():
    def __init__(self,params):
        self.tf_br = tf.TransformBroadcaster()
        self.params = params
        self.pub_kpt_psm1_cp    = rospy.Publisher("/kpt/pch/psm1_cp", TransformStamped, queue_size=10)
        self.pub_kpt_psm1jaw_cp = rospy.Publisher("/kpt/pch/psm1jaw_cp", TransformStamped, queue_size=10)
        self.g_ecmopencv_ecmdvrk = self.load_tf_data(params["tf_path"])
        self.g_psmtip_psmjaw = tf_utils.cv2vecs2g(np.array([0.0,0.0,0.0]),np.array([-0.005, -0.0025, 0.0147]))
        self.offset = None
        self.count = 0

    def __del__(self):
        print("Destructing class PUB_KPT_INSTTIP...")

    def load_tf_data(self,tf_path):
        tf_data = aruco_utils.load_tf_data(tf_path)
        g_ecmopencv_ecmdvrk = tf_utils.ginv(np.array(tf_data["g_ecmdvrk_ecmopencv"]))
        return g_ecmopencv_ecmdvrk

    def pub_psm1tip(self, g_psm1_cp):
        kpt_insttip_msg = tf_utils.g2tfstamped(
            g_psm1_cp, rospy.Time.now(),
            self.params["frame_id"],
            self.params["tip_child_frame_id"]
        )
        self.pub_kpt_psm1_cp.publish(kpt_insttip_msg)
        self.tf_br.sendTransformMessage(kpt_insttip_msg)

    def pub_psm1jaw(self, g_psm1jaw_cp):
        kpt_instjaw_msg = tf_utils.g2tfstamped(
            g_psm1jaw_cp, rospy.Time.now(),
            self.params["frame_id"],
            self.params["jaw_child_frame_id"]
        )
        self.pub_kpt_psm1jaw_cp.publish(kpt_instjaw_msg)
        self.tf_br.sendTransformMessage(kpt_instjaw_msg)

    def callback(self, psm1_msg, kpt_pch_msg):
        g_psm1tip = self.g_ecmopencv_ecmdvrk.dot(tf_utils.posestamped2g(psm1_msg))
        g_psm1jaw = copy.deepcopy(g_psm1tip)
        pch_kpt_nms = kpt_pch_msg.name
        pch_kpts_3d = ma_utils.ma2arr(kpt_pch_msg.kpts3d)
        g_psm1tip[:3,3] = pch_kpts_3d[pch_kpt_nms.index("CentralScrew")]
        if "TipHook" in pch_kpt_nms:
            # Tip position based on the point cloud
            g_psm1jaw[:3,3] = pch_kpts_3d[pch_kpt_nms.index("TipHook")]
            temp = np.linalg.norm(g_psm1tip[:3,3] - g_psm1jaw[:3,3])
            if temp < 0.01 or temp > 0.02:
                if self.offset is None:
                    g_psm1jaw = g_psm1tip.dot(self.g_psmtip_psmjaw)
                else:
                    g_psm1jaw[:3,3] = g_psm1tip[:3,3] + self.offset/self.count
            else:
                g_offset = tf_utils.ginv(g_psm1tip).dot(g_psm1jaw)
                if self.offset is None:
                    self.offset = g_offset[:3,3]
                else:
                    self.offset += g_offset[:3,3]
                self.count += 1
        else:
            if self.offset is None:
                g_psm1jaw = g_psm1tip.dot(self.g_psmtip_psmjaw)
            else:
                g_psm1jaw[:3,3] = g_psm1tip[:3,3] + self.offset/self.count
        self.pub_psm1tip(g_psm1tip)
        self.pub_psm1jaw(g_psm1jaw)

if __name__ == '__main__':
    rospy.init_node("pub_kpt_pch_cp")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)
    rospy.loginfo("Start detecting inst tip positions...")
    app = PUB_KPT_PCH_CP(params)
    try:
        topic_list = []
        topic_list.append(message_filters.Subscriber("/PSM1/custom/setpoint_cp",PoseStamped))
        topic_list.append(message_filters.Subscriber("/kpt/pch/raw", Dt2KptState))
        ts = message_filters.ApproximateTimeSynchronizer(
                    topic_list, slop=0.1, queue_size=10
        )
        ts.registerCallback(app.callback)
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)
    finally:
        print(app.offset/app.count)
        del app