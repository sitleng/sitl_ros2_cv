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

class PUB_KPT_FBF_CP():
    def __init__(self,params):
        self.tf_br = tf.TransformBroadcaster()
        self.params = params
        self.pub_kpt_psm2_cp    = rospy.Publisher("/kpt/fbf/psm2_cp", TransformStamped, queue_size=10)
        self.pub_kpt_psm2jaw_cp = rospy.Publisher("/kpt/fbf/psm2jaw_cp", TransformStamped, queue_size=10)
        self.g_psm2tip_psm2jaw = tf_utils.cv2vecs2g(
            np.array([0.0, 0.0, 0.0]), np.array([-0.004, 0.0, 0.019])
        )
        self.g_ecmdvrk_ecmopencv = self.load_tf_data(params["tf_path"])
        # self.g_ecmopencv_ecmdvrk = self.load_tf_data(params["tf_path"])

    def __del__(self):
        print("Destructing class PUB_KPT_FBF_CP...")

    def load_tf_data(self,tf_path):
        tf_data = aruco_utils.load_tf_data(tf_path)
        # g_ecmopencv_ecmdvrk = tf_utils.ginv(np.array(tf_data["g_ecmdvrk_ecmopencv"]))
        # return g_ecmopencv_ecmdvrk
        g_ecmdvrk_ecmopencv = np.array(tf_data["g_ecmdvrk_ecmopencv"]).reshape(4,4)
        return g_ecmdvrk_ecmopencv

    def pub_psm2tip(self, g_psm2_cp):
        kpt_insttip_msg = tf_utils.g2tfstamped(
            g_psm2_cp, rospy.Time.now(),
            self.params["frame_id"],
            self.params["tip_child_frame_id"]
        )
        self.pub_kpt_psm2_cp.publish(kpt_insttip_msg)
        self.tf_br.sendTransformMessage(kpt_insttip_msg)

    def pub_psm2jaw(self, g_psm2jaw_cp):
        kpt_instjaw_msg = tf_utils.g2tfstamped(
            g_psm2jaw_cp, rospy.Time.now(),
            self.params["frame_id"],
            self.params["jaw_child_frame_id"]
        )
        self.pub_kpt_psm2jaw_cp.publish(kpt_instjaw_msg)
        self.tf_br.sendTransformMessage(kpt_instjaw_msg)

    def callback(self, psm2_msg, kpt_fbf_msg):
        g_psm2tip = tf_utils.posestamped2g(psm2_msg)
        g_psm2jaw = copy.deepcopy(g_psm2tip)
        fbf_kpt_nms = kpt_fbf_msg.name
        fbf_kpts_3d = ma_utils.ma2arr(kpt_fbf_msg.kpts3d)
        if "Center" not in fbf_kpt_nms:
            return
        g_psm2tip[:3,3] = tf_utils.gdotp(self.g_ecmdvrk_ecmopencv, fbf_kpts_3d[fbf_kpt_nms.index("Center")])
        g_psm2jaw = g_psm2tip.dot(self.g_psm2tip_psm2jaw)
        # g_psm2jaw[:3,3] = (fbf_kpts_3d[fbf_kpt_nms.index("TipLeft")] + fbf_kpts_3d[fbf_kpt_nms.index("TipRight")])/2
        self.pub_psm2tip(g_psm2tip)
        self.pub_psm2jaw(g_psm2jaw)

if __name__ == '__main__':
    rospy.init_node("pub_kpt_fbf_cp")
    node_name = rospy.get_name()
    params = rospy.get_param(node_name)
    app = PUB_KPT_FBF_CP(params)
    try:
        topic_list = []
        topic_list.append(message_filters.Subscriber("/PSM2/custom/setpoint_cp",PoseStamped))
        topic_list.append(message_filters.Subscriber("/kpt/fbf/raw", Dt2KptState))
        ts = message_filters.ApproximateTimeSynchronizer(
                    topic_list, slop=0.1, queue_size=10
        )
        ts.registerCallback(app.callback)
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)
    finally:
        del app