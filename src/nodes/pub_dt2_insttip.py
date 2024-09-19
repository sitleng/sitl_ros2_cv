#!/usr/bin/env python3

# Import open source libraries
import math
import numpy as np

# Import ROS libraries
import rospy
import tf
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CompressedImage, PointCloud2
from cv_bridge import CvBridge
import message_filters
from ros_numpy.point_cloud2 import pointcloud2_to_array

# Import custom libraries
from utils import dt2_utils, aruco_utils, tf_utils, pcl_utils

class PUB_DT2_INSTTIP():
    def __init__(self,params):
        self.br = CvBridge()
        self.tf_br = tf.TransformBroadcaster()
        self.params = params
        if params["psm1_inst"]:
            self.pub_psm1_insttip  = rospy.Publisher(
                "/dt2/{}/insttip".format(params["psm1_inst"]),
                TransformStamped,
                queue_size=10
            )
            self.pub_psm1_instjaw  = rospy.Publisher(
                "/dt2/{}/instjaw".format(params["psm1_inst"]),
                TransformStamped,
                queue_size=10
            )
        if params["psm2_inst"]:
            self.pub_psm2_insttip  = rospy.Publisher(
                "/dt2/{}/insttip".format(params["psm2_inst"]),
                TransformStamped,
                queue_size=10
            )
            self.pub_psm2_instjaw  = rospy.Publisher(
                "/dt2/{}/instjaw".format(params["psm2_inst"]),
                TransformStamped,
                queue_size=10
            )
        self.g_tipe_tipc = tf_utils.cv2vecs2g(np.array([0.0,0.0,0.0]),np.array([-0.003,-0.003,0.006]))
        self.g_psmtip_psmjaw = tf_utils.cv2vecs2g(np.array([0.0,0.0,0.0]),np.array([0.0015,-0.003,0.015]))
        self.g_ecmopencv_ecmdvrk = self.load_tf_data(params["tf_path"])
        self.keypt_predictor = dt2_utils.load_keypt_predictor(
            params['model_path'], params['score_thresh']
        )
        self.keypt_metadata = dt2_utils.load_keypt_metadata()

    def __del__(self):
        print("Destructing class PUB_DT2_INSTTIP...")

    def load_tf_data(self,tf_path):
        tf_data = aruco_utils.load_tf_data(tf_path)
        g_ecmopencv_ecmdvrk = tf_utils.ginv(np.array(tf_data["g_ecmdvrk_ecmopencv"]))
        return g_ecmopencv_ecmdvrk
    
    def pub_msgs(self,g_psm1tip):
        dt2_insttip_msg = tf_utils.g2tfstamped(
            g_psm1tip, rospy.Time.now(),
            self.params["frame_id"],
            self.params["tip_child_frame_id"]
        )
        self.pub_psm1_insttip.publish(dt2_insttip_msg)
        self.tf_br.sendTransformMessage(dt2_insttip_msg)

        g_psm1jaw = g_psm1tip.dot(self.g_psmtip_psmjaw)
        dt2_instjaw_msg = tf_utils.g2tfstamped(
            g_psm1jaw, rospy.Time.now(),
            self.params["frame_id"],
            self.params["jaw_child_frame_id"]
        )
        self.pub_psm1_instjaw.publish(dt2_instjaw_msg)
        self.tf_br.sendTransformMessage(dt2_instjaw_msg)

    def psm1_only_inst_cb(self,pcl_msg,psm1_msg,cam1_rect_color_msg):
        img = self.br.compressed_imgmsg_to_cv2(cam1_rect_color_msg)
        g_psm1tip = self.g_ecmopencv_ecmdvrk.dot(tf_utils.tfstamped2g(psm1_msg))
        pcl_xyz   = pointcloud2_to_array(pcl_msg)
        pcl_array = pcl_utils.xyzarr_to_nparr(pcl_xyz)
        cur_tipc, cur_tipe = dt2_utils.extract_single_insttip_2d(
            self.params["psm1_inst"], img,
            self.keypt_predictor, self.keypt_metadata
        )
        if cur_tipc is not None and cur_tipe is not None:
            rospy.loginfo("Both kpts detected...")
            cur_tipc_3d_pt = pcl_utils.win_avg_3d_pt(cur_tipc,pcl_array,self.params["window_size"])
            cur_tipe_3d_pt = pcl_utils.win_avg_3d_pt(cur_tipe,pcl_array,self.params["window_size"])
            dist = np.linalg.norm(cur_tipc_3d_pt-cur_tipe_3d_pt)
            rospy.loginfo("dist: {}".format(dist))
            if dist > 0.02 or dist < 1e-4:
                return
                # g_psm1tip[:3,3] = cur_tipe_3d_pt
                # g_psm1tip = g_psm1tip.dot(self.g_tipe_tipc)
            g_psm1tip[:3,3] = cur_tipc_3d_pt
            self.pub_msgs(g_psm1tip)
        elif cur_tipc is not None and cur_tipe is None:
            # rospy.loginfo("Only Center detected...")
            cur_tipc_3d_pt = pcl_utils.win_avg_3d_pt(cur_tipc,pcl_array,self.params["window_size"])
            g_psm1tip[:3,3] = cur_tipc_3d_pt
            self.pub_msgs(g_psm1tip)
        elif cur_tipc is None and cur_tipe is not None:
            rospy.loginfo("Only Edge detected...")
        #     cur_tipe_3d_pt = pcl_utils.win_avg_3d_pt(cur_tipe,pcl_array,self.params["window_size"])
        #     g_psm1tip[:3,3] = cur_tipe_3d_pt + self.T_tipc_tipe
        #     self.pub_msgs(g_psm1tip)
        else:
            rospy.loginfo("No kpts detected...")

    def psm1_psm2_inst_cb(self):
        return False

if __name__ == '__main__':
    rospy.init_node("pub_dt2_insttip")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)
    rospy.loginfo("Start detecting inst tip positions...")
    app = PUB_DT2_INSTTIP(params)
    try:
        pcl_msg  = message_filters.Subscriber("/ecm/points2",PointCloud2)
        topic_list = [pcl_msg]
        if params["psm1_inst"]:
            psm1_msg = message_filters.Subscriber("/PSM1/custom/setpoint_cp",TransformStamped)
            topic_list.append(psm1_msg)
        if params["psm2_inst"]:
            psm2_msg = message_filters.Subscriber("/PSM2/custom/setpoint_cp",TransformStamped)
            topic_list.append(psm2_msg)
        if params["calib_dir"] == "L2R":
            left_rect_color_msg  = message_filters.Subscriber("/ecm/left_rect/image_color",CompressedImage)
            topic_list.append(left_rect_color_msg)
        elif params["calib_dir"] == "R2L":
            right_rect_color_msg = message_filters.Subscriber("/ecm/right_rect/image_color",CompressedImage)
            topic_list.append(right_rect_color_msg)
        ts = message_filters.ApproximateTimeSynchronizer(
                    topic_list, slop=0.02, queue_size=10
        )
        if len(topic_list) == 3 and psm1_msg in topic_list:
            ts.registerCallback(app.psm1_only_inst_cb)
        elif len(topic_list) == 4:
            ts.registerCallback(app.psm1_psm2_inst_cb)
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)
    finally:
        del app