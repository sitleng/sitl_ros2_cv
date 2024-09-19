#!/usr/bin/env python3

# Import ROS libraries
import rospy
from sensor_msgs.msg import CompressedImage, PointCloud2
from cv_bridge import CvBridge
import message_filters
from ros_numpy.point_cloud2 import pointcloud2_to_array

# Import custom libraries
from sitl_dvrk.msg import Dt2KptState
from utils import dt2_utils, pcl_utils, ma_utils

class PUB_KPT_PCH_RAW():
    def __init__(self,params):
        self.br = CvBridge()
        self.params = params
        self.pub_pch  = rospy.Publisher("/kpt/pch/raw", Dt2KptState, queue_size=10)
        self.keypt_predictor = dt2_utils.load_kpt_predictor(
            params['model_path'], params['model_score_thr'], "PCH"
        )
        self.keypt_metadata = dt2_utils.load_kpt_metadata("PCH")

    def __del__(self):
        print("Destructing class PUB_KPT_PCH...")

    def callback(self, pcl_msg, cam1_rect_color_msg):
        pcl_array = pcl_utils.xyzarr_to_nparr(pointcloud2_to_array(pcl_msg))
        img = self.br.compressed_imgmsg_to_cv2(cam1_rect_color_msg)
        pch_kpt_nms, pch_2d = dt2_utils.get_inst_kpts_2d(
            "PCH", img, self.keypt_predictor, self.keypt_metadata, self.params["kpt_score_thr"]
        )
        if pch_2d is None or pch_2d.size == 0:
            rospy.loginfo("Hook not detected!")
            return
        pch_kpt_nms, pch_3d = dt2_utils.win_avg_3d_kpts(
            pch_2d, pch_kpt_nms, pcl_array, self.params["window_size"], self.params["mad_thr"]
        )
        # if pch_3d is None or not dt2_utils.validate_hook_kpts(pch_kpt_nms, pch_3d):
        if pch_3d is None or "CentralScrew" not in pch_kpt_nms:
            rospy.loginfo("Invalid Hook Keypoints!")
            return
        # if pch_3d is None:
        #     rospy.loginfo("Invalid Hook Keypoints!")
        #     return
        pch_msg = ma_utils.gen_dt2kptstate(pch_kpt_nms, pch_2d, pch_3d, rospy.Time.now())
        self.pub_pch.publish(pch_msg)

if __name__ == '__main__':
    rospy.init_node("pub_kpt_pch_raw")
    node_name = rospy.get_name()
    params = rospy.get_param(node_name)
    rospy.loginfo("Start detecting inst tip positions...")
    app = PUB_KPT_PCH_RAW(params)
    try:
        pcl_msg  = message_filters.Subscriber("/ecm/points2",PointCloud2)
        topic_list = [pcl_msg]
        if params["calib_dir"] == "L2R":
            left_rect_color_msg  = message_filters.Subscriber("/ecm/left_rect/image_color",CompressedImage)
            topic_list.append(left_rect_color_msg)
        elif params["calib_dir"] == "R2L":
            right_rect_color_msg = message_filters.Subscriber("/ecm/right_rect/image_color",CompressedImage)
            topic_list.append(right_rect_color_msg)
        ts = message_filters.ApproximateTimeSynchronizer(
                    topic_list, slop=0.02, queue_size=10
        )
        ts.registerCallback(app.callback)
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)
    finally:
        del app