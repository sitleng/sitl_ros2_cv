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

class PUB_KPT_FBF_RAW():
    def __init__(self,params):
        self.br = CvBridge()
        self.params = params
        self.pub_fbf  = rospy.Publisher("/kpt/fbf/raw", Dt2KptState, queue_size=10)
        self.keypt_predictor = dt2_utils.load_kpt_predictor(
            params['model_path'], params['model_score_thr'], "FBF"
        )
        self.keypt_metadata = dt2_utils.load_kpt_metadata("FBF")

    def __del__(self):
        print("Destructing class PUB_KPT_FBF_RAW...")

    def callback(self, pcl_msg, cam1_rect_color_msg):
        pcl_array = pcl_utils.xyzarr_to_nparr(pointcloud2_to_array(pcl_msg))
        img = self.br.compressed_imgmsg_to_cv2(cam1_rect_color_msg)
        fbf_kpt_nms, fbf_2d = dt2_utils.get_inst_kpts_2d(
            "FBF", img, self.keypt_predictor, self.keypt_metadata, self.params["kpt_score_thr"]
        )
        if fbf_2d is None or fbf_2d.size == 0:
            rospy.loginfo("Forceps not detected!")
            return
        fbf_kpt_nms, fbf_3d = dt2_utils.win_avg_3d_kpts(
            fbf_2d, fbf_kpt_nms, pcl_array, self.params["window_size"], self.params["mad_thr"]
        )
        # if fbf_3d is None or not kpt_utils.validate_frcp_kpts(fbf_kpt_nms, fbf_3d):
        if fbf_3d is None:
            rospy.loginfo("Invalid Forceps Keypoints!")
            return
        fbf_msg = ma_utils.gen_dt2kptstate(fbf_kpt_nms, fbf_2d, fbf_3d, rospy.Time.now())
        self.pub_fbf.publish(fbf_msg)

if __name__ == '__main__':
    rospy.init_node("pub_kpt_fbf_raw")
    node_name = rospy.get_name()
    params = rospy.get_param(node_name)
    rospy.loginfo("Start detecting Forceps positions...")
    app = PUB_KPT_FBF_RAW(params)
    try:
        topic_list = []
        topic_list.append(message_filters.Subscriber("/ecm/points2",PointCloud2))
        if params["calib_dir"] == "L2R":
            topic_list.append(message_filters.Subscriber("/ecm/left_rect/image_color",CompressedImage))
        elif params["calib_dir"] == "R2L":
            topic_list.append(message_filters.Subscriber("/ecm/right_rect/image_color",CompressedImage))
        ts = message_filters.ApproximateTimeSynchronizer(
            topic_list, slop=0.02, queue_size=10
        )
        ts.registerCallback(app.callback)
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)
    finally:
        del app