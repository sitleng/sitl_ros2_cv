#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
import message_filters
from utils import tf_utils, ecm_utils, cv_cuda_utils, pcl_utils

class PUB_ECM_PCL():
    def __init__(self,params):
        self.br = CvBridge()
        self.params = params
        self.left_info, self.right_info = self.load_caminfo()
        self.Q, self.B, self.f, self.g_LR = self.load_stereo_calib_local(
            self.params,
            self.left_info.width,
            self.left_info.height
        )
        self.pub_ecm_pcl  = rospy.Publisher("/ecm/points2",PointCloud2,queue_size=1)
        self.cam1_rect_color = None

    def load_caminfo(self):
        rospy.loginfo("Loading Camera Info ...")
        left_info  = rospy.wait_for_message("/ecm/left/camera_info",CameraInfo)
        right_info = rospy.wait_for_message("/ecm/right/camera_info",CameraInfo)
        return left_info, right_info

    def load_stereo_calib_local(self,params,width,height):
        rospy.loginfo("Loading Stereo Calibration Parameters ...")
        Q, R, T, B, f = ecm_utils.load_stereo_calib(params,width,height)
        return Q, B, f, tf_utils.ginv(tf_utils.gen_g(R,T/params["depth_scale"]))

    def callback(self, cam1_rect_color_msg, disp_msg):
        cam1_rect_color = self.br.compressed_imgmsg_to_cv2(cam1_rect_color_msg)
        disp = self.br.imgmsg_to_cv2(disp_msg)
        cam1_rect_color_cuda = cv_cuda_utils.cvmat2gpumat(cam1_rect_color)
        disp_cuda = cv_cuda_utils.cvmat2gpumat(disp)
        ros_pcl = pcl_utils.disp2pcl2_cuda(
            cam1_rect_color_cuda.download(),
            disp_cuda, self.Q, 
            cam1_rect_color_msg.header.frame_id,
            self.params
        )
        ros_pcl.header.stamp = rospy.Time.now()
        self.pub_ecm_pcl.publish(ros_pcl)

if __name__ == '__main__':
    rospy.init_node("pub_ecm_pcl")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)
    try:
        app = PUB_ECM_PCL(params)
        rospy.loginfo("Start Publishing Point Clouds ...")
        disp_msg = message_filters.Subscriber("/ecm/disparity", Image)
        if params["calib_dir"] == "L2R":
            left_rect_color_msg  = message_filters.Subscriber("/ecm/left_rect/image_color", CompressedImage)
            ts                   = message_filters.ApproximateTimeSynchronizer(
                [left_rect_color_msg,disp_msg],
                slop=0.01, queue_size=10
            )
        elif params["calib_dir"] == "R2L":
            right_rect_color_msg = message_filters.Subscriber("/ecm/right_rect/image_color", CompressedImage)
            ts                   = message_filters.ApproximateTimeSynchronizer(
                [right_rect_color_msg,disp_msg],
                slop=0.01, queue_size=10
            )
        ts.registerCallback(app.callback)
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)