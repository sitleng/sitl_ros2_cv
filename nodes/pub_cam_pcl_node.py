#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
from cv_bridge import CvBridge
import message_filters

from utils import tf_utils, ecm_utils, cv_cuda_utils, pcl_utils

class PUB_CAM_PCL(Node):
    def __init__(self,params):
        super().__init__("pub_ecm_pcl")
        self.br = CvBridge()
        self.params = params
        self.res = ecm_utils.Resolution(params["resolution"])
        self.Q, self.B, self.f, self.g_LR = self.load_stereo_calib_local(
            self.params,
            self.res.width,
            self.res.height
        )
        self.pub_ecm_pcl = self.create_publisher(PointCloud2, "/ecm/points2", 10)
        disp_msg = message_filters.Subscriber(self, Image, "/ecm/disparity")
        if params["calib_dir"] == "L2R":
            cam_rect_color_msg = message_filters.Subscriber(self, CompressedImage, "/ecm/left_rect/image_color")
        elif params["calib_dir"] == "R2L":
            cam_rect_color_msg = message_filters.Subscriber(self, CompressedImage, "/ecm/right_rect/image_color")
        ts = message_filters.ApproximateTimeSynchronizer(
            [cam_rect_color_msg, disp_msg],
            queue_size=10, slop=params["slop"]
        )
        ts.registerCallback(self.callback)

    def load_stereo_calib_local(self, params, width, height):
        self.get_logger().info("Loading Stereo Calibration Parameters ...")
        Q, R, T, B, f = ecm_utils.load_stereo_calib(params, width, height)
        return Q, B, f, tf_utils.ginv(tf_utils.gen_g(R, T/params["depth_scale"]))

    def callback(self, cam1_rect_color_msg, disp_msg):
        cam1_rect_color = self.br.compressed_imgmsg_to_cv2(cam1_rect_color_msg)
        disp = self.br.imgmsg_to_cv2(disp_msg)
        cam1_rect_color_cuda = cv_cuda_utils.cvmat2gpumat(cam1_rect_color)
        disp_cuda = cv_cuda_utils.cvmat2gpumat(disp)
        ros_pcl = pcl_utils.disp2pcl2_cuda(
            cam1_rect_color_cuda.download(),
            disp_cuda, self.Q, 
            cam1_rect_color_msg.header.frame_id,
            self.params,
            self.get_clock().now().to_msg()
        )
        # self.get_logger().info(f"{ros_pcl.header.stamp}")
        self.pub_ecm_pcl.publish(ros_pcl)
