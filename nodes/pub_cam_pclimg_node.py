#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from utils import tf_utils, ecm_utils, pcl_utils

class PUB_CAM_PCLIMG(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        self.params = params
        self.br = CvBridge()

        # Initialize stuffs for 3D reconstruction
        self.res = ecm_utils.Resolution(params["resolution"])
        self.Q, self.B, self.f, self.g_LR = self.load_stereo_calib_local(
            self.params,
            self.res.width,
            self.res.height
        )

        # Publishers
        self.pub_cam_3d_pclimg = self.create_publisher(Image, "pclimg", params["queue_size"])

        # Subscribers
        self.sub_disp = self.create_subscription(
            Image, "disparity", self.callback, params["queue_size"]
        )

    def load_stereo_calib_local(self, params, width, height):
        self.get_logger().info("Loading Stereo Calibration Parameters ...")
        Q, R, T, B, f = ecm_utils.load_stereo_calib(params, width, height)
        return Q, B, f, tf_utils.ginv(tf_utils.gen_g(R, T/params["depth_scale"]))

    def callback(self, disp_msg):
        disp = self.br.imgmsg_to_cv2(disp_msg)
        self.get_logger().info(f"{disp.dtype}", once=True)
        pclimg = pcl_utils.disp2pclimg(disp, self.Q, self.params)
        # pclimg = pcl_utils.disp2pclimg_cuda(disp, self.Q, self.params)
        pclimg_msg = self.br.cv2_to_imgmsg(pclimg)
        pclimg_msg.header.frame_id = disp_msg.header.frame_id
        pclimg_msg.header.stamp = self.get_clock().now().to_msg()
        # self.get_logger().info(f'{cam2_rect_mono_msg.header}')
        self.pub_cam_3d_pclimg.publish(pclimg_msg)
