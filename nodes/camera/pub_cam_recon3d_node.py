#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import message_filters

import numpy as np
import time
from sitl_ros2_interfaces.utils import ros2_utils, tf_utils
from utils import ecm_utils, cv_cuda_utils, pcl_utils

class PUB_CAM_RECON3D(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        self.params = params
        self.br = CvBridge()

        # Initialize stuffs for 3D reconstruction
        self.cam1_sgm = cv_cuda_utils.load_cam1_sgm(self.params)
        self.res = ecm_utils.Resolution(params["resolution"])
        self.Q, self.B, self.f, self.g_LR = self.load_stereo_calib_local(
            self.params,
            self.res.width,
            self.res.height
        )

        # Initialize filters if necessary
        if params["wls_filter_flag"]:
            self.cam2_sgm = cv_cuda_utils.load_cam2_sgm(self.params)
            self.wls_filter = cv_cuda_utils.load_wls_filter(self.cam1_sgm, self.params)
        else:
            self.wls_filter = None
        if params["dbf_flag"]:
            self.dbf = cv_cuda_utils.load_dbf(self.params)
        else:
            self.dbf = None

        # Publishers
        qos_profile = ros2_utils.custom_qos_profile(params["queue_size"])
        # self.pub_cam_disp = self.create_publisher(Image, "disparity", params["queue_size"])
        self.pub_cam_3d_pclimg = self.create_publisher(Image, "pclimg", qos_profile)
        # self.pub_cam_3d_pclimg = self.create_publisher(CompressedImage, "pclimg", params["queue_size"])

        # Subscribers
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(self, CompressedImage, params["cam1_topic"], qos_profile=qos_profile),
                message_filters.Subscriber(self, CompressedImage, params["cam2_topic"], qos_profile=qos_profile)
            ],
            queue_size=params["queue_size"], slop=params["slop"]
        )
        self.ts.registerCallback(self.sub_callback)

    def load_stereo_calib_local(self, params, width, height):
        self.get_logger().info("Loading Stereo Calibration Parameters ...")
        Q, R, T, B, f = ecm_utils.load_stereo_calib(params, width, height)
        return Q, B, f, tf_utils.ginv(tf_utils.gen_g(R, T/params["depth_scale"]))

    # def sub_callback(self, cam1_rect_mono_msg, cam2_rect_mono_msg):
    #     pclimg = np.random.random_sample((720,1280,3))
    #     self.get_logger().info(f'{pclimg.flatten().shape}', once=True)
    #     # pclimg_msg = self.br.cv2_to_compressed_imgmsg(pclimg, dst_format='tiff')
    #     pclimg_msg = self.br.cv2_to_imgmsg(pclimg)
    #     pclimg_msg.header.frame_id = cam1_rect_mono_msg.header.frame_id
    #     pclimg_msg.header.stamp = self.get_clock().now().to_msg()
    #     self.pub_cam_3d_pclimg.publish(pclimg_msg)

    def sub_callback(self, cam1_rect_mono_msg, cam2_rect_mono_msg):
        start_time = time.time()
        if self.params["wls_filter_flag"]:
            disp = cv_cuda_utils.cuda_sgm_wls_filter(
                self.cam1_sgm,
                self.cam2_sgm,
                self.br.compressed_imgmsg_to_cv2(cam1_rect_mono_msg),
                self.br.compressed_imgmsg_to_cv2(cam2_rect_mono_msg),
                self.wls_filter
            )
        else:
            disp = cv_cuda_utils.cuda_sgm_dbf(
                self.cam1_sgm,
                self.br.compressed_imgmsg_to_cv2(cam1_rect_mono_msg),
                self.br.compressed_imgmsg_to_cv2(cam2_rect_mono_msg),
                self.dbf
            )
        disp = cv_cuda_utils.apply_bf(
            disp/16, self.params["bf_size"]
        )
        self.get_logger().info(f'{np.min(disp), np.max(disp)}', once=True)
        # disp_msg = self.br.cv2_to_imgmsg(disp/16)
        # disp_msg = self.br.cv2_to_imgmsg(disp)
        # self.get_logger().info(f"{disp.dtype}", once=True)
        # disp_msg.header.frame_id = cam1_rect_mono_msg.header.frame_id
        # disp_msg.header.stamp = self.get_clock().now().to_msg()
        # self.pub_cam_disp.publish(disp_msg)
        # pclimg = pcl_utils.disp2pclimg(disp, self.Q, self.params["pcl_scale"], self.params["depth_trunc"])
        pclimg = pcl_utils.disp2pclimg_cuda(disp, self.Q, self.params["pcl_scale"], self.params["depth_trunc"])
        # pclimg_msg = self.br.cv2_to_compressed_imgmsg(pclimg)
        pclimg_msg = self.br.cv2_to_imgmsg(pclimg)
        pclimg_msg.header.frame_id = cam1_rect_mono_msg.header.frame_id
        pclimg_msg.header.stamp = self.get_clock().now().to_msg()
        # self.get_logger().info(f'{time.time() - start_time}')
        self.pub_cam_3d_pclimg.publish(pclimg_msg)
