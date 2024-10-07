#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import message_filters

import numpy as np
import time
from utils import cv_cuda_utils, pcl_utils

class PUB_CAM_DISP(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        self.params = params
        self.br = CvBridge()

        # Initialize stuffs for 3D reconstruction
        self.cam1_sgm = cv_cuda_utils.load_cam1_sgm(self.params)

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
        self.pub_cam_disp = self.create_publisher(Image, "disparity", params["queue_size"])

        # Subscribers
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(self, CompressedImage, params["cam1_topic"]),
                message_filters.Subscriber(self, CompressedImage, params["cam2_topic"])
            ],
            queue_size=params["queue_size"], slop=params["slop"]
        )
        self.ts.registerCallback(self.sub_callback)

    def sub_callback(self, cam1_rect_mono_msg, cam2_rect_mono_msg):
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
            disp, self.params["bf_size"]
        )
        disp_msg = self.br.cv2_to_imgmsg(disp/16)
        self.get_logger().info(f"{disp.dtype}", once=True)
        disp_msg.header.frame_id = cam1_rect_mono_msg.header.frame_id
        disp_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_cam_disp.publish(disp_msg)
