#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import message_filters

import numpy as np
from utils import cv_cuda_utils

class PUB_CAM_DISP(Node):
    def __init__(self, params):
        super().__init__("pub_ecm_disp")
        self.params = params
        self.br = CvBridge()
        self.get_logger().info("Loading Stereo Matching Models ...", once=True)
        self.cam1_sgm = cv_cuda_utils.load_cam1_sgm(self.params)
        self.cam2_sgm = cv_cuda_utils.load_cam2_sgm(self.params)
        if params["wls_filter_flag"]:
            self.wls_filter = cv_cuda_utils.load_wls_filter(self.cam1_sgm, self.params)
        else:
            self.wls_filter = None
        if params["dbf_flag"]:
            self.dbf = cv_cuda_utils.load_dbf(self.params)
        else:
            self.dbf = None
        self.pub_ecm_disp = self.create_publisher(Image, "/ecm/disparity", 10)

        if params["calib_dir"] == "L2R":
            self.cam1_topic = "/ecm/left_rect/image_mono"
            self.cam2_topic = "/ecm/right_rect/image_mono"
        elif params["calib_dir"] == "R2L":
            self.cam1_topic = "/ecm/right_rect/image_mono"
            self.cam2_topic = "/ecm/left_rect/image_mono"

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(self, CompressedImage, self.cam1_topic),
                message_filters.Subscriber(self, CompressedImage, self.cam2_topic)
            ],
            queue_size=10, slop=params["slop"] # queue_size, slop
        )
        self.ts.registerCallback(self.sub_callback)

    def sub_callback(self, cam1_rect_mono_msg, cam2_rect_mono_msg):
        cam1_rect_mono_cuda = cv_cuda_utils.cvmat2gpumat(
            self.br.compressed_imgmsg_to_cv2(cam1_rect_mono_msg)
        )
        cam2_rect_mono_cuda = cv_cuda_utils.cvmat2gpumat(
            self.br.compressed_imgmsg_to_cv2(cam2_rect_mono_msg)
        )
        if self.params["wls_filter_flag"]:
            disp_cuda = cv_cuda_utils.cuda_sgm_wls_filter(
                self.cam1_sgm,
                self.cam2_sgm,
                cam1_rect_mono_cuda,
                cam2_rect_mono_cuda,
                self.wls_filter
            )
            disp_cuda = cv_cuda_utils.apply_bf(
                disp_cuda, self.params["bf_size"]
            )
        else:
            disp_cuda = cv_cuda_utils.cuda_sgm_dbf(
                self.cam1_sgm,
                cam1_rect_mono_cuda,
                cam2_rect_mono_cuda,
                self.dbf
            )
            disp_cuda = cv_cuda_utils.apply_bf(
                disp_cuda, self.params["bf_size"]
            )
        disp_msg = self.br.cv2_to_imgmsg(np.float32(disp_cuda.download()/16))
        disp_msg.header.frame_id = cam1_rect_mono_msg.header.frame_id
        disp_msg.header.stamp = self.get_clock().now().to_msg()
        # self.get_logger().info(f'{cam2_rect_mono_msg.header}')
        self.pub_ecm_disp.publish(disp_msg)