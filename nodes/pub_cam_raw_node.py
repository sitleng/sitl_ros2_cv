#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, CameraInfo

import cv2
from cv_bridge import CvBridge
from utils import ecm_utils, cv_cuda_utils

class PUB_CAM_RAW(Node):
    def __init__(self, params):
        super().__init__(params["cam_side"]+"_image")
        self.params = params
        self.br = CvBridge()
        self.pub_img_mono     = self.create_publisher(CompressedImage, params["cam_side"]+'/image_mono',  10)
        self.pub_img_color    = self.create_publisher(CompressedImage, params["cam_side"]+'/image_color', 10)
        self.res              = ecm_utils.Resolution(params["resolution"])
        self.camera = ecm_utils.init_camera(params, self.res)
        self.sub_cam_info = self.create_subscription(
            CameraInfo,
            params["cam_side"]+'/camera_info',
            self.callback,
            10
        )

    def callback(self, cam_info_msg):
        ret, color = self.camera.read()
        if ret:
            if not self.params["gpu_flag"]:
                mono        = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
                color_msg   = self.br.cv2_to_compressed_imgmsg(color, dst_format="jpg")
                mono_msg    = self.br.cv2_to_compressed_imgmsg(mono, dst_format="jpg")
            else:
                mono  = cv2.cuda.cvtColor(cv_cuda_utils.cvmat2gpumat(color), cv2.COLOR_BGR2GRAY)
                color_msg   = self.br.cv2_to_compressed_imgmsg(color, dst_format="jpg")
                mono_msg    = self.br.cv2_to_compressed_imgmsg(mono.download(), dst_format="jpg")
            color_msg.header.frame_id = mono_msg.header.frame_id = cam_info_msg.header.frame_id
            color_msg.header.stamp = mono_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_img_color.publish(color_msg)
            self.pub_img_mono.publish(mono_msg)
