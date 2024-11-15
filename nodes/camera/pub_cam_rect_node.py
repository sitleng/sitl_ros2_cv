#!/usr/bin/env python3

from rclpy.node import Node
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import CompressedImage, CameraInfo

import cv2
from utils import ecm_utils, cv_cuda_utils, ros2_utils

class PUB_CAM_RECT(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        self.br = CvBridge()
        self.map_x, self.map_y = ecm_utils.load_rect_maps(params["cam_side"], params)
        qos_profile = ros2_utils.custom_qos_profile(params["queue_size"])
        self.pub_img_rect_mono  = self.create_publisher(
            CompressedImage, 'rect/image_mono', qos_profile
        )
        self.pub_img_rect_color = self.create_publisher(
            CompressedImage, 'rect/image_color', qos_profile
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [
                # "/ecm/"+params["cam_side"]+
                message_filters.Subscriber(self, CameraInfo, 'camera_info', qos_profile=qos_profile),
                message_filters.Subscriber(self, CompressedImage, 'image_color', qos_profile=qos_profile)
            ],
            queue_size=params["queue_size"], slop=params["slop"] # queue_size, slop
        )
        if params["gpu_flag"]:
            self.ts.registerCallback(self.cb_w_cuda)
        else:
            self.ts.registerCallback(self.cb_wo_cuda)

    def cb_wo_cuda(self, caminfo_msg, color_msg):
        rect_color     = cv2.remap(
            self.br.compressed_imgmsg_to_cv2(color_msg), 
            self.map_x, self.map_y, cv2.INTER_LINEAR
        )
        rect_mono      = cv2.cvtColor(rect_color, cv2.COLOR_BGR2GRAY)
        rect_color_msg = self.br.cv2_to_compressed_imgmsg(rect_color, dst_format="jpg")
        rect_mono_msg  = self.br.cv2_to_compressed_imgmsg(rect_mono, dst_format="jpg")
        rect_color_msg.header.frame_id = rect_mono_msg.header.frame_id = caminfo_msg.header.frame_id
        rect_color_msg.header.stamp = rect_mono_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_img_rect_color.publish(rect_color_msg)
        self.pub_img_rect_mono.publish(rect_mono_msg)

    def cb_w_cuda(self, caminfo_msg, color_msg):
        rect_color     = cv2.cuda.remap(
            cv_cuda_utils.cvmat2gpumat(self.br.compressed_imgmsg_to_cv2(color_msg)), 
            self.map_x, self.map_y, cv2.INTER_LINEAR
        )
        rect_mono      = cv2.cuda.cvtColor(rect_color, cv2.COLOR_BGR2GRAY)
        rect_color_msg = self.br.cv2_to_compressed_imgmsg(rect_color.download(), dst_format="jpg")
        rect_mono_msg  = self.br.cv2_to_compressed_imgmsg(rect_mono.download(), dst_format="jpg")
        rect_color_msg.header.frame_id = rect_mono_msg.header.frame_id = caminfo_msg.header.frame_id
        rect_color_msg.header.stamp = rect_mono_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_img_rect_color.publish(rect_color_msg)
        self.pub_img_rect_mono.publish(rect_mono_msg)
