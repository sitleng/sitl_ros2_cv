#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge

import cv2
from utils import ros2_utils

class PUB_VIDEO(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        self.br  = CvBridge()
        self.video = cv2.VideoCapture(params["video_path"])

        qos_profile = ros2_utils.custom_qos_profile(params["queue_size"])
        self.pub_video_mono  = self.create_publisher(
            CompressedImage, 'rect/image_mono', qos_profile
        )
        self.pub_video_color = self.create_publisher(
            CompressedImage, 'rect/image_color', qos_profile
        )
        self.sub_cam_info = self.create_subscription(
            CameraInfo,
            'camera_info',
            self.callback,
            qos_profile
        )

    def callback(self, caminfo_msg):
        ret, frame = self.video.read()
        if ret:
            color = frame[:720, :1280, :]
            color_msg = self.br.cv2_to_compressed_imgmsg(color)
            mono_msg  = self.br.cv2_to_compressed_imgmsg(
                cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            )
        else:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        color_msg.header.frame_id = mono_msg.header.frame_id = caminfo_msg.header.frame_id
        color_msg.header.stamp = mono_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_video_color.publish(color_msg)
        self.pub_video_mono.publish(mono_msg)
        