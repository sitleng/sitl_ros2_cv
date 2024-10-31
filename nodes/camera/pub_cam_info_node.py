#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import CameraInfo

from sitl_ros2_interfaces.utils import ros2_utils
from utils import ecm_utils

class PUB_CAM_INFO(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        qos_profile = ros2_utils.custom_qos_profile(params["queue_size"])
        self.pub_cam_info = self.create_publisher(
            CameraInfo, 'camera_info', qos_profile
        )
        self.res = ecm_utils.Resolution(params["resolution"])
        self.cam_info_msg = ecm_utils.gen_caminfo(
            ecm_utils.load_raw_caminfo(params["cam_side"], params, self.res),
            self.get_clock().now().to_msg()
        )
        self.loop_rate = self.create_timer(1/params["fps"], self.callback)

    def callback(self):
        self.cam_info_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_cam_info.publish(self.cam_info_msg)
