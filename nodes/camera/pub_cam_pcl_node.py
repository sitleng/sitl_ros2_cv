#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from cv_bridge import CvBridge
import message_filters

from sitl_ros2_interfaces.utils import ros2_utils
from utils import pcl_utils

class PUB_CAM_PCL(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        self.br = CvBridge()
        qos_profile = ros2_utils.custom_qos_profile(params["queue_size"])
        self.pub_cam_pcl = self.create_publisher(PointCloud2, "pcl2", qos_profile)
        ts = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(self, CompressedImage, params["ref_cam_topic"], qos_profile=qos_profile),
                message_filters.Subscriber(self, Image, "pclimg", qos_profile=qos_profile)
                # message_filters.Subscriber(self, CompressedImage, "pclimg", qos_profile=qos_profile)
            ],
            queue_size=params["queue_size"], slop=params["slop"]
        )
        ts.registerCallback(self.callback)

    def callback(self, cam1_rect_color_msg, pclimg_msg):
        cam1_rect_color = self.br.compressed_imgmsg_to_cv2(cam1_rect_color_msg)
        pclimg = self.br.imgmsg_to_cv2(pclimg_msg)
        # pclimg = self.br.compressed_imgmsg_to_cv2(pclimg_msg)
        self.get_logger().info(f'{pclimg.dtype}', once=True)
        pcl_msg = pcl_utils.pclimg2pcl2(
            cam1_rect_color,
            pclimg,
            pclimg_msg.header.frame_id,
            self.get_clock().now().to_msg()
        )
        self.pub_cam_pcl.publish(pcl_msg)
