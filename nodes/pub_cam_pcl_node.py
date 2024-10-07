#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from cv_bridge import CvBridge
import message_filters

from utils import pcl_utils

class PUB_CAM_PCL(Node):
    def __init__(self,params):
        super().__init__(params["node_name"])
        self.br = CvBridge()
        self.pub_cam_pcl = self.create_publisher(PointCloud2, "pcl2", params["queue_size"])
        ts = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(self, CompressedImage, params["ref_cam_topic"]),
                message_filters.Subscriber(self, Image, "pclimg")
            ],
            queue_size=params["queue_size"], slop=params["slop"]
        )
        ts.registerCallback(self.callback)

    def callback(self, cam1_rect_color_msg, pclimg_msg):
        cam1_rect_color = self.br.compressed_imgmsg_to_cv2(cam1_rect_color_msg)
        pclimg = self.br.imgmsg_to_cv2(pclimg_msg)
        pcl_msg = pcl_utils.gen_pcl(
            cam1_rect_color,
            pclimg,
            pclimg_msg.header.frame_id,
            self.get_clock().now().to_msg()
        )
        self.pub_cam_pcl.publish(pcl_msg)
