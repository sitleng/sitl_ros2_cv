#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import message_filters

import cv2
from utils import ecm_utils, misc_utils, ros2_utils

class REC_STEREO_CAM(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        
        # Initialize variables
        self.br = CvBridge()
        self.res = ecm_utils.Resolution(params['resolution'])
        self.ext = int(0.1*self.res.height)
        misc_utils.check_empty_path(params['save_path'])
        self.cam1_video = cv2.VideoWriter(
            params['save_path']+'/cam1.mp4',
            cv2.VideoWriter_fourcc(*params['fourcc']),
            params['fps'],
            (self.res.width, self.res.height + self.ext),
            True
        )
        self.cam2_video = cv2.VideoWriter(
            params['save_path']+'/cam2.mp4',
            cv2.VideoWriter_fourcc(*params['fourcc']),
            params['fps'],
            (self.res.width, self.res.height + self.ext),
            True
        )

        # Publishers
        qos_profile = ros2_utils.custom_qos_profile(params["queue_size"])

        # Subscribers
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(self, CompressedImage, params["cam1_topic"], qos_profile=qos_profile),
                message_filters.Subscriber(self, CompressedImage, params["cam2_topic"], qos_profile=qos_profile)
            ],
            queue_size=params["queue_size"], slop=params["slop"]
        )
        self.ts.registerCallback(self.sub_callback)

    def sub_callback(self, cam1_rect_msg, cam2_rect_msg):
        cam1_img = self.br.compressed_imgmsg_to_cv2(cam1_rect_msg)
        cam2_img = self.br.compressed_imgmsg_to_cv2(cam2_rect_msg)
        cam1_img_ext = cv2.copyMakeBorder(cam1_img, 0, self.ext, 0, 0, cv2.BORDER_CONSTANT, None, [255,255,255])
        cam2_img_ext = cv2.copyMakeBorder(cam2_img, 0, self.ext, 0, 0, cv2.BORDER_CONSTANT, None, [255,255,255])
        cam1_img_ext = cv2.putText(
            cam1_img_ext,
            str(ros2_utils.to_sec(cam1_rect_msg)),
            (self.ext, int(1.07*self.res.height)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA
        )
        cam2_img_ext = cv2.putText(
            cam2_img_ext,
            str(ros2_utils.to_sec(cam2_rect_msg)),
            (self.ext, int(1.07*self.res.height)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA
        )
        self.cam1_video.write(cam1_img_ext)
        self.cam2_video.write(cam2_img_ext)
        
