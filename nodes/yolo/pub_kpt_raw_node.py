#!/usr/bin/env python3

# Import ROS libraries
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import message_filters

# Import custom libraries
from sitl_ros2_interfaces.msg import Dt2KptState
from utils import yolo_utils, ros2_utils, kpt_utils

class PUB_KPT_RAW_YOLO(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        self.br = CvBridge()
        self.load_params(params)
        self.keypt_model = yolo_utils.load_model(params['model_path'])
        self.keypt_metadata = yolo_utils.load_kpt_metadata(self.inst_name)

        qos_profile = ros2_utils.custom_qos_profile(params["queue_size"])
        self.pub_pch  = self.create_publisher(Dt2KptState, "raw", qos_profile)
        ts = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(self, CompressedImage, params["refimg_topic"], qos_profile=qos_profile),
                message_filters.Subscriber(self, Image, params["pclimg_topic"], qos_profile=qos_profile)
            ],
            queue_size=params["queue_size"], slop=params["slop"]
        )
        ts.registerCallback(self.callback)

    def load_params(self, params):
        self.inst_name = params["inst_name"]
        self.ct_kpt_nm = params["ct_kpt_nm"]
        self.mad_thr = params["mad_thr"]
        self.window_size = params["window_size"]

    def callback(self, img_msg, pclimg_msg):
        img = self.br.compressed_imgmsg_to_cv2(img_msg)
        pclimg = self.br.imgmsg_to_cv2(pclimg_msg)
        kpts_2d = yolo_utils.get_kpts_2d(img, self.keypt_model)
        if kpts_2d is None or kpts_2d.size == 0:
            ros2_utils.loginfo(self, f"{self.inst_name} not detected!")
            return
        kpt_nms = self.keypt_metadata['kpt_nms']
        kpt_nms, kpts_3d = kpt_utils.win_avg_3d_kpts(
            kpts_2d, kpt_nms, pclimg, self.window_size, self.mad_thr
        )
        if kpts_3d is None or self.ct_kpt_nm not in kpt_nms:
            ros2_utils.loginfo(self, f"Invalid {self.inst_name} Keypoints!")
            return
        pch_msg = kpt_utils.gen_dt2kptstate(kpt_nms, kpts_2d, kpts_3d, ros2_utils.now(self))
        self.pub_pch.publish(pch_msg)
