#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import message_filters
from utils import cv_cuda_utils

class PUB_ECM_DISP():
    def __init__(self,params):
        self.br = CvBridge()
        self.params = params
        rospy.loginfo("Loading Stereo Matching Models ...")
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
        self.pub_ecm_disp = rospy.Publisher("/ecm/disparity", Image, queue_size=10)
    
    def callback(self, cam1_rect_mono_msg, cam2_rect_mono_msg):
        # if self.params["bf_size"] > 1:
        #     cam1_rect_mono_cuda = cv_cuda_utils.apply_bf(
        #         cv_cuda_utils.cvmat2gpumat(
        #             self.br.compressed_imgmsg_to_cv2(cam1_rect_mono_msg)
        #         ), self.params["bf_size"]
        #     )
        #     cam2_rect_mono_cuda = cv_cuda_utils.apply_bf(
        #         cv_cuda_utils.cvmat2gpumat(
        #             self.br.compressed_imgmsg_to_cv2(cam2_rect_mono_msg)
        #         ), self.params["bf_size"]
        #     )
        # else:
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
        elif self.params["dbf_flag"]:
            disp_cuda = cv_cuda_utils.cuda_sgm_dbf(
                self.cam1_sgm,
                cam1_rect_mono_cuda,
                cam2_rect_mono_cuda,
                self.dbf
            )
        else:
            disp_cuda = cv_cuda_utils.cuda_sgm_dbf(
                self.cam1_sgm,
                cam1_rect_mono_cuda,
                cam2_rect_mono_cuda,
                None
            )
            disp_cuda = cv_cuda_utils.apply_bf(
                disp_cuda, self.params["bf_size"]
            )
        disp_msg = self.br.cv2_to_imgmsg(np.float32(disp_cuda.download()/16))
        disp_msg.header.frame_id = cam1_rect_mono_msg.header.frame_id
        disp_msg.header.stamp = rospy.Time.now()
        self.pub_ecm_disp.publish(disp_msg)

if __name__ == '__main__':
    rospy.init_node("pub_ecm_disp")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)
    
    try:
        app = PUB_ECM_DISP(params)
        rospy.loginfo("Start Publishing Disparity Images ...")
        left_rect_mono_msg  = message_filters.Subscriber( "/ecm/left_rect/image_mono", CompressedImage)
        right_rect_mono_msg = message_filters.Subscriber("/ecm/right_rect/image_mono", CompressedImage)
        # left_rect_color_msg  = message_filters.Subscriber("/ecm/left_rect/image_color" ,CompressedImage)
        # right_rect_color_msg = message_filters.Subscriber("/ecm/right_rect/image_color",CompressedImage)
        if params["calib_dir"] == "L2R":
            ts                   = message_filters.ApproximateTimeSynchronizer(
                [left_rect_mono_msg, right_rect_mono_msg],
                slop=0.02, queue_size=10
            )
        elif params["calib_dir"] == "R2L":
            ts                   = message_filters.ApproximateTimeSynchronizer(
                [right_rect_mono_msg,left_rect_mono_msg],
                slop=0.02, queue_size=10
            )
        ts.registerCallback(app.callback)
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)