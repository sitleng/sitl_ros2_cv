#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import CompressedImage, CameraInfo
from skimage import exposure
from cv_bridge import CvBridge
from utils import ecm_utils, cv_cuda_utils

class PUB_ECM():
    def __init__(self, node_name, params):
        self.loop_rate = rospy.Rate(params["fps"])
        self.br = CvBridge()
        self.pub_img_mono     = rospy.Publisher(node_name+'/image_mono',CompressedImage,queue_size=10)
        self.pub_img_color    = rospy.Publisher(node_name+'/image_color',CompressedImage,queue_size=10)
        self.pub_raw_info     = rospy.Publisher(node_name+'/camera_info',CameraInfo,queue_size=10)
        self.res              = ecm_utils.Resolution(params["resolution"])
        self.raw_caminfo_dict = ecm_utils.load_raw_caminfo(node_name, params, self.res)
        self.camera = ecm_utils.init_camera(params, self.res)

    def __del__(self):
        print("Shutting down...")

    def start_wo_cuda(self):
        while not rospy.is_shutdown() and self.camera.isOpened():
            ret, color = self.camera.read()
            if ret:
                # color     = exposure.adjust_gamma(color, 1/self.params["gamma"])
                mono        = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
                t = rospy.Time.now()
                caminfo_msg = ecm_utils.gen_caminfo(self.raw_caminfo_dict, t)
                color_msg   = self.br.cv2_to_compressed_imgmsg(color, dst_format="jpg")
                mono_msg    = self.br.cv2_to_compressed_imgmsg(mono, dst_format="jpg")
                color_msg.header = mono_msg.header = caminfo_msg.header
                self.pub_raw_info.publish(caminfo_msg)
                self.pub_img_color.publish(color_msg)
                self.pub_img_mono.publish(mono_msg)
            self.loop_rate.sleep()

    def start_w_cuda(self):
        while not rospy.is_shutdown() and self.camera.isOpened():
            ret, color = self.camera.read()
            color = cv2.UMat(color)
            if ret:
                t = rospy.get_rostime().now()
                # color     = exposure.adjust_gamma(color, 1/self.params["gamma"])
                color_cuda = cv_cuda_utils.cvmat2gpumat(color)
                mono_cuda  = cv2.cuda.cvtColor(color_cuda,cv2.COLOR_BGR2GRAY)
                t = rospy.Time.now()
                caminfo_msg = ecm_utils.gen_caminfo(self.raw_caminfo_dict, t)
                color_msg   = self.br.cv2_to_compressed_imgmsg(color_cuda.download(),dst_format="jpg")
                mono_msg    = self.br.cv2_to_compressed_imgmsg(mono_cuda.download(),dst_format="jpg")
                color_msg.header = mono_msg.header = caminfo_msg.header
                self.pub_raw_info.publish(caminfo_msg)
                self.pub_img_color.publish(color_msg)
                self.pub_img_mono.publish(mono_msg)
            self.loop_rate.sleep()

if __name__ == '__main__':
    rospy.init_node("init_ecm")
    node_name = rospy.get_name()
    params = rospy.get_param(node_name)
    try:
        pub_ecm = PUB_ECM(node_name,params)
        if not params["gpu_flag"]:
            pub_ecm.start_wo_cuda()
        else:
            pub_ecm.start_w_cuda()
    except Exception as e:
        rospy.logerr(e)
    finally:
        pub_ecm.camera.release()
        del pub_ecm