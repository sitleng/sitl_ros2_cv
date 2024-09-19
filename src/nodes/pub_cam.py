#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import CompressedImage, CameraInfo
from skimage import exposure
from cv_bridge import CvBridge
from utils import ecm_utils, cv_cuda_utils

class PUB_CAM():
    def __init__(self,node_name,params):
        self.loop_rate = rospy.Rate(params["fps"])
        self.br = CvBridge()
        self.pub_img_mono  = rospy.Publisher(node_name+'/image_mono',CompressedImage,queue_size=10)
        self.pub_img_color = rospy.Publisher(node_name+'/image_color',CompressedImage,queue_size=10)
        # self.pub_raw_info  = rospy.Publisher(node_name+'/camera_info',CameraInfo,queue_size=10)
        self.res           = ecm_utils.Resolution(params["resolution"])
        # self.caminfo_raw   = ecm_utils.gen_caminfo(node_name,params,self.res)
        self.params = params

    def start(self):
        camera = ecm_utils.init_camera(self.params,self.res)
        while not rospy.is_shutdown() and camera.isOpened():
            ret, color = camera.read()
            if ret:
                t = rospy.get_rostime().now()
                # color_cuda = cv_cuda_utils.cvmat2gpumat(color)
                # mono_cuda  = cv2.cuda.cvtColor(color_cuda,cv2.COLOR_BGR2GRAY)
                # color     = exposure.adjust_gamma(color, 1/self.params["gamma"])
                mono      = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
                color_msg = self.br.cv2_to_compressed_imgmsg(color,dst_format="jpg")
                mono_msg  = self.br.cv2_to_compressed_imgmsg(mono,dst_format="jpg")
                # color_msg = self.br.cv2_to_compressed_imgmsg(color_cuda.download(),dst_format="jpg")
                # mono_msg  = self.br.cv2_to_compressed_imgmsg(mono_cuda.download(),dst_format="jpg")
                # self.caminfo_raw.header = color_msg.header
                # self.caminfo_raw.header.stamp = color_msg.header.stamp = mono_msg.header.stamp = t
                color_msg.header.stamp = mono_msg.header.stamp = t
                self.pub_img_color.publish(color_msg)
                self.pub_img_mono.publish(mono_msg)
                # self.pub_raw_info.publish(self.caminfo_raw)
            self.loop_rate.sleep()
        camera.release()

if __name__ == '__main__':
    rospy.init_node("init_cam")
    node_name = rospy.get_name()
    params = rospy.get_param(node_name)
    try:
        pub_ecm = PUB_CAM(node_name,params)
        pub_ecm.start()
    except Exception as e:
        rospy.logerr(e)