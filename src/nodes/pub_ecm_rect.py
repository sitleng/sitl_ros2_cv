#!/usr/bin/env python3

import cv2

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
import message_filters

from utils import ecm_utils, cv_cuda_utils

class PUB_ECM_RECT():
    def __init__(self, node_name, params):
        self.br = CvBridge()
        self.pub_img_rect_mono  = rospy.Publisher(node_name+'/image_mono',CompressedImage,queue_size=10)
        self.pub_img_rect_color = rospy.Publisher(node_name+'/image_color',CompressedImage,queue_size=10)
        self.map_x, self.map_y = ecm_utils.load_rect_maps(node_name, params)

    def __del__(self):
        print("Shutting down...")

    def callback_wo_cuda(self, caminfo_msg, color_msg):
        color      = self.br.compressed_imgmsg_to_cv2(color_msg)
        rect_color = cv2.remap(color, self.map_x, self.map_y, cv2.INTER_LINEAR)
        rect_mono  = cv2.cvtColor(rect_color, cv2.COLOR_BGR2GRAY)

        rect_color_msg = self.br.cv2_to_compressed_imgmsg(rect_color, dst_format="jpg")
        rect_mono_msg  = self.br.cv2_to_compressed_imgmsg(rect_mono, dst_format="jpg")
        
        rect_color_msg.header.frame_id = rect_mono_msg.header.frame_id = caminfo_msg.header.frame_id
        rect_color_msg.header.stamp = rect_mono_msg.header.stamp = rospy.Time.now()

        self.pub_img_rect_color.publish(rect_color_msg)
        self.pub_img_rect_mono.publish(rect_mono_msg)

    def callback_w_cuda(self, caminfo_msg, color_msg):
        color_cuda      = cv_cuda_utils.cvmat2gpumat(self.br.compressed_imgmsg_to_cv2(color_msg))
        rect_color_cuda = cv2.cuda.remap(color_cuda, self.map_x, self.map_y, cv2.INTER_LINEAR)
        rect_mono_cuda  = cv2.cuda.cvtColor(rect_color_cuda, cv2.COLOR_BGR2GRAY)

        rect_color_msg = self.br.cv2_to_compressed_imgmsg(rect_color_cuda.download(),dst_format="jpg")
        rect_mono_msg  = self.br.cv2_to_compressed_imgmsg(rect_mono_cuda.download(),dst_format="jpg")

        rect_color_msg.header.frame_id = rect_mono_msg.header.frame_id = caminfo_msg.header.frame_id
        rect_color_msg.header.stamp = rect_mono_msg.header.stamp = rospy.Time.now()

        self.pub_img_rect_color.publish(rect_color_msg)
        self.pub_img_rect_mono.publish(rect_mono_msg)

if __name__ == '__main__':
    rospy.init_node("init_ecm")
    node_name = rospy.get_name()
    params = rospy.get_param(node_name)
    pub_ecm_rect = PUB_ECM_RECT(node_name,params)
    topic_list = []
    topic_list.append(message_filters.Subscriber(node_name.split("_")[0]+"/camera_info", CameraInfo))
    topic_list.append(message_filters.Subscriber(node_name.split("_")[0]+"/image_color", CompressedImage))
    ts = message_filters.ApproximateTimeSynchronizer(topic_list, queue_size=10, slop=0.01)
    try:
        if not params["gpu_flag"]:
            ts.registerCallback(pub_ecm_rect.callback_wo_cuda)
        else:
            ts.registerCallback(pub_ecm_rect.callback_w_cuda)
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)
    finally:
        del pub_ecm_rect