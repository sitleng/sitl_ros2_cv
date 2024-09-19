#!/usr/bin/env python3

import cv2
import rospy
import os
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class ECM_RECORD():
    def __init__(self):
        self.id_image  = 0
        self.filepath  = "/home/leo/dt2_dataset/kpts"
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        self.br = CvBridge()

    def callback(self,imgmsgL):
        imgL = self.br.compressed_imgmsg_to_cv2(imgmsgL)
        if self.id_image%5 == 0 and self.id_image < 10000:
            str_id_image = str(self.id_image)
            cv2.imwrite(self.filepath+'/'+str_id_image+'.png', imgL)
        self.id_image += 1
        
if __name__ == "__main__":
    rospy.init_node("ecm_record")
    app = ECM_RECORD()
    rospy.loginfo("Start recording ecm images...")
    imgmsgL = rospy.Subscriber('/ecm/left_rect/image_color',CompressedImage,app.callback)
    rospy.spin()