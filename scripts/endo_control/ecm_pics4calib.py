#!/usr/bin/env python3

import cv2
import argparse
import rospy
import os
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge


class ECM_PICS4CALIB():
    def __init__(self,args):
        self.id_image  = 0
        self.caminfo   = rospy.wait_for_message("/ecm/left/camera_info",CameraInfo)
        calib_path = args.save_dir + "/" + args.cam_type + "/{}x{}".format(self.caminfo.width,self.caminfo.height)
        self.leftpath  = calib_path + "/left/"
        self.rightpath = calib_path + "/right/"
        if not os.path.exists(self.leftpath) or not os.path.exists(self.rightpath):
            os.makedirs(self.leftpath)
            os.makedirs(self.rightpath)
        self.chess_size = (5,8)
        self.criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.br = CvBridge()

    def callback(self,imgmsgL,imgmsgR):
        rospy.loginfo("Entered Callback...")
        imgL = self.br.imgmsg_to_cv2(imgmsgL)
        imgR = self.br.imgmsg_to_cv2(imgmsgR)
        retL, cornersL = cv2.findChessboardCorners(imgL,self.chess_size,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK)
        retR, cornersR = cv2.findChessboardCorners(imgR,self.chess_size,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK)
        if retL and retR:
            rospy.loginfo("Chessboard detected...!")
            # corners2L = cv2.cornerSubPix(imgL,cornersL,(11,11),(-1,-1),self.criteria)
            # corners2R = cv2.cornerSubPix(imgR,cornersR,(11,11),(-1,-1),self.criteria)
            # cv2.drawChessboardCorners(imgL,self.chess_size,corners2L,retL)
            # cv2.drawChessboardCorners(imgR,self.chess_size,corners2R,retR)
            if self.id_image%5 == 0:
                str_id_image = str(self.id_image)
                cv2.imwrite(self.leftpath +'left' +str_id_image+'.png',imgL)
                cv2.imwrite(self.rightpath+'right'+str_id_image+'.png',imgR)
            self.id_image += 1
        
        
if __name__ == "__main__":
    rospy.init_node("ecm_pics4calib")
    parser = argparse.ArgumentParser(
        prog='ECM_PICS4CALIB',
        description='Capture Images for Endoscope Calibration',
        epilog='Text at the bottom of help')
    parser.add_argument('-sd', '--save_dir', type=str, help='directory to save the images...')
    parser.add_argument('-c', '--cam_type', type=str, help='Choose the endoscope type: 0 or 30')
    args = parser.parse_args()
    
    app = ECM_PICS4CALIB(args)
    rospy.loginfo("Start finding chessboard...")
    imgmsgL = message_filters.Subscriber('/ecm/left/image_mono',Image)
    imgmsgR = message_filters.Subscriber('/ecm/right/image_mono',Image)
    ts = message_filters.ApproximateTimeSynchronizer([imgmsgL,imgmsgR], queue_size=10, slop=0.05)
    ts.registerCallback(app.callback)
    r = rospy.Rate(60)
    while not rospy.is_shutdown():
        if app.id_image > 2500:
            break
        r.sleep()