#! /usr/bin/env python3

import numpy as np
np.seterr(all="ignore")
import os
import sys
import argparse
import cv2
import pickle
import rospy
import scipy.io as sio
from utils import ecm_utils, tf_utils

def load_calibration_file(calib_path,image_size,args) :
    calib_fn = "{}/{}x{}/{}x{}_{}_caminfo.mat".format(
        calib_path,
        image_size.width,
        image_size.height,
        image_size.width,
        image_size.height,
        args.calib_dir
    )
    calib_data = sio.loadmat(calib_fn,squeeze_me=True)
    return calib_data

def init_calibration(calib_data, image_size, args):

    # R = calib_data["rotationOfCamera2"]

    # T = calib_data["translationOfCamera2"].reshape(-1,1)

    # g = tf_utils.ginv(tf_utils.gen_g(R,T))

    # R = cv2.Rodrigues(g[:3,:3])[0]

    # T = g[:3,3]

    R = cv2.Rodrigues(calib_data["rotationOfCamera2"])[0]

    T = calib_data["translationOfCamera2"]

    cameraMatrix_left  = calib_data["intrinsicMatrix1"]

    cameraMatrix_right = calib_data["intrinsicMatrix2"]

    distCoeffs_left    = calib_data["distortionCoefficients1"]

    distCoeffs_right   = calib_data["distortionCoefficients2"]

    R1 = R2 = P1 = P2 = np.array([])

    if args.calib_dir == "L2R":
        R1, R2, P1, P2, Q = cv2.stereoRectify(
            cameraMatrix1=cameraMatrix_left,
            cameraMatrix2=cameraMatrix_right,
            distCoeffs1=distCoeffs_left,
            distCoeffs2=distCoeffs_right,
            R=R, T=T,
            #   flags=0,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
            imageSize=(image_size.width, image_size.height),
            newImageSize=(image_size.width, image_size.height)
        )[0:5]
    elif args.calib_dir == "R2L":
        R2, R1, P2, P1, Q = cv2.stereoRectify(
            cameraMatrix1=cameraMatrix_right,
            cameraMatrix2=cameraMatrix_left,
            distCoeffs1=distCoeffs_right,
            distCoeffs2=distCoeffs_left,
            R=R, T=T,
            #   flags=0,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
            imageSize=(image_size.width, image_size.height),
            newImageSize=(image_size.width, image_size.height)
        )[0:5]
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, (image_size.width, image_size.height), cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, (image_size.width, image_size.height), cv2.CV_32FC1)

    return P1, P2, Q, map_left_x, map_left_y, map_right_x, map_right_y

if __name__ == "__main__":
    rospy.init_node("preprocess_ecm_calib_data")

    parser = argparse.ArgumentParser(
        prog='Preprocess_Cam_Calib',
        description='Stereo Calibrate the endosope',
        epilog='Text at the bottom of help')
    parser.add_argument('-d', '--dir', type=str, help='directory of the mono calibration files...')
    parser.add_argument('-cd', '--calib_dir',   type=str,  help='Direction of the calibration: L2R or R2L...')
    parser.add_argument('-c', '--cam_type', type=str, help='Choose the endoscope type: 0 or 30')
    parser.add_argument('-r', '--res', type=str, help='Choose one of the following resolution: HD1080, HD720, VGA, AUTO')
    args = parser.parse_args()

    image_size = ecm_utils.Resolution(args.res)

    calib_path = args.dir + '/' + args.cam_type

    calib_data = load_calibration_file(calib_path,image_size,args)
    if calib_data  == "":
        exit(1)
    print("Calibration file found. Loading...")

    PL, PR, Q, map_left_x, map_left_y, map_right_x, map_right_y = init_calibration(calib_data, image_size, args)

    print("Saving Calibration File for OpenCV...")

    calib_dict = {
        "ecm_left_rect_K" : PL[:,:3],
        "ecm_right_rect_K": PR[:,:3],
        "ecm_left_rect_P" : PL,
        "ecm_right_rect_P": PR,
        "ecm_R"           : calib_data["rotationOfCamera2"],
        "ecm_T"           : calib_data["translationOfCamera2"].reshape(-1,1),
        "ecm_Q"           : Q,
        "ecm_map_left_x"  : map_left_x,
        "ecm_map_left_y"  : map_left_y,
        "ecm_map_right_x" : map_right_x,
        "ecm_map_right_y" : map_right_y
    }

    filepath = calib_path+"/{}x{}/".format(image_size.width,image_size.height)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filename = "ECM_STEREO_{}x{}_{}_calib_data_matlab.pkl".format(image_size.width,image_size.height,args.calib_dir)

    with open(filepath+filename, 'wb') as handle:
        pickle.dump(calib_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)