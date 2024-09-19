#! /usr/bin/env python3

import numpy as np
import os
import argparse
import cv2
import pickle
import rospy
import glob
import scipy.io as sio
from utils import ecm_utils, tf_utils

class StereoCalibration(object):
    def __init__(self,args):
        self.mono_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.stereo_criteria_cal = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((8*5, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)
        self.objp *= args.square_size

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.image_size = ecm_utils.Resolution(args.res)
        self.img_shape = (self.image_size.width,self.image_size.height)

        self.calib_path = args.dir + "/" + args.cam_type + "/{}x{}".format(self.image_size.width,self.image_size.height)
        self.args = args

    def mono_calibration(self):
        rospy.loginfo("Start Mono Calibration...")
        rospy.loginfo("Loading images for calibration...")
        calib_imgs_fn = "{}/{}_calib_images.mat".format(
            self.calib_path, self.args.calib_dir
        )
        calib_imgs = sio.loadmat(calib_imgs_fn,squeeze_me=True)["img_cell"]
        if "left" in calib_imgs[0,0]:
            images_left  = calib_imgs[0,:]
            images_right = calib_imgs[1,:]
        else:
            images_left  = calib_imgs[1,:]
            images_right = calib_imgs[0,:]
        images_left.sort()
        images_right.sort()
        rospy.loginfo("Loading images complete!")
        rospy.loginfo("Detecting chessboards in the images...")
        for i in range(len(images_right)):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (8,5), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (8,5), None)

            # If found, add object points, image points (after refining them)
            if ret_l and ret_r:
                self.objpoints.append(self.objp)
                corners_l_rf = cv2.cornerSubPix(gray_l,corners_l,(11,11),(-1,-1),self.mono_criteria)
                self.imgpoints_l.append(corners_l_rf)
                corners_r_rf = cv2.cornerSubPix(gray_r,corners_r,(11,11),(-1,-1),self.mono_criteria)
                self.imgpoints_r.append(corners_r_rf)
        rospy.loginfo("Finished detecting chessboards in the images!")
        rospy.loginfo("Start Calibrating Left Camera...")
        rt, self.ML, self.dL, self.rL, self.tL = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, self.img_shape, None, None)
        rospy.loginfo("Finished Calibrating Left Camera!")
        rospy.loginfo("Start Calibrating Right Camera...")
        rt, self.MR, self.dR, self.rR, self.tR = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, self.img_shape, None, None)
        rospy.loginfo("Finished Calibrating Right Camera!")
        N = len(self.objpoints)
        mean_error_l = 0
        mean_error_r = 0
        for i in range(N):
            imgpoints_l2, _ = cv2.projectPoints(self.objpoints[i], self.rL[i], self.tL[i], self.ML, self.dL)
            error_l = cv2.norm(self.imgpoints_l[i], imgpoints_l2, cv2.NORM_L2)/len(imgpoints_l2)
            mean_error_l += error_l
            imgpoints_RR, _ = cv2.projectPoints(self.objpoints[i], self.rR[i], self.tR[i], self.MR, self.dR)
            error_r = cv2.norm(self.imgpoints_r[i], imgpoints_RR, cv2.NORM_L2)/len(imgpoints_RR)
            mean_error_r += error_r
        rospy.loginfo("Average Reprojection Error (Left): {}".format(mean_error_l/N))
        rospy.loginfo("Average Reprojection Error (Right): {}".format(mean_error_r/N))

    def stereo_calibrate(self):
        flags = 0
        # flags |= cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_USE_EXTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_THIN_PRISM_MODEL
        # flags |= cv2.CALIB_FIX_S1_S2_S3_S4
        # flags |= cv2.CALIB_TILTED_MODEL
        # flags |= cv2.CALIB_FIX_TAUX_TAUY

        rospy.loginfo("Start Stereo Calibration...")
        if self.args.calib_dir == "L2R":
            ret,self.ML,self.dL,self.MR,self.dR,self.R,self.T,self.E,self.F = cv2.stereoCalibrate(
                self.objpoints,self.imgpoints_l,self.imgpoints_r,self.ML,self.dL,self.MR,self.dR,
                self.img_shape,criteria=self.stereo_criteria_cal,flags=flags)
        elif self.args.calib_dir == "R2L":
            ret,self.MR,self.dR,self.ML,self.dL,self.R,self.T,self.E,self.F = cv2.stereoCalibrate(
                self.objpoints,self.imgpoints_r,self.imgpoints_l,self.MR,self.dR,self.ML,self.dL,
                self.img_shape,criteria=self.stereo_criteria_cal,flags=flags)
        rospy.loginfo("Finished Stereo Calibration!")
        rospy.loginfo("Final Reprojection Error of Stereo Calibration:{}".format(ret))
        if self.args.calib_dir == "L2R":
            self.RL, self.RR, self.PL, self.PR, self.Q = cv2.stereoRectify(
                cameraMatrix1=self.ML,
                cameraMatrix2=self.MR,
                distCoeffs1=self.dL,
                distCoeffs2=self.dR,
                R=self.R, T=self.T,
                # flags=0,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0,
                imageSize=self.img_shape,
                newImageSize=self.img_shape)[0:5]
        elif self.args.calib_dir == "R2L":
            self.RR, self.RL, self.PR, self.PL, self.Q = cv2.stereoRectify(
                cameraMatrix1=self.MR,
                cameraMatrix2=self.ML,
                distCoeffs1=self.dR,
                distCoeffs2=self.dL,
                R=self.R, T=self.T,
                # flags=0,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0,
                imageSize=self.img_shape,
                newImageSize=self.img_shape)[0:5]
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(self.ML, self.dL, self.RL, self.PL, self.img_shape, cv2.CV_32FC1)
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(self.MR, self.dR, self.RR, self.PR, self.img_shape, cv2.CV_32FC1)

if __name__ == "__main__":
    rospy.init_node("stereo_calib_ecm_opencv")

    parser = argparse.ArgumentParser(
        prog='stereo_calib_ecm_opencv',
        description='Stereo Calibrate the endosope with opencv',
        epilog='Text at the bottom of help')
    parser.add_argument('-d',  '--dir',         type=str,  help='directory of the calibration images...')
    parser.add_argument('-cd', '--calib_dir',   type=str,  help='Direction of the calibration: L2R or R2L...')
    parser.add_argument('-c',  '--cam_type',    type=str,  help='Choose the endoscope type: 0 or 30')
    parser.add_argument('-r',  '--res',         type=str,  help='Choose one of the following resolution: HD1080, HD720, VGA, AUTO')
    parser.add_argument('-s',  '--square_size', type=int,  help='The square size of the chessboard: e.g. 15 (mm)')
    args = parser.parse_args()
    try:
        app = StereoCalibration(args)
        app.mono_calibration()
        app.stereo_calibrate()
        rospy.loginfo("Saving Mono Calibration File for OpenCV...")
        left_dict = {
            "DistortionModel": "plumb_bob",
            "D": app.dL,
            "K": app.ML,
            "R": app.RL,
            "P": app.PL
        }
        right_dict = {
            "DistortionModel": "plumb_bob",
            "D": app.dR,
            "K": app.MR,
            "R": app.RR,
            "P": app.PR
        }
        mono_calib_dict = {
            "caminfoL": left_dict,
            "caminfoR": right_dict
        }
        filename = "ECM_MONO_{}x{}_{}_calib_data_opencv.pkl".format(
            app.image_size.width,app.image_size.height,args.calib_dir
        )
        with open(app.calib_path+"/"+filename, 'wb') as handle:
            pickle.dump(mono_calib_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        rospy.loginfo("Saving Stereo Calibration File for OpenCV...")
        stereo_calib_dict = {
            "ecm_left_rect_K" : app.PL[:,:3],
            "ecm_right_rect_K": app.PR[:,:3],
            "ecm_left_rect_P" : app.PL,
            "ecm_right_rect_P": app.PR,
            "ecm_R"           : app.R,
            "ecm_T"           : app.T,
            "ecm_Q"           : app.Q,
            "ecm_map_left_x"  : app.map_left_x,
            "ecm_map_left_y"  : app.map_left_y,
            "ecm_map_right_x" : app.map_right_x,
            "ecm_map_right_y" : app.map_right_y
        }
        filename = "ECM_STEREO_{}x{}_{}_calib_data_opencv.pkl".format(
            app.image_size.width,app.image_size.height,args.calib_dir
        )
        with open(app.calib_path+"/"+filename, 'wb') as handle:
            pickle.dump(stereo_calib_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        rospy.loginfo(e)