#!/usr/bin/env python3

import cv2
import math
import numpy as np
np.seterr(all="ignore")
import yaml
from scipy.spatial.transform import Rotation as R

from utils import tf_utils

def load_aruco_detector():
    # check https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
    aruco_dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    aruco_params   = cv2.aruco.DetectorParameters()
    aruco_params.adaptiveThreshWinSizeStep             = 5    # d: 10
    aruco_params.adaptiveThreshWinSizeMin              = 3    # d: 3
    aruco_params.adaptiveThreshWinSizeMax              = 23   # d: 23
    aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.15 # d: 0.13
    # aruco_params.aprilTagMinWhiteBlackDiff             = 5
    aruco_params.cornerRefinementMethod                = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_params.cornerRefinementWinSize               = 3
    aruco_params.cornerRefinementMaxIterations         = 1000
    aruco_params.cornerRefinementMinAccuracy           = 0.001
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict,aruco_params)
    return aruco_detector

def load_default_tfs(calib_arm_name):
    if calib_arm_name == "PSM1" or calib_arm_name == "PSM2":
        g_ecm_dvrk_cv2 = tf_utils.cv2vecs2g(np.array([0,0,1])*math.radians(180),np.array([0,0,0])).dot(
            tf_utils.cv2vecs2g(np.array([1,0,0])*math.radians(30),np.array([0,0,0])))
        g_psmtip_aruco_dvrk = tf_utils.cv2vecs2g(np.array([0,1,0])*math.radians(-90),np.array([0,0,0]))
        return g_ecm_dvrk_cv2, g_psmtip_aruco_dvrk
    elif calib_arm_name == "ECM":
        g_ecm_aruco_dvrk    = tf_utils.cv2vecs2g(np.array([1,0,0])*math.radians(30),np.array([0,0,0]))
        g_psmtip_aruco_dvrk = tf_utils.cv2vecs2g(np.array([0,1,0])*math.radians(-90),np.array([0,0,0]))
        return g_ecm_aruco_dvrk, g_psmtip_aruco_dvrk
    else:
        print("Give the correct name of the calibration arm...")
        

def load_tf_data(tf_data_path):
    with open(tf_data_path, 'r') as file:
        tf_data = yaml.safe_load(file)
    return tf_data

def load_caminfo(caminfomsg):
    cam_mtx  = np.asarray(caminfomsg.K).reshape(3,3)
    cam_dist = np.asarray(caminfomsg.D)
    return cam_mtx, cam_dist

def avg_aruco_rvecs(rvecs):
    r = R.from_rotvec(rvecs)
    return r.mean().as_rotvec()

def fix_rot(prev_rvecs,cur_rvec):
    if prev_rvecs.ndim == 1:
        return cur_rvec
    elif np.linalg.norm(cv2.Rodrigues(cur_rvec)[0].dot(cv2.Rodrigues(avg_aruco_rvecs(prev_rvecs))[0])) > 1e-6:
        return avg_aruco_rvecs(prev_rvecs)
    else:
        return cur_rvec