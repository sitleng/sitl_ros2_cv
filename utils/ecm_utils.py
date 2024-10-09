#!/usr/bin/env python3

import os
import cv2
import numpy as np
import pickle
from scipy import io
from sensor_msgs.msg import CameraInfo
from utils import cv_cuda_utils

class Resolution():
    def __init__(self, res_type):
        if res_type == "HD1080":
            self.width  = 1920
            self.height = 1080
        elif res_type == "HD900":
            self.width  = 1440
            self.height = 900
        elif res_type == "HD720":
            self.width  = 1280
            self.height = 720
        elif res_type == "VGA":
            self.width  = 640
            self.height = 480
        else:
            print("Setting default Resolution: HD720")
            self.width = 1280
            self.height = 720

def load_base_params(
        gpu_flag=False,
        cam_type=30,
        calib_dir="L2R",
        calib_type="opencv",
        resolution="HD720",
        calib_path="/home/" + os.getlogin() + "/ecm_si_calib_data",
    ):
    return {
        "gpu_flag"   : gpu_flag,
        "cam_type"   : cam_type,
        "calib_dir"  : calib_dir,
        "calib_type" : calib_type,
        "resolution" : resolution,
        "calib_path" : calib_path,
    }

def init_camera(params, res):
    camera = cv2.VideoCapture(params["cam_id"], cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    # camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','P','4','V'))
    camera.set(cv2.CAP_PROP_FPS, params["fps"])
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, res.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, res.height)
    camera.set(cv2.CAP_PROP_BRIGHTNESS, params["brightness"]) # default = -11
    camera.set(cv2.CAP_PROP_CONTRAST, params["contrast"])   # default = 148
    camera.set(cv2.CAP_PROP_SATURATION, params["saturation"]) # default = 180
    return camera

def load_raw_caminfo(node_name, params, res):
    caminfo_dict = {}
    caminfo_dict["width"]  = res.width
    caminfo_dict["height"] = res.height
    if "left" in node_name:
        caminfo = "caminfoL"
        caminfo_dict["frame_id"] = "ecm_left"
    elif "right" in node_name:
        caminfo = "caminfoR"
        caminfo_dict["frame_id"] = "ecm_right"
    if params["calib_type"] == "matlab":
        calib_fn = "{}/{}/{}x{}/{}x{}_{}_caminfo.mat".format(
            params["calib_path"],
            params["cam_type"],
            res.width,res.height,
            res.width,res.height,
            params["calib_dir"]
        )
        calib_data = io.loadmat(calib_fn,squeeze_me=True)
        caminfo_dict["distortion_model"] = calib_data[caminfo]["DistortionModel"].item()
        caminfo_dict["D"]                = calib_data[caminfo]["D"].item()
        caminfo_dict["K"]                = calib_data[caminfo]["K"].item()
        caminfo_dict["R"]                = calib_data[caminfo]["R"].item()
        caminfo_dict["P"]                = calib_data[caminfo]["P"].item()
    elif params["calib_type"] == "opencv":
        calib_fn = "{}/{}/{}x{}/ECM_MONO_{}x{}_{}_calib_data_opencv.pkl".format(
            params["calib_path"],params["cam_type"],
            res.width,res.height,res.width,res.height,
            params["calib_dir"]
        )
        with open(calib_fn, "rb") as f:
            calib_data = pickle.load(f)
        caminfo_dict["distortion_model"] = calib_data[caminfo]["DistortionModel"]
        caminfo_dict["D"]                = tuple(calib_data[caminfo]["D"].reshape(-1))
        caminfo_dict["K"]                = tuple(calib_data[caminfo]["K"].reshape(-1))
        caminfo_dict["R"]                = tuple(calib_data[caminfo]["R"].reshape(-1))
        caminfo_dict["P"]                = tuple(calib_data[caminfo]["P"].reshape(-1))
    return caminfo_dict

def load_rect_maps(node_name, params):
    res = Resolution(params["resolution"])
    calib_fn = "{}/{}/{}x{}/ECM_STEREO_{}x{}_{}_calib_data_{}.pkl".format(
        params["calib_path"],
        params["cam_type"],
        res.width,
        res.height,
        res.width,
        res.height,
        params["calib_dir"],
        params["calib_type"]
    )
    with open(calib_fn, "rb") as f:
        calib_data = pickle.load(f)
    if "left" in node_name:
        if not params["gpu_flag"]:
            map_x = calib_data["ecm_map_left_x"]
            map_y = calib_data["ecm_map_left_y"]
        else:
            map_x = cv_cuda_utils.cvmat2gpumat(calib_data["ecm_map_left_x"])
            map_y = cv_cuda_utils.cvmat2gpumat(calib_data["ecm_map_left_y"])
    elif "right" in node_name:
        if not params["gpu_flag"]:
            map_x = calib_data["ecm_map_right_x"]
            map_y = calib_data["ecm_map_right_y"]
        else:
            map_x = cv_cuda_utils.cvmat2gpumat(calib_data["ecm_map_right_x"])
            map_y = cv_cuda_utils.cvmat2gpumat(calib_data["ecm_map_right_y"])
    return map_x, map_y

def gen_caminfo(caminfo_dict, t):
    msg = CameraInfo()
    msg.header.frame_id  = caminfo_dict["frame_id"]
    msg.header.stamp     = t
    msg.width            = caminfo_dict["width"]
    msg.height           = caminfo_dict["height"]
    msg.distortion_model = caminfo_dict["distortion_model"]
    msg.d                = caminfo_dict["D"]
    msg.k                = caminfo_dict["K"]
    msg.r                = caminfo_dict["R"]
    msg.p                = caminfo_dict["P"]
    return msg

def load_stereo_calib(params, width, height):
    calib_fn = "{}/{}/{}x{}/ECM_STEREO_{}x{}_{}_calib_data_{}.pkl".format(
        params["calib_path"],
        params["cam_type"],
        width, height,
        width, height,
        params["calib_dir"],
        params["calib_type"]
    )
    with open(calib_fn, "rb") as f:
        calib_data = pickle.load(f)
    Q = calib_data["ecm_Q"].astype(np.float32)
    R = calib_data["ecm_R"].astype(np.float32)
    T = calib_data["ecm_T"].astype(np.float32)
    f = Q[2,3]
    B = -1/Q[3,2]
    return Q, R, T, B.astype(np.float32), f.astype(np.float32)