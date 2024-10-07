#!/usr/bin/env python3

import cv2
import numpy as np

def cvmat2gpumat(image):
    image_cuda = cv2.cuda_GpuMat()
    image_cuda.upload(image)
    return image_cuda

def cvmat_resize(image, dsize, fx=0, fy=0, intp_flag=cv2.INTER_LINEAR):
    gpu_in = cvmat2gpumat(image)
    gpu_out = cv2.cuda_GpuMat()
    cv2.cuda.resize(gpu_in, dsize, gpu_out, fx, fy, intp_flag)
    return gpu_out.download()

def load_cam1_sgm(params):
    return cv2.cuda.createStereoSGM(
        minDisparity    = params["min_disp"],
        numDisparities  = params["sgm_ndisp"], 
        P1              = params["P1"],
        P2              = params["P2"],
        uniquenessRatio = params["uniq_ratio"],
        mode            = cv2.StereoSGBM_MODE_HH
    )

def load_cam2_sgm(params):
    return cv2.cuda.createStereoSGM(
        minDisparity    = params["min_disp"] - params["sgm_ndisp"],
        numDisparities  = params["sgm_ndisp"], 
        P1              = params["P1"],
        P2              = params["P2"],
        uniquenessRatio = params["uniq_ratio"],
        mode            = cv2.StereoSGBM_MODE_HH
    )

def load_dbf(params):
    return cv2.cuda.createDisparityBilateralFilter(
        ndisp=params["dbf_ndisp"],
        radius=params["radius"],
        iters=params["iters"]
    )

def load_wls_filter(cam1_sgm, params):
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(cam1_sgm)
    wls_filter.setLambda(params["wls_lambda"])
    wls_filter.setSigmaColor(params["wls_sigma"])
    return wls_filter

def apply_bf(img, kernel_size):
    return cv2.cuda.bilateralFilter(
        cvmat2gpumat(img),
        kernel_size=kernel_size,
        sigma_color=kernel_size*2,
        # sigma_spatial=int(kernel_size/2)
        sigma_spatial=int(kernel_size**2)
    ).download()

def cuda_sgm_dbf(cam1_sgm, cam1_rect_mono, cam2_rect_mono, dbf_cuda=None):
    cam1_rect_mono_cuda = cvmat2gpumat(cam1_rect_mono)
    cam2_rect_mono_cuda = cvmat2gpumat(cam2_rect_mono)
    disp_sgm_cuda = cv2.cuda_GpuMat()
    disp_sgm_cuda = cam1_sgm.compute(cam1_rect_mono_cuda,cam2_rect_mono_cuda,disp_sgm_cuda)
    if dbf_cuda is not None:
        disp_sgm_cuda = dbf_cuda.apply(disp_sgm_cuda, cam1_rect_mono_cuda)
    disp_sgm_cuda = cv2.cuda.normalize(disp_sgm_cuda, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    return disp_sgm_cuda.download()

def cuda_sgm_wls_filter(cam1_sgm, cam2_sgm, cam1_rect_mono_cuda, cam2_rect_mono_cuda, wls_filter=None):
    disp_cam1 = cv2.cuda_GpuMat()
    disp_cam2 = cv2.cuda_GpuMat()
    disp_cam1 = cam1_sgm.compute(cam1_rect_mono_cuda, cam2_rect_mono_cuda, disp_cam1)
    disp_cam1_norm = cv2.cuda.normalize(disp_cam1, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    if wls_filter is not None:
        disp_cam2 = cam2_sgm.compute(cam2_rect_mono_cuda, cam1_rect_mono_cuda, disp_cam2)
        disp_cam2_norm = cv2.cuda.normalize(disp_cam2, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
        disp_filt = cvmat2gpumat(
            wls_filter.filter(disp_cam1_norm.download(), cam1_rect_mono_cuda.download(), disparity_map_right=disp_cam2_norm.download())
        )
    else:
        disp_filt = disp_cam1_norm
    return disp_filt

def preprocess_disp(disp_cuda):
    disp_cv = np.float32(disp_cuda.download()/16)
    disp_cuda.upload(disp_cv)
    return disp_cuda