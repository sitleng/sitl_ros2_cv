#!/usr/bin/env python3

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from ros2_numpy import point_cloud2
from utils import misc_utils, cv_cuda_utils

BIT_MOVE_16 = 2**16
BIT_MOVE_8  = 2**8

def transform_pcl(pcl,g):
    h,w,d = pcl.shape
    temp = np.dstack((pcl,np.ones((h,w,1)))).reshape(h*w,d+1).T
    temp_matmul = g.dot(temp)
    return temp_matmul.T.reshape([h,w,d+1])[:,:,:d]

def xyzarr_to_nparr(xyzarr):
    nparray = np.empty(xyzarr.shape + (3,))
    if len(xyzarr.shape) == 1:
        nparray[:,0] = xyzarr['x']
        nparray[:,1] = xyzarr['y']
        nparray[:,2] = xyzarr['z']
    elif len(xyzarr.shape) == 2:
        nparray[:,:,0] = xyzarr['x']
        nparray[:,:,1] = xyzarr['y']
        nparray[:,:,2] = xyzarr['z']
    return nparray

def disp2pclimg(disp, Q, pcl_scale, depth_trunc):
    pclimg   = cv2.reprojectImageTo3D(disp, Q)*pcl_scale
    # pclimg   = pclimg/params["depth_scale"]*params["pcl_scale"]
    trunc_idx = np.linalg.norm(pclimg, axis=2) < depth_trunc
    pclimg[~trunc_idx, :] = None
    # pclimg[:,:,0][trunc_idx] = None
    # pclimg[:,:,1][trunc_idx] = None
    # pclimg[:,:,2][trunc_idx] = None
    return pclimg

def disp2pclimg_cuda(disp, Q, pcl_scale, depth_trunc):
    disp_cuda = cv_cuda_utils.cvmat2gpumat(disp)
    pclimg_cuda = cv2.cuda.reprojectImageTo3D(disp_cuda, Q, dst_cn=3)
    # pclimg      = pclimg_cuda.download()/params["depth_scale"]#*params["pcl_scale"]
    pclimg    = pclimg_cuda.download()*pcl_scale
    trunc_idx = np.linalg.norm(pclimg, axis=2) < depth_trunc
    pclimg[~trunc_idx, :] = None
    return pclimg

def gen_pcl(ref_img, pclimg, depth_scale, frame_id, stamp=None):
    pclimg = pclimg/depth_scale
    is_color = (len(ref_img.shape) > 2)
    trunc_idx = ~np.isnan(pclimg[:,:,0])
    n_points = np.sum(trunc_idx)
    # n_points = ref_img.shape[:2]
    if is_color:
        data = np.zeros(n_points, dtype=[('x', np.float32),
                                         ('y', np.float32),
                                         ('z', np.float32),
                                         ('rgb', np.uint32)])
    else:
        data = np.zeros(n_points, dtype=[('x', np.float32),
                                         ('y', np.float32),
                                         ('z', np.float32)])
    # data['x'] = pclimg[:,:,0]
    # data['y'] = pclimg[:,:,1]
    # data['z'] = pclimg[:,:,2]
    data['x'] = pclimg[:,:,0][trunc_idx]
    data['y'] = pclimg[:,:,1][trunc_idx]
    data['z'] = pclimg[:,:,2][trunc_idx]
    if is_color:
        rgb_data = ref_img[:,:,2]*BIT_MOVE_16 + ref_img[:,:,1]*BIT_MOVE_8 + ref_img[:,:,0]
        rgb_data = rgb_data.astype(np.uint32)
        # data['rgb'] = rgb_data
        data['rgb'] = rgb_data[trunc_idx]
    return point_cloud2.array_to_pointcloud2(data, frame_id=frame_id, stamp=stamp)

def pt_array_3d_to_pcl(pt_array_3d,color=(255,0,0)):
    data = np.zeros(
        pt_array_3d.shape[0],
        dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ]
    )
    data['x'] = pt_array_3d[0]
    data['y'] = pt_array_3d[1]
    data['z'] = pt_array_3d[2]
    color = np.repeat(np.array([np.asarray(color)]),pt_array_3d.shape[0],axis=0)
    rgb_data = color[:,0]*BIT_MOVE_16 + color[:,1]*BIT_MOVE_8 + color[:,2]
    rgb_data = rgb_data.astype(np.uint32)
    data['rgb'] = rgb_data
    return data

def pts_array_3d_to_pcl(pts_array_3d,color=(255,0,0)):
    data = np.zeros(
        pts_array_3d.shape[0],
        dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ]
    )
    data['x'] = pts_array_3d[:,0]
    data['y'] = pts_array_3d[:,1]
    data['z'] = pts_array_3d[:,2]
    color = np.repeat(np.array([np.asarray(color)]),pts_array_3d.shape[0],axis=0)
    rgb_data = color[:,0]*BIT_MOVE_16 + color[:,1]*BIT_MOVE_8 + color[:,2]
    rgb_data = rgb_data.astype(np.uint32)
    data['rgb'] = rgb_data
    return data

def win_avg_3d_pt(cur_pt_2d, pcl_array, window_size, mad_thr):
    r = int(window_size/2)
    cur_pts_3d = pcl_array[
        cur_pt_2d[1]-r:cur_pt_2d[1]+r,
        cur_pt_2d[0]-r:cur_pt_2d[0]+r
    ].reshape(-1,3)
    cur_pts_3d = misc_utils.rm_nan(cur_pts_3d)
    cur_pts_3d = misc_utils.rm_outlier_mad(cur_pts_3d, mad_thr)
    if cur_pts_3d.ndim == 0 or cur_pts_3d.size == 0:
        return None
    elif cur_pts_3d.ndim == 1:
        return cur_pts_3d
    else:
        return np.mean(cur_pts_3d, axis=0)

def win_avg_3d_pts(cur_pts_2d, pcl_array, window_size, mad_thr):
    res = None
    for cur_pt_2d in cur_pts_2d:
        cur_pt_3d = win_avg_3d_pt(cur_pt_2d, pcl_array, window_size, mad_thr)
        if cur_pt_3d is None:
            continue
        if res is None:
            res = cur_pt_3d[np.newaxis, :]
        else:
            res = np.vstack((res, cur_pt_3d))
    return res

def rm_outl_pts(pts, ref_pts, dist_thr):
    distances = cdist(pts, ref_pts)
    mask = np.any(distances <= dist_thr, axis=-1)
    return pts[mask]