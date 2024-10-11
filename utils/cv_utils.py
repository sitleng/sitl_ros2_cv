#!/usr/bin/env python3

import cv2
import copy
import numpy as np
from sklearn.cluster import DBSCAN
from skimage.morphology import medial_axis, convex_hull_image
from scipy.ndimage import binary_fill_holes

from utils import misc_utils

def scatter(img, pts, radius, color):
    out = copy.deepcopy(img)
    if len(pts.shape) == 1:
        cv2.circle(out, tuple(pts), radius, color, -1)
    else:
        for pt in pts:
            cv2.circle(out, tuple(pt), radius, color, -1)
    return out

def remove_glare(img, thr, morph_kernel, morph_iter, inpaint_r):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, morph_kernel, iterations=morph_iter)
    return cv2.inpaint(img, mask, inpaintRadius=inpaint_r, flags=cv2.INPAINT_TELEA)

def mask_cnts(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    polys = []
    for cnt in cnts:
        poly = np.int32(cnt.reshape(-1,2))
        polys.append(poly)
    return np.concatenate(polys)

def max_contour(contours):
    max_cnt_idx = None
    max_cnt_area = 0
    for i, cnt in enumerate(contours):
        cur_cnt_area = cv2.contourArea(cnt)
        if cur_cnt_area > max_cnt_area:
            max_cnt_idx = i
            max_cnt_area = cur_cnt_area
    if max_cnt_idx is None:
        return None, None
    return max_cnt_idx, contours[max_cnt_idx].reshape(-1,2)

def cnt_centroid(cnt):
    if cnt is None:
        return None
    mu = cv2.moments(cnt)
    mc = np.array([mu['m10'] / (mu['m00'] + 1e-6), mu['m01'] / (mu['m00'] + 1e-6)], dtype=np.int32)
    return mc

def skeleton_corners_harris(skel, blockSize=9, ksize=3, k=0.1):
    dst = cv2.cornerHarris(skel, blockSize, ksize, k)
    return np.stack(np.where(dst.T > 0.1*dst.max()), axis=1)

def skeleton_corners_good(skel, max_corners=10, qual_level=0.3, min_dist=25):
    return cv2.goodFeaturesToTrack(
        skel, max_corners, qual_level, min_dist
    ).reshape(-1,2).astype(np.int32)

def mask_skeleton(mask, max_dist_ratio):
    skel, distance = medial_axis(mask, return_distance=True)
    img_cols, img_rows = np.where(distance == distance.max())
    ctrd = np.uint16([img_rows.mean(), img_cols.mean()])
    skel_inds = np.stack(np.where(skel.T > 0), axis=1)
    skel_dist = distance[skel_inds[:,1], skel_inds[:,0]]
    skel_inds = skel_inds[skel_dist > max_dist_ratio*skel_dist.max()]
    new_skel = np.zeros_like(skel, dtype=np.uint8)
    new_skel[skel_inds[:,1], skel_inds[:,0]] = 255
    corners = skeleton_corners_harris(new_skel, blockSize=25, ksize=3)
    new_skel[corners[:,1], corners[:,0]] = 0
    skel_inds = np.stack(np.where(new_skel.T > 0), axis=1)
    return skel_inds, ctrd

def branch_len(branch):
    return cv2.arcLength(branch, False)

def skeleton_branches(skel_inds):
    model = DBSCAN(eps=2, min_samples=2)
    labels = model.fit_predict(skel_inds)
    uniq_lables = np.unique(labels)
    branches = []
    for label in uniq_lables:
        branches.append(skel_inds[labels == label])
    branches.sort(key=branch_len, reverse=True)
    return branches

def prune_skeleton(skel_inds, ang_thr=20):
    branches = skeleton_branches(skel_inds)
    if len(branches) == 1:
        return skel_inds
    branch_vecs = [misc_utils.unit_vector(branch[0] - branch[-1]) for branch in branches]
    branch_angs = []
    for branch_vec in branch_vecs:
        angle = misc_utils.angle_btw_vecs(branch_vecs[0], branch_vec)
        if angle > np.radians(90):
            angle = misc_utils.angle_btw_vecs(branch_vecs[0], -branch_vec)
        branch_angs.append(angle)
    valid_branch = branch_angs <= np.radians(ang_thr)
    pruned_branches = []
    for branch, validity in zip(branches, valid_branch):
        if validity:
            pruned_branches.append(branch)
    res_skel = np.concatenate(pruned_branches)
    img_corner = np.array([1280, 720])
    skel_angles = np.array([np.arctan2(x[1], x[0]) for x in res_skel - img_corner])
    res_skel = res_skel[np.argsort(skel_angles)]
    res_skel = misc_utils.interp_2d(res_skel, 0.01, False)
    return res_skel
    

# Just use skimage.morhpology.convex_hull_mask instead
def mask_convex_hull(mask):
    return np.uint8(convex_hull_image(mask)*255)

def erode_mask(mask, kernel_type=cv2.MORPH_RECT, kernel_size=5, iter=1):
    kernel = cv2.getStructuringElement(kernel_type, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=iter)

def dilate_mask(mask, kernel_type=cv2.MORPH_RECT, kernel_size=5, iter=1):
    kernel = cv2.getStructuringElement(kernel_type, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=iter)

def close_mask(mask, kernel_type=cv2.MORPH_RECT, kernel_size=5, iter=1):
    kernel = cv2.getStructuringElement(kernel_type, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iter)

def open_mask(mask, kernel_type=cv2.MORPH_RECT, kernel_size=5, iter=1):
    kernel = cv2.getStructuringElement(kernel_type, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iter)

def fill_mask(mask):
    return np.uint8(binary_fill_holes(mask)*255)