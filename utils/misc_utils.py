#!/usr/bin/env python3

import os
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.interpolate import PchipInterpolator
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

def check_empty_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def farthest_first_traversal(points, num_selected_points):
    if points is None or num_selected_points > len(points):
        return None

    selected_indices = [0]
    distances = np.linalg.norm(points - points[selected_indices[0]], axis=1)

    for _ in range(1, num_selected_points):
        new_point_index = np.argmax(distances)
        selected_indices.append(new_point_index)
        new_distances = np.linalg.norm(points - points[new_point_index], axis=1)
        distances = np.minimum(distances, new_distances)

    selected_points = points[sorted(selected_indices)]
    return selected_points

def nn_kdtree(tree_pts, query_pts, dub):
    tree = cKDTree(tree_pts)
    dist, index = tree.query(query_pts, distance_upper_bound=dub, k=1, workers=-1)
    return dist, index

def sort_pts_dist(pts_arr):
    if pts_arr.ndim != 2:
        return pts_arr
    dist = cdist(pts_arr, pts_arr)
    max_dist_pair = np.unravel_index(np.argmax(dist), dist.shape)
    # Trajectory is sorted from right to left of the 2D image
    if pts_arr[max_dist_pair[0]][0] > pts_arr[max_dist_pair[1]][0]:
        dist = dist[max_dist_pair[0],:]
    else:
        dist = dist[max_dist_pair[1],:]
    return pts_arr[np.argsort(dist)]

def prop_pt(p1,p2,d):
    return p1 + d/np.linalg.norm(p1-p2)*(p2-p1)

def unit_vector(v):
    return v/np.linalg.norm(v)

def unit_vecs(vecs):
    return vecs/np.linalg.norm(vecs, axis=1)[:, np.newaxis]

def angle_btw_vecs(v1, v2):
    return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2.T))

def law_of_cos_ang(x1, x2, y):
    return np.arccos((x1**2 + x2**2 - y**2)/(2*x1*x2))

def curve_length(curve):
    return np.sum(np.linalg.norm(curve[1:] - curve[:-1], axis=1))

# Select contour points based on the contour centroid, major axis, and the threshold angle
def cnt_pts_angle_vector(pca_comps, cnts, vec, thr_angle):
    ct = cnts.mean(axis=0)
    v_n = unit_vector(np.cross(pca_comps[0], pca_comps[1]))
    ct2cnts = unit_vecs(cnts - ct)
    angles = np.arctan2(np.dot(np.cross(vec, ct2cnts), v_n), np.dot(vec, ct2cnts.T))
    return cnts[abs(angles) <= np.radians(thr_angle)]

def cnt_axes_3d(cnt):
    pca = PCA(n_components=3)
    pca.fit(cnt)
    # axes = [np.array([1, 0, 0]), np.array([0, 1, 0])]
    # dots = [np.abs(np.dot(pca.components_, ax)) for ax in axes]
    # inds = [np.argmax(dot) for dot in dots]
    # pca_comps = pca.components_[inds]
    pca_comps = pca.components_
    for i, pca_comp in enumerate(pca_comps):
        if pca_comp[i] < 0:
            pca_comps[i] = -pca_comp
    # comp_z = unit_vector(np.cross(pca_comps[0], pca_comps[1]))
    # if comp_z[-1] < 0:
    #     comp_z = -comp_z
    # pca_comps = np.vstack([pca_comps, comp_z])
    return pca_comps

def seq_dists(points):
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dists = np.insert(dists, 0, 0)
    return dists

def proj_vec_to_plane(a, u, v):
    n = np.cross(u, v)
    p_n = (np.dot(a, n) / np.dot(n, n)) * n
    return unit_vector(a - p_n)

def interp_2d(orig_points, num_interp_ratio=0.1, is_closed=True):
    if is_closed:
        orig_points = np.append(orig_points, orig_points[0].reshape(1,2), axis=0)
    dists = seq_dists(orig_points)
    filt_orig_pts = orig_points[~np.isclose(dists, 0)]
    n_orig = np.cumsum(dists[~np.isclose(dists, 0)])
    n_orig = n_orig/n_orig[-1]
    des_num_points = int(filt_orig_pts.shape[0]*(1 + num_interp_ratio))
    n_des  = np.linspace(0, 1, num = des_num_points)
    interpolated_points = np.zeros((des_num_points, 2), dtype=np.int32)
    for i in range(2):
        interp_func = PchipInterpolator(n_orig, filt_orig_pts[:,i])
        interpolated_points[:,i] = interp_func(n_des)
    return interpolated_points

def interp_3d(orig_points, num_interp_ratio=0.1, is_closed=True):
    if orig_points is None or orig_points.size < 2:
        return None
    if is_closed:
        orig_points = np.append(orig_points, orig_points[0].reshape(1,3), axis=0)
    dists = seq_dists(orig_points)
    filt_orig_pts = orig_points[~np.isclose(dists, 0)]
    n_orig = np.cumsum(dists[~np.isclose(dists, 0)])
    n_orig = n_orig/n_orig[-1] # normalize the array so the values are between [0, 1]
    des_num_points = int(filt_orig_pts.shape[0]*(1 + num_interp_ratio))
    n_des  = np.linspace(0, 1, num = des_num_points)
    interpolated_points = np.zeros((des_num_points, 3))
    for i in range(3):  # Loop over x, y, z dimensions
        interp_func = PchipInterpolator(n_orig, filt_orig_pts[:,i])
        interpolated_points[:,i] = interp_func(n_des)
    return interpolated_points

def smooth_cnt(cnt, win_r):
    cnt_filt = np.copy(cnt)
    win_len = max([5, int(cnt.shape[0]*win_r)])
    for i in range(cnt.shape[1]):
        cnt_filt[:,i] = savgol_filter(
            cnt[:,i], window_length=win_len, polyorder=2, axis=0
        )
    return cnt_filt

def rm_outlier_mad(points, thresh=3.5):
    if points.ndim == 1:
        points = points.reshape(-1,1)
    elif points.size == 0 or points.ndim == 0:
        return points
    median = np.median(points, axis=0)
    diff = np.sqrt(np.sum((points - median)**2, axis=-1))
    med_abs_deviation = np.median(diff)
    if np.isclose(med_abs_deviation, 0):
        med_abs_deviation += 1e-6
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return np.squeeze(points[modified_z_score < thresh])

def rm_nan(x):
    return x[~np.isnan(x).all(axis=1)]

def remove_outliers_iso(points):
    if points is None:
        return None
    iso = IsolationForest(n_estimators=100, contamination="auto", max_features=points.shape[points.ndim-1], warm_start=True)
    return points[iso.fit_predict(points) == 1]

def remove_outliers_lof(points):
    if points is None:
        return None
    lof = LocalOutlierFactor(n_neighbors=int(points.shape[0]/3), leaf_size=30, metric="minkowski")
    return points[lof.fit_predict(points) == 1]

def remove_outliers_dbs(points):
    if points is None:
        return None
    dbscan = DBSCAN(eps=seq_dists(points).mean(), min_samples=2)
    labels = dbscan.fit_predict(points)
    return points[labels > labels.mean()]

def proj_curve_to_line(r, curve, pt):
    curve_len = curve_length(curve)
    curve2pt_vecs = pt - curve
    mid_pt_vecs = curve + curve2pt_vecs*r
    mid_pt = np.mean(mid_pt_vecs, axis=0)
    
    # Calculate the direction vector for the stretched line as before
    endpt_vec = curve[-1] - curve[0]
    endpt_vec /= np.linalg.norm(endpt_vec)
    
    # Use the new midpoint and the same direction to define the stretched line
    return np.linspace(
        mid_pt - endpt_vec * curve_len / 2,
        mid_pt + endpt_vec * curve_len / 2,
        curve.shape[0]
    )

def orthogonal_vector(pt, line):
    if pt.shape[-1] != line.shape[-1]:
        print("Wrong dimension between the point and the line")
        return None
    line_vec = line[-1]-line[0]
    t = np.dot(line_vec, pt-line[0])/np.dot(line_vec, line_vec)
    return pt-(line[0]+t*line_vec)

def intersect_boxes(box1, box2):
    # Unpack the coordinates of the two boxes
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate the coordinates of the intersection box
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    # Check if the intersection is valid (i.e., if the boxes overlap)
    if xi1 <= xi2 and yi1 <= yi2:
        # Return the coordinates of the intersection box
        return np.array([xi1, yi1, xi2, yi2], dtype=int)
    else:
        # Return zero if there is no valid intersection (i.e., the boxes do not overlap)
        return np.zeros(4)

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def find_overlap_perc(box1, box2):
    overlap_box = intersect_boxes(box1, box2)
    return (bbox_area(overlap_box) / bbox_area(box2)) * 100

def restore_cropped_pixels(crop_box, cropped_pixels):
    # Extract the top-left coordinates of the crop_box
    if cropped_pixels.ndim == 1:
        return np.array([cropped_pixels[0] + crop_box[0], cropped_pixels[1] + crop_box[1]]).T
    else:
        return np.array([cropped_pixels[:,0] + crop_box[0], cropped_pixels[:,1] + crop_box[1]]).T
    
def restore_cropped_masks(orig_img_size, crop_box, cropped_mask):
    orig_mask = np.zeros(orig_img_size, dtype=cropped_mask.dtype)
    orig_mask[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = cropped_mask
    return orig_mask

# def win_avg_3d_pt(cur_pt_2d, pcl_array, window_size, mad_thr):
#     r = int(window_size/2)
#     cur_pts_3d = pcl_array[
#         cur_pt_2d[1]-r:cur_pt_2d[1]+r,
#         cur_pt_2d[0]-r:cur_pt_2d[0]+r
#     ].reshape(-1,3)
#     cur_pts_3d = rm_nan(cur_pts_3d)
#     cur_pts_3d = rm_outlier_mad(cur_pts_3d, mad_thr)
#     if cur_pts_3d.ndim == 0 or cur_pts_3d.size == 0:
#         return None
#     elif cur_pts_3d.ndim == 1:
#         return cur_pts_3d
#     else:
#         return np.mean(cur_pts_3d, axis=0)