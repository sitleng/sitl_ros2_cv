import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis

from utils import misc_utils, cv_utils, ma_utils
from sitl_ros2_interfaces.msg import SegStamped

def seg_score(seg_prob):
    return int(seg_prob.split('%')[0])/100

def mask_to_polygons(mask):
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return []
    res = res[-2]
    res = [x.flatten() for x in res]
    res = [x + 0.5 for x in res if len(x) >= 6]
    return res

def get_obj_seg(seg_mask, cnt_area_thr):
    polys = []
    seg_polygons = mask_to_polygons(seg_mask)
    for _, poly in enumerate(seg_polygons):
        poly = np.int32(poly.reshape(-1,2))
        if cv2.contourArea(poly) > cnt_area_thr:
            polys.append(poly)
    if len(polys) != 0:
        return np.concatenate(polys)
    else:
        return None
    
def get_cnt_mask(seg_cnt, img_shape):
    seg_cnt = np.int32(seg_cnt)
    mask = np.zeros(img_shape)
    try:
        cv2.drawContours(mask, [seg_cnt], -1, 255, thickness = cv2.FILLED)
    except:
        cv2.drawContours(mask, [cv_utils.cnt_convex_hull(seg_cnt)], -1, 255, thickness = cv2.FILLED)
    return mask
    
def align_cnt(cnt, clockwise=True):
    ctrd = cv_utils.cnt_centroid(cnt)
    # Flip y-coordinates to correct for image coordinate system
    cnt_angles = np.arctan2(-(cnt[:, 1] - ctrd[1]), cnt[:, 0] - ctrd[0])
    sorted_indices = np.argsort(cnt_angles)
    if clockwise:
        sorted_indices = sorted_indices[::-1]  # Reverse for clockwise order
    return cnt[sorted_indices]

def rm_cnt_outliers_pca(cnt):
    # Apply PCA
    pca = PCA(n_components=3)
    transformed_points = pca.fit_transform(cnt)  # Transform into PCA space

    # Compute Mahalanobis distance
    mean_pca = np.mean(transformed_points, axis=0)
    cov_inv = np.linalg.inv(np.cov(transformed_points.T))
    distances = np.array([mahalanobis(p, mean_pca, cov_inv) for p in transformed_points])

    # Define an outlier threshold ()
    threshold = np.percentile(distances, 70)
    new_cnt = cnt[distances <= threshold]
    return new_cnt

def gen_gallb_mask(gallb_score, gallb_cnts, liver_score, liver_cnts, img_shape, dub):
    if gallb_cnts is None or liver_cnts is None:
        return None, None, None
    gallb_cnts = misc_utils.interp_2d(gallb_cnts)
    gallb_cnts = align_cnt(gallb_cnts)
    liver_cnts = misc_utils.interp_2d(liver_cnts)
    liver_cnts = align_cnt(liver_cnts)

    dist, inds = misc_utils.nn_kdtree(liver_cnts, gallb_cnts, dub=dub)
    adj_liver = liver_cnts[inds[~np.isinf(dist)]]
    adj_gallb = gallb_cnts[~np.isinf(dist)]

    adj_cnt = np.int32(
        (adj_liver*liver_score + adj_gallb*gallb_score)/(liver_score + gallb_score)
    )

    new_gallb_cnts = np.copy(gallb_cnts)
    new_gallb_cnts[~np.isinf(dist)] = adj_cnt

    new_gallb_mask = np.zeros(img_shape)
    cv2.drawContours(new_gallb_mask, [new_gallb_cnts], -1, 255, thickness = cv2.FILLED)
    new_gallb_mask = cv2.morphologyEx(
        new_gallb_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)),
        iterations=5
    )
    new_gallb_mask = cv2.morphologyEx(
        new_gallb_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)),
        iterations=1
    )
    return adj_cnt, new_gallb_cnts, new_gallb_mask

def segment_centroids(segments):
    """Compute centroids of each segment."""
    return np.array([np.mean(seg, axis=0) for seg in segments])

def right_bottom_segment(segments):
    """Selects the segment with the centroid closest to the positive x-axis."""
    centroids = segment_centroids(segments)
    if centroids.size == 0:
        return None
    right_bottom_idx = np.argmax(centroids[:, 0] + 0.25*centroids[:, 1])
    return segments[right_bottom_idx]

# def gen_segstamped(seg_label, cnt_2d, cnt_3d, stamp):
#     msg = SegStamped()
#     msg.header.stamp = stamp
#     msg.name = seg_label
#     msg.cnt2d.layout = ma_utils.get_ma_layout(cnt_2d)
#     msg.cnt2d.data   = ma_utils.arr2data(cnt_2d)
#     msg.cnt3d.layout = ma_utils.get_ma_layout(cnt_3d)
#     msg.cnt3d.data   = ma_utils.arr2data(cnt_3d)
#     return msg

def gen_segstamped(seg_label, cnt_2d, stamp):
    msg = SegStamped()
    msg.header.stamp = stamp
    msg.name = seg_label
    msg.cnt2d.layout = ma_utils.get_ma_layout(cnt_2d)
    msg.cnt2d.data   = ma_utils.arr2data(cnt_2d)
    return msg

# def get_bnd(gb_skel_3d, adj_3d):
#     # Fit a linear model to the green points in the x-y plane
#     model = LinearRegression().fit(gb_skel_3d[:, 1:], gb_skel_3d[:, 0])
    
#     # Compute y values on the fitted line for red points' x-coordinates
#     y_line = model.predict(adj_3d[:, 1:])
    
#     # Select red points where y is greater than the line's y (i.e., on the right)
#     right_adj_3d = adj_3d[adj_3d[:, 0] > y_line]

#     return right_adj_3d

def get_bnd(gb_skel_3d, adj_3d):
    # Fit a plane to adj_3d using PCA
    adj_pca_comps = misc_utils.cnt_axes_3d(adj_3d)

    line_origin = gb_skel_3d.mean(axis=0)

    # Compute rightward direction
    right_vector = adj_pca_comps[1]
    right_vector /= np.linalg.norm(right_vector)
    if right_vector[0] < 0:
        right_vector = -right_vector

    # Find right-side points
    # right_mask = misc_utils.angle_btw_vecs(adj_3d - line_origin, right_vector) < np.radians(60)
    thr = np.linalg.norm(adj_3d - line_origin, axis=1)*np.cos(np.radians(60))
    right_mask = np.dot(adj_3d - line_origin, right_vector) > thr
    # right_mask = np.dot(adj_3d - line_origin, right_vector) > 0
    # right_mask = np.where(
    #     np.linalg.norm(np.cross(adj_3d - line_origin, right_vector), axis=1) < 0.01
    # )[0]
    right_adj_3d = adj_3d[right_mask]

    return right_adj_3d

def adj_cnts_3d(cnt1, cnt2, dub):
    dist, inds = misc_utils.nn_kdtree(cnt1, cnt2, dub=dub)
    if sum(dist != np.inf) < 5:
        return cnt2
    adj_cnt1 = cnt1[inds[~np.isinf(dist)]]
    adj_cnt2 = cnt2[~np.isinf(dist)]
    return (adj_cnt1 + adj_cnt2)/2

def find_grasp_point(centroid, boundary_points, fraction=0.5):
    """
    Finds a grasp point between the centroid and the closest boundary point.

    Params:
    - centroid: (3,) array, centroid of the gallbladder surface.
    - boundary_points: (N, 3) array, boundary points attached to the liver.
    - fraction: Fraction of the way from centroid to boundary.

    Returns:
    - grasp_point: (3,) array, the computed grasp point.
    """
    # # Find the closest boundary point
    # distances = np.linalg.norm(boundary_points - centroid, axis=1)
    # closest_boundary = boundary_points[np.argmin(distances)]

    # # Compute grasp point along the line
    # grasp_point = centroid + fraction * (closest_boundary - centroid)

    grasp_point = (centroid + boundary_points.mean(axis=0))*fraction

    return grasp_point