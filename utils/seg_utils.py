import cv2
import numpy as np

from utils import misc_utils, cv_utils

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
    cv2.drawContours(mask, [seg_cnt], -1, 255, thickness = cv2.FILLED)
    return mask
    
def align_cnt(cnt, clockwise=True):
    ctrd = cv_utils.cnt_centroid(cnt)
    # Flip y-coordinates to correct for image coordinate system
    cnt_angles = np.arctan2(-(cnt[:, 1] - ctrd[1]), cnt[:, 0] - ctrd[0])
    sorted_indices = np.argsort(cnt_angles)
    if clockwise:
        sorted_indices = sorted_indices[::-1]  # Reverse for clockwise order
    return cnt[sorted_indices]

def gen_gallb_mask(gallb_score, gallb_cnts, liver_score, liver_cnts, img_shape, dub):
    if gallb_cnts is None or liver_cnts is None:
        return None
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
    return adj_cnt, new_gallb_cnts, new_gallb_mask

def segment_centroids(segments):
    """Compute centroids of each segment."""
    return np.array([np.mean(seg, axis=0) for seg in segments])

def right_bottom_segment(segments):
    """Selects the segment with the centroid closest to the positive x-axis."""
    centroids = segment_centroids(segments)
    right_bottom_idx = np.argmax(centroids[:, 0] + centroids[:, 1])  # Select segment with max x-coordinate
    return segments[right_bottom_idx]