import cv2
import numpy as np

from detectron2.data.catalog import Metadata

from utils import misc_utils, cv_utils

def load_seg_metadata():
    seg_metadata = Metadata()
    seg_metadata.set(thing_classes = ['Meat','Skin','Liver','Gallbladder','FBF','PCH'])
    seg_metadata.set(evaluator_type = 'coco')
    seg_metadata.set(thing_colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (100, 50, 0), (10, 150, 75), (48, 92, 38)])
    return seg_metadata

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

def gen_gallb_mask(gallb_score, gallb_mask, liver_score, liver_mask, cnt_area_thr, dub):
    gallb_cnts = get_obj_seg(gallb_mask, cnt_area_thr)
    liver_cnts = get_obj_seg(liver_mask, cnt_area_thr)
    if gallb_cnts is None or liver_cnts is None:
        return None
    dist, inds = misc_utils.nn_kdtree(liver_cnts, gallb_cnts, dub=dub)
    adj_liver = liver_cnts[inds[~np.isinf(dist)]]
    adj_gallb = gallb_cnts[~np.isinf(dist)]
    adj_cnt = np.int32((adj_liver*liver_score + adj_gallb*gallb_score)/(liver_score + gallb_score))
    if adj_cnt.size == 0:
        return None
    adj_ctrd = cv_utils.cnt_centroid(adj_cnt)
    adj_cnt_angles = np.array([np.arctan2(x[1], x[0]) for x in adj_cnt - adj_ctrd])
    adj_cnt = adj_cnt[np.argsort(adj_cnt_angles)]
    adj_cnt = misc_utils.interp_2d(adj_cnt)
    new_gallb_mask = np.zeros_like(gallb_mask)
    cv2.drawContours(new_gallb_mask, [adj_cnt], -1, 255, thickness = cv2.FILLED)
    return new_gallb_mask