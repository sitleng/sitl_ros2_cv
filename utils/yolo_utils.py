from ultralytics import YOLO
import numpy as np
import cv2

from utils import seg_utils, cv_utils

def load_model(model_path):
    return YOLO(model_path)

def load_kpt_metadata(inst_nm):
    if inst_nm == "PCH":
        return {
            "instrument": inst_nm,
            "kpt_nms": [
                'LeftScrewBottom', 'LeftScrewTop', 'CentralScrew', 'StartHook',
                'CentralHook', 'TipHook', 'RightScrewTop', 'RightScrewBottom'
            ],
            "kpt_rules": [
                ('LeftScrewBottom', 'LeftScrewTop', (0, 255, 0)), ('LeftScrewTop', 'CentralScrew', (0, 255, 0)),
                ('RightScrewBottom', 'RightScrewTop', (0, 255, 0)), ('RightScrewTop', 'CentralScrew', (0, 255, 0)),
                ('CentralScrew', 'StartHook', (0, 255, 0)), ('StartHook','CentralHook',(0, 255, 0)), 
                ('CentralHook','TipHook', (0, 255, 0))
            ],
            "color": [(238, 130, 238)]
        }
    elif inst_nm == "FBF":
        return {
            "instrument": inst_nm,
            "kpt_nms": ['Head', 'Edge', 'Center', 'TipLeft', 'TipRight'],
            "kpt_rules": [
                ('TipLeft', 'Center', (0, 255, 0)), ('TipRight', 'Center', (0, 255, 0)),
                ('Center', 'Edge', (0, 255, 0)), ('Edge','Head', (0, 255, 0))
            ],
            "color": [(238, 130, 238)]
        }

def get_segs_2d(img, seg_model, conf_thr=0.25, iou_thr=0.7):
    result = seg_model.predict(
        img,
        retina_masks=True,
        verbose=False,
        conf=conf_thr,
        iou=iou_thr
    )[0]
    seg_cls = list(result.boxes.cls.cpu().numpy())
    seg_nms = result.names
    seg_labels = [seg_nms[seg_cl] for seg_cl in seg_cls]
    seg_scores = list(result.boxes.conf.cpu().numpy())
    seg_cnts = result.masks.xy
    return seg_scores, seg_labels, seg_cnts
    
def get_kpts_2d(img, kpt_model):
    result = kpt_model(img, verbose=False)[0]
    if len(result) == 0:
        return None
    inst_kpts_2d = []
    for kpt in result.keypoints.xy.cpu().numpy()[0]:
        x, y = kpt
        inst_kpts_2d.append([x, y])
    return np.array(inst_kpts_2d, dtype=np.uint16)

def process_gallb_cnt(gallb_cnts, img_shape, cnt_area_thr):
    gallb_mask = seg_utils.get_cnt_mask(gallb_cnts, img_shape)
    gallb_mask = cv_utils.open_mask(gallb_mask, cv2.MORPH_ELLIPSE, 9, 5)
    # gallb_mask = cv_utils.close_mask(gallb_mask, cv2.MORPH_ELLIPSE, 9, 5)
    gallb_cnts = seg_utils.get_obj_seg(gallb_mask, cnt_area_thr)
    return gallb_cnts