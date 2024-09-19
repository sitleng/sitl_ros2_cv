#!/usr/bin/env python3

import sys
sys.path.append('/home/leo/MaskDINO/')

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Instances

from utils import dt2_utils, misc_utils

def load_seg_predictor(config_file, model_weights):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.freeze()
    return DefaultPredictor(cfg)

def preprocess_insts(insts, score_thr):
    insts_filt = Instances(insts.image_size)
    scores = insts.scores if insts.has("scores") else None
    keys = list(insts.get_fields().keys())
    for key in keys:
        insts_filt.set(key, insts.get(key)[scores > score_thr])
    return insts_filt

def mask_dino_seg(img, seg_predictor, seg_metadata, score_thr):
    preds = seg_predictor(img)
    insts = preprocess_insts(preds["instances"].to("cpu"), score_thr)
    pred_boxes, _, _, pred_labels, _, pred_masks = dt2_utils.extract_inst_preds(insts, seg_metadata)
    seg_labels = []
    seg_probs  = []
    seg_masks  = []
    seg_boxes  = []
    for i, mask in enumerate(pred_masks):
        cur_label, cur_pred = pred_labels[i].split(" ")
        if cur_label not in seg_labels:
            seg_labels.append(cur_label)
            seg_probs.append(cur_pred)
            seg_masks.append(mask.mask)
            seg_boxes.append(pred_boxes[i])
        else:
            prev_idx = seg_labels.index(cur_label)
            if cur_pred > seg_probs[prev_idx]:
                seg_probs[prev_idx] = cur_pred
                seg_masks[prev_idx] = mask.mask
                seg_boxes[prev_idx] = pred_boxes[i]
    return seg_probs, seg_labels, seg_boxes, seg_masks

