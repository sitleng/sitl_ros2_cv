#!/usr/bin/env python3

import traceback
import numpy as np
import math
import copy
import cv2
from scipy.spatial import cKDTree
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)

from utils import misc_utils

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.catalog import Metadata

from detectron2.utils.visualizer import Visualizer, _create_text_labels, GenericMask

def load_seg_metadata():
    seg_metadata = Metadata()
    seg_metadata.set(thing_classes = ['Liver','Gallbladder', 'LiverBed'])
    seg_metadata.set(evaluator_type = 'coco')
    seg_metadata.set(thing_colors=[(255, 255, 0), (255, 0, 255), (0, 255, 255)])
    return seg_metadata

def load_kpt_metadata(inst_nm):
    kpt_metadata = Metadata()
    kpt_metadata.set(thing_classes = [inst_nm])
    kpt_metadata.set(evaluator_type = 'coco')
    kpt_metadata.set(
        keypoint_flip_map = []
    )
    if inst_nm == "PCH":
        kpt_metadata.set(
            keypoint_names = [
                'LeftScrewBottom', 'LeftScrewTop', 'CentralScrew', 'StartHook',
                'CentralHook', 'TipHook', 'RightScrewTop', 'RightScrewBottom'
            ]
        )
        kpt_metadata.set(
            keypoint_connection_rules = [
                ('LeftScrewBottom', 'LeftScrewTop', (0, 255, 0)), ('LeftScrewTop', 'CentralScrew', (0, 255, 0)),
                ('RightScrewBottom', 'RightScrewTop', (0, 255, 0)), ('RightScrewTop', 'CentralScrew', (0, 255, 0)),
                ('CentralScrew', 'StartHook', (0, 255, 0)), ('StartHook','CentralHook',(0, 255, 0)), 
                ('CentralHook','TipHook', (0, 255, 0))
            ]
        )
        kpt_metadata.set(thing_colors=[(238, 130, 238)])
    elif inst_nm == "FBF":
        kpt_metadata.set(
            keypoint_names = ['Head', 'Edge', 'Center', 'TipLeft', 'TipRight']
        )
        kpt_metadata.set(
            keypoint_connection_rules = [
                ('TipLeft', 'Center', (0, 255, 0)), ('TipRight', 'Center', (0, 255, 0)),
                ('Center', 'Edge', (0, 255, 0)), ('Edge','Head', (0, 255, 0))
            ]
        )
        kpt_metadata.set(thing_colors=[(0, 165, 255)])
    else:
        print("Invalid Instrument (Metadata)...")
        return None
    return kpt_metadata

def draw_box(ax, box_coord, color, label, alpha=1, line_style="-"):
    x0, y0, x1, y1 = box_coord
    width = x1 - x0
    height = y1 - y0

    ax.add_patch(
        mpl.patches.Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor=tuple(x / 255.0 for x in color),
            linewidth=3,
            alpha=alpha,
            linestyle=line_style,
            label = label
        )
    )

def draw_circle(ax, circle_coord, color, radius=5):
    ax.add_patch(
        mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color)
    )

def draw_line(ax, x_data, y_data, color, linestyle="-", linewidth=5):
    ax.add_line(
        mpl.lines.Line2D(
            x_data,
            y_data,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
        )
    )

def draw_and_connect_keypoints_plt(metadata, keypoints, bbox, box_c, label, ax):
    visible = {}
    keypoint_names = metadata.get("keypoint_names")
    
    draw_box(ax,bbox,box_c,label)
    
    for idx, keypoint in enumerate(keypoints):
        # draw keypoint
        x, y, prob = keypoint
        if prob > 0.02:
            draw_circle(ax, (x, y), color="r")
            if keypoint_names:
                keypoint_name = keypoint_names[idx]
                visible[keypoint_name] = (x, y)

    if metadata.get("keypoint_connection_rules"):
        for kp0, kp1, color in metadata.keypoint_connection_rules:
            if kp0 in visible and kp1 in visible:
                x0, y0 = visible[kp0]
                x1, y1 = visible[kp1]
                color = tuple(x / 255.0 for x in color)
                draw_line(ax, [x0, x1], [y0, y1], color=color)

def draw_and_connect_keypoints_cv(img, metadata, keypoints, bbox, box_c, prob_thr):
    visible = {}
    out_img = copy.deepcopy(img)
    keypoint_names = metadata.get("keypoint_names")
    bbox = bbox.astype(np.int32)
    
    cv2.rectangle(out_img, tuple(bbox[:2]), tuple(bbox[2:4]), color=box_c, thickness=3)
    
    for idx, keypoint in enumerate(keypoints):
        # draw keypoint
        x, y, prob = keypoint
        if prob > prob_thr:
            ct = (x.astype(np.int32),y.astype(np.int32))
            cv2.circle(out_img, ct, 8, (0,255,255), -1)
            if keypoint_names:
                keypoint_name = keypoint_names[idx]
                visible[keypoint_name] = (x, y)

    if metadata.get("keypoint_connection_rules"):
        for kp0, kp1, color in metadata.keypoint_connection_rules:
            if kp0 in visible and kp1 in visible:
                x0, y0 = visible[kp0]
                x1, y1 = visible[kp1]
                pt1 = (x0.astype(np.int32),y0.astype(np.int32))
                pt2 = (x1.astype(np.int32),y1.astype(np.int32))
                cv2.line(out_img, pt1, pt2, (0,255,0), 2)
    return out_img

def load_seg_predictor(model_path, score_thr):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thr
    return DefaultPredictor(cfg)

def set_seg_pred_score_thr(predictor, score_thr):
    predictor.cfg["MODEL"]["ROI_HEADS"]["SCORE_THRESH_TEST"] = score_thr
    return predictor

def load_kpt_predictor(model_path, score_thresh, inst_nm):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    if inst_nm == "PCH":
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 8
    elif inst_nm == "FBF":
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
    else:
        print("Invalid Instrument (Predictor)...")
        return None
    # 68%: 0.81, 95%: 0.61, 99%: 0.32
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.61]*cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    return DefaultPredictor(cfg)

def rm_dup_legend():
    # Remove duplicate legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),loc="upper right",fontsize="xx-large")

def gen_mask_img(img, obj1_idx, obj2_idx, masks):
    mask_img = np.zeros_like(img)
    c = None
    for i, mask in enumerate(masks):
        if i == obj1_idx:
            c = np.array([0, 0, 255])
        elif i == obj2_idx:
            c = np.array([0, 255, 0])
        else:
            c = None
        if c is not None:
            mask_img[mask==1, :] = c
    return mask_img

def gen_seg_img(img, mask_img, alpha):
    seg_img = copy.deepcopy(img)
    return cv2.addWeighted(seg_img, 1, mask_img, alpha, 0)

def draw_full_seg(img, dt2_seg_labels, obj1_idx, obj2_idx, cnts, save_fn=None):
    c = None
    for i, cnt in enumerate(cnts):
        if i == obj1_idx:
            c = 'b'
        elif i == obj2_idx:
            c = 'g'
        if c is not None:
            for poly in cnt:
                plt.scatter(
                    poly[:,0],poly[:,1],
                    color=c,linewidth=2,
                    label=dt2_seg_labels[i].split()[0]
                )
            c = None
    plt.gca().set_axis_off()
    rm_dup_legend()
    plt.imshow(img.copy()[:,:,::-1])
    if save_fn is not None:
        plt.savefig(save_fn, bbox_inches='tight', pad_inches = 0)
        plt.clf()

def draw_adj_traj(img, adj_obj1_obj2, obj1_label, obj2_label, save_fn=None):
    plt.scatter(
        adj_obj1_obj2[:,0],adj_obj1_obj2[:,1],
        color='b',linewidth=3,
        label=obj1_label.split()[0]
    )
    plt.scatter(
        adj_obj1_obj2[:,2],adj_obj1_obj2[:,3],
        color='g',linewidth=3,
        label=obj2_label.split()[0]
    )
    plt.gca().set_axis_off()
    rm_dup_legend()
    plt.imshow(img.copy()[:,:,::-1])
    if save_fn is not None:
        plt.savefig(save_fn, bbox_inches='tight', pad_inches = 0)

def draw_bnd_pts(img,goal_bnd_2d,save_fn=None):
    plt.scatter(
        goal_bnd_2d[:,0],goal_bnd_2d[:,1],
        color='r',
        linewidth=3,
        label="boundary points"
    )
    plt.gca().set_axis_off()
    plt.legend(loc="upper right",fontsize="xx-large")
    plt.imshow(img.copy()[:,:,::-1])
    if save_fn is not None:
        plt.savefig(save_fn, bbox_inches='tight', pad_inches = 0)
        plt.clf()

def draw_key_pts(dt2_kpt_boxes, dt2_kpt_classes, dt2_kpt_labels, dt2_kpts, keypt_metadata, img, save_fn=None):
    img_copy = copy.deepcopy(img)
    fig, ax = plt.subplots()

    for i, dt2_kpt in enumerate(dt2_kpts):
        label = dt2_kpt_labels[i].split()[0]
        box_c = keypt_metadata.get("thing_colors")[dt2_kpt_classes[i]]
        draw_and_connect_keypoints_plt(keypt_metadata, dt2_kpt, dt2_kpt_boxes[i], box_c,label,ax)

    plt.gca().set_axis_off()
    rm_dup_legend()
    plt.imshow(img_copy[:,:,::-1])
    if save_fn is not None:
        plt.savefig(save_fn, bbox_inches='tight', pad_inches = 0)
        plt.clf()

def extract_inst_preds(insts, metadata):
    boxes = np.asarray(insts.pred_boxes.tensor, dtype=np.int32) if insts.has("pred_boxes") else None
    scores = insts.scores if insts.has("scores") else None
    classes = insts.pred_classes.tolist() if insts.has("pred_classes") else None
    labels = _create_text_labels(classes, scores, metadata.get("thing_classes", None))
    keypoints = insts.pred_keypoints if insts.has("pred_keypoints") else None
    if insts.has("pred_masks"):
        masks = np.asarray(insts.pred_masks)
        masks = [GenericMask(x, insts.image_size[0], insts.image_size[1]) for x in masks]
    else:
        masks = None
    return boxes, scores, classes, labels, keypoints, masks

def run_dt2_seg(img, seg_metadata, seg_predictor):
    outputs = seg_predictor(img)
    pred_boxes, _, _, pred_labels, _, pred_masks = extract_inst_preds(
        outputs["instances"].to("cpu"), seg_metadata
    )
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
    
def run_dt2_kpt(img, keypt_predictor, keypt_metadata):
    outputs = keypt_predictor(img)
    dt2_kpt_boxes, _, dt2_kpt_classes, dt2_kpt_labels, dt2_kpts, _ = extract_inst_preds(
        outputs["instances"].to("cpu"), keypt_metadata
    )
    return dt2_kpt_boxes, dt2_kpt_classes, dt2_kpt_labels, np.asarray(dt2_kpts)

def get_inst_kpts_2d(inst_nm, img, kpt_predictor, kpt_metadata, score_thr):
    _, _, dt2_kpt_labels, dt2_kpts = run_dt2_kpt(img, kpt_predictor, kpt_metadata)
    if len(dt2_kpt_labels) == 0:
        return None, None
    inst_idx = None
    for i, label in enumerate(dt2_kpt_labels):
        if inst_nm in label:
            inst_idx = i
            break
    if inst_idx is None:
        return None, None
    kpt_nms = kpt_metadata.get("keypoint_names")
    inst_kpts_nms = []
    inst_kpts_2d = []
    try:
        for i, kpt in enumerate(dt2_kpts[inst_idx]):
            x, y, score = kpt
            if score > score_thr:
                inst_kpts_nms.append(kpt_nms[i])
                inst_kpts_2d.append([x, y])
    except:
        print(dt2_kpts[inst_idx])
        print(kpt_nms)
    return inst_kpts_nms, np.array(inst_kpts_2d, dtype=np.uint16)

def validate_hook_kpts(pch_kpt_nms, pch_kpts_3d):
    ch = None
    cs = None
    th = None
    for i, pch_kpt in enumerate(pch_kpts_3d):
        if pch_kpt_nms[i] == "CentralHook":
            ch = pch_kpt
        elif pch_kpt_nms[i] == "CentralScrew":
            cs = pch_kpt
        elif pch_kpt_nms[i] == "TipHook":
            th = pch_kpt
    if ch is None or cs is None or th is None:
        return False
    vec_cs_ch = ch - cs
    vec_cs_ch_len = np.linalg.norm(vec_cs_ch)
    vec_cs_th = th - cs
    vec_cs_th_len = np.linalg.norm(vec_cs_th)
    # angle = math.degrees(misc_utils.angle_btw_vecs(vec_cs_ch, vec_cs_th))
    # ratio = vec_cs_th_len/vec_cs_ch_len
    if np.isclose(vec_cs_ch_len, 0) or np.isclose(vec_cs_th_len, 0):
        return False
    # if vec_cs_ch_len > 0.016 or vec_cs_ch_len < 0.013 or vec_cs_th_len > 0.018 or vec_cs_th_len < 0.015:
    #     print("Rejected cs_ch: {}, cs_th: {}".format(vec_cs_ch_len, vec_cs_th_len))
    #     return False
    # if vec_cs_th_len > 0.02 or vec_cs_th_len < 0.015:
    if vec_cs_th_len > 0.025 or vec_cs_th_len < 0.01:
        print("Rejected cs_th: {}".format(vec_cs_th_len))
        return False
    # if abs(angle) > 15 or abs(angle) < 12 or ratio > 1.3 or ratio < 1.1:
    #     print("Rejected Angle: {}, Ratio: {}".format(angle, ratio))
    #     return False
    return True

def validate_frcp_kpts(fbf_kpt_nms, fbf_kpts_3d):
    fbfc = None
    fbfe = None
    fbfh = None
    for i, pch_kpt in enumerate(fbf_kpts_3d):
        if fbf_kpt_nms[i] == "Center":
            fbfc = pch_kpt
        elif fbf_kpt_nms[i] == "Edge":
            fbfe = pch_kpt
        elif fbf_kpt_nms[i] == "Head":
            fbfh = pch_kpt
    if fbfc is None or fbfe is None or fbfh is None:
        return False
    vec_ce = fbfe - fbfc
    vec_ce_len = np.linalg.norm(vec_ce)
    vec_ch = fbfh - fbfc
    vec_ch_len = np.linalg.norm(vec_ch)
    angle = math.degrees(misc_utils.angle_btw_vecs(vec_ce, vec_ch))
    ratio = vec_ch_len/vec_ce_len
    if np.isclose(vec_ce_len, 0) or np.isclose(vec_ch_len, 0):
        print("Rejected ce: {}, ch: {}".format(vec_ce_len, vec_ch_len))
        return False
    if vec_ce_len > 0.008 or vec_ce_len < 0.004 or vec_ch_len > 0.009 or vec_ch_len < 0.005:
        print("Rejected ce: {}, ch: {}".format(vec_ce_len, vec_ch_len))
        return False
    if abs(angle) < 13 or ratio > 1.3 or ratio < 1:
        print("Rejected Angle: {}, Ratio: {}".format(angle, ratio))
        return False
    return True

def get_cur_traj_pt(prev_traj_pt, bnd_2d, dt2_kpts, kpt_nms):
    cur_traj_pt = prev_traj_pt
    vec_st_et = bnd_2d[-1] - bnd_2d[0]
    # Old Keypoint PCH Model
    # vec_tc_tl = dt2_kpts[kpt_nms.index("TipLeft")][:2] - dt2_kpts[kpt_nms.index("TipCenter")][:2]
    # vec_tc_tr = dt2_kpts[kpt_nms.index("TipRight")][:2] - dt2_kpts[kpt_nms.index("TipCenter")][:2]
    # angle = math.degrees(misc_utils.angle_btw_vecs(vec_tc_tl, vec_tc_tr))
    # ratio = np.linalg.norm(vec_tc_tr)/np.linalg.norm(vec_tc_tl)
    # if abs(angle) < 15 and ratio > 2.3:
    #     tr_pt = dt2_kpts[0][kpt_nms.index("TipRight")][:2]
    #     traj_tree = KDTree(bnd_2d)
    #     des_dist = 100
    #     matching_indices = traj_tree.query_ball_point(tr_pt, des_dist)
    #     if matching_indices:
    #         max_dist = 0
    #         for idx in matching_indices:
    #             dist = np.linalg.norm(bnd_2d[idx] - tr_pt)
    #             if dist > max_dist:
    #                 vec_tr_ct = bnd_2d[idx] - tr_pt
    #                 angle2 = math.degrees(misc_utils.angle_btw_vecs(vec_st_et, vec_tr_ct))
    #                 if abs(angle2) < 15:
    #                     max_dist = dist
    #                     cur_traj_pt = bnd_2d[idx]

    # New Keypoint PCH Model
    th_pt = dt2_kpts[0][kpt_nms.index("TipHook")][:2]
    vec_cs_ch = dt2_kpts[0][kpt_nms.index("CentralHook")][:2] - dt2_kpts[0][kpt_nms.index("CentralScrew")][:2]
    vec_cs_th = th_pt - dt2_kpts[0][kpt_nms.index("CentralScrew")][:2]
    if not np.isclose(np.linalg.norm(vec_cs_ch), 0) and not np.isclose(np.linalg.norm(vec_cs_th), 0):
        angle = math.degrees(misc_utils.angle_btw_vecs(vec_cs_ch, vec_cs_th))
        ratio = np.linalg.norm(vec_cs_th)/np.linalg.norm(vec_cs_ch)
        if abs(angle) < 30 and 1 < ratio < 2 and np.linalg.norm(prev_traj_pt-th_pt) < 10:
            traj_tree = cKDTree(bnd_2d)
            des_dist = 100
            matching_indices = traj_tree.query_ball_point(th_pt, des_dist)
            if matching_indices:
                max_dist = 0
                for idx in matching_indices:
                    dist = np.linalg.norm(bnd_2d[idx] - th_pt)
                    if dist > max_dist:
                        vec_th_ct = bnd_2d[idx] - th_pt
                        angle2 = math.degrees(misc_utils.angle_btw_vecs(vec_st_et, vec_th_ct))
                        if abs(angle2) < 15:
                            max_dist = dist
                            cur_traj_pt = bnd_2d[idx]
    return cur_traj_pt

def max_len_traj(trajs):
    max_len = 0
    max_traj = None
    for traj in trajs:
        cur_len = cv2.arcLength(traj, False)
        if cur_len > max_len:
            max_len = cur_len
            max_traj = traj
    return max_traj

# def get_adj_traj(traj_cnds, target):
#     min_dist = np.inf
#     final_traj = None
#     for traj_cnd in traj_cnds:
#         dist0 = np.linalg.norm(traj_cnd[0]-target)
#         dist1 = np.linalg.norm(traj_cnd[-1]-target)
#         if min(dist0, dist1) < min_dist:
#             min_dist = min(dist0, dist1)
#             if dist0 < dist1:
#                 final_traj = traj_cnd
#             else:
#                 final_traj = np.flip(traj_cnd, axis=0)
#     return final_traj