import numpy as np

from detectron2.data.catalog import Metadata

from sitl_ros2_interfaces.msg import Dt2KptState
from utils import pcl_utils, ma_utils, tf_utils

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

def win_avg_3d_kpts(cur_kpts_2d, cur_kpt_nms, pcl_array, window_size, mad_thr):
    res_kpts = None
    res_kpt_nms = []
    for i, cur_kpt_2d in enumerate(cur_kpts_2d):
        cur_kpt_3d = pcl_utils.win_avg_3d_pt(cur_kpt_2d, pcl_array, window_size, mad_thr)
        if cur_kpt_3d is None:
            continue
        else:
            res_kpt_nms.append(cur_kpt_nms[i])
        if res_kpts is None:
            res_kpts = cur_kpt_3d[np.newaxis, :]
        else:
            res_kpts = np.vstack((res_kpts, cur_kpt_3d))
    return res_kpt_nms, res_kpts

def gen_dt2kptstate(kpt_nms, kpt_2d, kpt_3d, stamp):
    msg = Dt2KptState()
    msg.header.stamp = stamp
    msg.name = kpt_nms
    msg.kpts2d.layout = ma_utils.get_ma_layout(kpt_2d)
    msg.kpts2d.data   = ma_utils.arr2data(kpt_2d)
    msg.kpts3d.layout = ma_utils.get_ma_layout(kpt_3d)
    msg.kpts3d.data   = ma_utils.arr2data(kpt_3d)
    return msg

def pch_g_pcmjaw(kpt_nms, kpts_3d, g_psmtip, g_psmjaw, g_psmtip_psmjaw, offset, count):
    if "TipHook" in kpt_nms:
        # Tip position based on the point cloud
        g_psmjaw[:3,3] = kpts_3d[kpt_nms.index("TipHook")]
        temp = np.linalg.norm(g_psmtip[:3,3] - g_psmjaw[:3,3])
        if temp < 0.01 or temp > 0.02:
            if offset is None:
                g_psmjaw = g_psmtip.dot(g_psmtip_psmjaw)
            else:
                g_psmjaw[:3,3] = g_psmtip[:3,3] + offset/count
        else:
            g_offset = tf_utils.ginv(g_psmtip).dot(g_psmjaw)
            if offset is None:
                offset = g_offset[:3,3]
            else:
                offset += g_offset[:3,3]
            count += 1
    else:
        if offset is None:
            g_psmjaw = g_psmtip.dot(g_psmtip_psmjaw)
        else:
            g_psmjaw[:3,3] = g_psmtip[:3,3] + offset/count
    return g_psmjaw, offset, count