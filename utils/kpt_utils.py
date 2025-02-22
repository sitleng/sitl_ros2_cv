import numpy as np

from sitl_ros2_interfaces.msg import Dt2KptState
from utils import pcl_utils, ma_utils, tf_utils

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

def pch_g_pcmjaw(kpt_nms, kpts_3d, g_psmtip, g_psmjaw, g_psmtip_psmjaw):
    if "TipHook" in kpt_nms:
        # Tip position based on the point cloud
        g_psmjaw[:3,3] = kpts_3d[kpt_nms.index("TipHook")]
        g_offset = tf_utils.ginv(g_psmtip).dot(g_psmjaw)
        t_dist = np.linalg.norm(g_offset[:3,3] - g_psmtip_psmjaw[:3,3])
        if t_dist > 0.01:
            g_psmjaw = g_psmtip.dot(g_psmtip_psmjaw)
        else:
            g_psmjaw = g_psmtip.dot(g_offset)
    else:
        g_psmjaw = g_psmtip.dot(g_psmtip_psmjaw)
    return g_psmjaw