#!/usr/bin/env python3

import numpy as np
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from sitl_dvrk.msg import UInt8MultiArrayStamped, UInt16MultiArrayStamped, Dt2KptState

def get_ma_layout(arr):
    ma_layout = MultiArrayLayout()
    ma_layout.data_offset = 0
    ma_layout.dim = [MultiArrayDimension() for i in range(arr.ndim)]
    if arr.ndim == 2:
        ma_layout.dim[0].label  = "height"
        ma_layout.dim[0].size   = arr.shape[0]
        ma_layout.dim[0].stride = 1*arr.shape[1]*arr.shape[0]
        ma_layout.dim[1].label  = "width"
        ma_layout.dim[1].size   = arr.shape[1]
        ma_layout.dim[1].stride = 1*arr.shape[1]
    elif arr.ndim == 3:
        ma_layout.dim[0].label  = "height"
        ma_layout.dim[0].size   = arr.shape[0]
        ma_layout.dim[0].stride = arr.shape[2]*arr.shape[1]*arr.shape[0]
        ma_layout.dim[1].label  = "width"
        ma_layout.dim[1].size   = arr.shape[1]
        ma_layout.dim[1].stride = arr.shape[2]*arr.shape[1]
        ma_layout.dim[2].label  = "channel"
        ma_layout.dim[2].size   = arr.shape[2]
        ma_layout.dim[2].stride = arr.shape[2]
    return ma_layout

def ma2arr(ma_msg):
    n = len(ma_msg.layout.dim)
    arr = np.asarray(ma_msg.data)
    if n == 2:
        return arr.reshape(
            ma_msg.layout.dim[0].size, 
            ma_msg.layout.dim[1].size
        )
    elif n == 3:
        return arr.reshape(
            ma_msg.layout.dim[0].size,
            ma_msg.layout.dim[1].size,
            ma_msg.layout.dim[2].size
        )
    else:
        return arr

def arr2uint8(arr, stamp):
    msg = UInt8MultiArrayStamped()
    msg.header.stamp = stamp
    msg.layout = get_ma_layout(arr)
    msg.data = arr.flatten()
    return msg

def arr2uint16ma(arr, stamp):
    msg = UInt16MultiArrayStamped()
    msg.header.stamp = stamp
    msg.layout = get_ma_layout(arr)
    msg.data = arr.flatten()
    return msg

def gen_dt2kptstate(kpt_nms, kpt_2d, kpt_3d, stamp):
    msg = Dt2KptState()
    msg.header.stamp = stamp
    msg.name = kpt_nms
    msg.kpts2d.layout = get_ma_layout(kpt_2d)
    msg.kpts2d.data   = kpt_2d.flatten()
    msg.kpts3d.layout = get_ma_layout(kpt_3d)
    msg.kpts3d.data   = kpt_3d.flatten()
    return msg