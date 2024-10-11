#!/usr/bin/env python3

import numpy as np
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension

def arr2data(arr):
    return arr.flatten().tolist()
    
def np_arr2ma_msg(arr, msg):
    msg.layout = get_ma_layout(arr)
    msg.data = arr2data(arr)
    return msg

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

# def arr2uint8(arr, stamp):
#     msg = UInt8MultiArrayStamped()
#     msg.header.stamp = stamp
#     msg.layout = get_ma_layout(arr)
#     msg.data = arr.flatten()
#     return msg

# def arr2uint16ma(arr, stamp):
#     msg = UInt16MultiArrayStamped()
#     msg.header.stamp = stamp
#     msg.layout = get_ma_layout(arr)
#     msg.data = arr.flatten()
#     return msg
