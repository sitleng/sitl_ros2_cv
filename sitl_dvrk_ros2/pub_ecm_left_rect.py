#!/usr/bin/env python3

import rclpy

import os

from nodes import pub_cam_rect_node

def main(args=None):
    rclpy.init(args=args)

    params = {
        "gpu_flag"  : False,
        "cam_type"  : 30,
        "calib_dir" : "L2R",
        "calib_type": "opencv",
        "resolution": "HD720",
        "calib_path": "/home/" + os.getlogin() + "/ecm_si_calib_data",
        "cam_side"  : "left_rect",
        "fps"       : 60,
        "slop"      : 0.05
    }

    app = pub_cam_rect_node.PUB_CAM_RECT(params)

    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
