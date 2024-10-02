#! /usr/bin/env python3

import rclpy


import os
from nodes import pub_cam_disp_node

def main(args=None):
    rclpy.init(args=args)

    params = {
        "slop"            : 0.05,
        "cam_type"        : 30,
        "calib_dir"       : "L2R",
        "calib_type"      : "opencv",
        "resolution"      : "HD720",
        "calib_path"      : "/home/" + os.getlogin() + "/ecm_si_calib_data",
        "frame_id"        : "ecm_opencv",
        "bf_size"         : 5,
        "depth_scale"     : 1000,
        "min_disp"        : 0,
        "sgm_ndisp"       : 256,
        "P1"              : 30,
        "P2"              : 210,
        "uniq_ratio"      : 5,
        "wls_filter_flag" : False,
        "wls_lambda"      : 0.4,
        "wls_sigma"       : 8000,
        "dbf_flag"        : False,
        "dbf_ndisp"       : 256,
        "radius"          : 0,
        "iters"           : 1,
    }

    app = pub_cam_disp_node.PUB_CAM_DISP(params)

    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
