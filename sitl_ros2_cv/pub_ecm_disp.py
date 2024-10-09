#! /usr/bin/env python3

import rclpy

from nodes import pub_cam_disp_node
from utils import ecm_utils

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"       : "pub_ecm_disp",
        "queue_size"      : 10,
        "slop"            : 0.03,
        "cam1_topic"      : "/ecm/left/rect/image_mono",
        "cam2_topic"      : "/ecm/right/rect/image_mono",
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
    params.update(ecm_utils.load_base_params())

    app = pub_cam_disp_node.PUB_CAM_DISP(params)

    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
