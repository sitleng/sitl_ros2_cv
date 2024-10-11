#!/usr/bin/env python3

import rclpy

from nodes.camera import pub_cam_raw_node
from utils import ecm_utils

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"  : "ecm_left_raw",
        "queue_size" : 10,
        "cam_id"     : 0,
        "gamma"      : 1.5,
        "fps"        : 60,
        "brightness" : -11,
        "contrast"   : 148,
        "saturation" : 180,
        "fps"        : 60
    }
    params.update(ecm_utils.load_base_params())

    app = pub_cam_raw_node.PUB_CAM_RAW(params)

    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
