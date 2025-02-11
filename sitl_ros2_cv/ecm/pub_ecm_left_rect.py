#!/usr/bin/env python3

import rclpy

from nodes.camera import pub_cam_rect_node
from utils import ecm_utils

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"  : "ecm_left_rect",
        "queue_size" : 5,
        "cam_side"   : "left",
        "fps"        : 60,
        "slop"       : 0.01
    }
    params.update(ecm_utils.load_base_params())

    app = pub_cam_rect_node.PUB_CAM_RECT(params)

    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
