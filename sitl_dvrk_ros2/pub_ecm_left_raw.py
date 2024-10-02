#!/usr/bin/env python3

import rclpy

from nodes import pub_cam_raw_node

def main(args=None):
    rclpy.init(args=args)

    params = {
        "gpu_flag"  : False,
        "resolution": "HD720",
        "cam_id"    : 0,
        "cam_side"  : "left",
        "gamma"     : 1.5,
        "fps"       : 60,
        "brightness": -11,
        "contrast"  : 148,
        "saturation": 180,
        "fps"       : 60
    }

    app = pub_cam_raw_node.PUB_CAM_RAW(params)

    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
