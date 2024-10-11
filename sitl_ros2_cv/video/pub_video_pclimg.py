#! /usr/bin/env python3

import rclpy

from nodes.camera import pub_cam_pclimg_node
from utils import ecm_utils

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"   : "pub_video_pclimg",
        "queue_size"  : 5,
        "depth_scale" : 1000,
        "depth_trunc" : 0.2,
        "pcl_scale"   : 16,
    }
    params.update(ecm_utils.load_base_params())

    app = pub_cam_pclimg_node.PUB_CAM_PCLIMG(params)

    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
