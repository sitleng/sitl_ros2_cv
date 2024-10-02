#! /usr/bin/env python3

import rclpy

import os
from nodes import pub_cam_pcl_node

def main(args=None):
    rclpy.init(args=args)

    params = {
        "slop"            : 0.05,
        "cam_type"        : 30,
        "calib_dir"       : "L2R",
        "calib_type"      : "opencv",
        "resolution"      : "HD720",
        "calib_path"      : "/home/" + os.getlogin() + "/ecm_si_calib_data",
        "depth_scale"     : 1000,
        "depth_trunc"     : 0.15,
        "pcl_scale"       : 15,
    }

    app = pub_cam_pcl_node.PUB_CAM_PCL(params)

    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
