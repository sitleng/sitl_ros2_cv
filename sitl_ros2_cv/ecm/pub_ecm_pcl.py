#! /usr/bin/env python3

import rclpy

from nodes.camera import pub_cam_pcl_node

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"     : "pub_ecm_pcl",
        "queue_size"    : 5,
        "slop"          : 0.03,
        "ref_cam_topic" : "/ecm/left/rect/image_color",
    }

    app = pub_cam_pcl_node.PUB_CAM_PCL(params)

    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
