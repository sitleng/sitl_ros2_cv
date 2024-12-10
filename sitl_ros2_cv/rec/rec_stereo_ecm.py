#! /usr/bin/env python3

import rclpy

import os
from nodes.rec import rec_stereo_cam_node

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"       : "rec_stereo_ecm",
        "queue_size"      : 5,
        "slop"            : 0.03,
        "cam1_topic"      : "/ecm/left/rect/image_color",
        "cam2_topic"      : "/ecm/right/rect/image_color",
        "resolution"      : "HD720",
        "save_path"       : "/home/" + os.getlogin() + "/dvrk_rec/trial_8",
        "fps"             : 60,
        "fourcc"          : "avc1",
    }

    app = rec_stereo_cam_node.REC_STEREO_CAM(params)

    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
