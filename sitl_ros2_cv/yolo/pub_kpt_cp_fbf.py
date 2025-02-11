import rclpy

import os

from nodes.detect import pub_kpt_cp_node

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"          : "pub_kpt_cp_fbf",
        "queue_size"         : 5,
        "slop"               : 0.05,
        "inst_name"          : "FBF",
        "ct_kpt_nm"          : "Center",
        "tf_path"            : "/home/"+os.getlogin()+"/aruco_data/base_tfs.yaml",
        "frame_id"           : "ecm_left",
        "tip_child_frame_id" : "psm2_tip_yolo",
        "jaw_child_frame_id" : "psm2_jaw_yolo",
        "psm_topic"          : "/PSM2/custom/setpoint_cp",
    }

    app = pub_kpt_cp_node.PUB_KPT_CP(params)
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()