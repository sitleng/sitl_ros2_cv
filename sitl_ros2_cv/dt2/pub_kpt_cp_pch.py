import rclpy

import os

from nodes.detect import pub_kpt_cp_node

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"          : "pub_kpt_cp_pch",
        "queue_size"         : 5,
        "slop"               : 0.05,
        "inst_name"          : "PCH",
        "ct_kpt_nm"          : "CentralScrew",
        "tf_path"            : "/home/"+os.getlogin()+"/aruco_data/base_tfs.yaml",
        "frame_id"           : "ecm_left",
        "tip_child_frame_id" : "psm1_tip_dt2",
        "jaw_child_frame_id" : "psm1_jaw_dt2",
        "psm_topic"          : "/PSM1/custom/setpoint_cp",
    }

    app = pub_kpt_cp_node.PUB_KPT_CP(params)
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    