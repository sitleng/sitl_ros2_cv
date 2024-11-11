#! /usr/bin/env python3

# general libraries
import os

# ros2 libraries
import rclpy

# nodes
from nodes.rec import rec_dvrk_kin_node

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"       : "rec_dvrk_kin",
        "queue_size"      : 10,
        "slop"            : 0.01,
        "save_path"       : "/home/" + os.getlogin() + "/dvrk_rec",
    }
    
    topic_names = [
        "/ECM/custom/setpoint_cp",
        "/ECM/custom/local/setpoint_cp",
        "/ECM/measured_js",
        "/MTML/gripper/measured_js",
        "/MTML/local/measured_cp",
        "/MTML/measured_js",
        "/MTML/measured_cp",
        "/MTMR/gripper/measured_js",
        "/MTMR/local/measured_cp",
        "/MTMR/measured_js",
        "/MTMR/measured_cp",
        "/PSM1/custom/setpoint_cp",
        "/PSM1/custom/local/setpoint_cp",
        "/PSM1/jaw/measured_js",
        "/PSM1/measured_js",
        "/PSM2/custom/setpoint_cp",
        "/PSM2/custom/local/setpoint_cp",
        "/PSM2/jaw/measured_js",
        "/PSM2/measured_js"
    ]

    app = rec_dvrk_kin_node.REC_DVRK_KIN(params, topic_names)

    try:
        rclpy.spin(app)
    except KeyboardInterrupt:
        pass
    finally:
        app.close_bag()
        app.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
