import rclpy

import os

from nodes.yolo import pub_kpt_raw

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"       : "pub_kpt_raw_fbf",
        "queue_size"      : 5,
        "slop"            : 0.05,
        # "refimg_topic"    : "/video/left/rect/image_color",
        # "pclimg_topic"    : "/video/recon3d/pclimg",
        "refimg_topic"    : "/ecm/left/rect/image_color",
        "pclimg_topic"    : "/ecm/recon3d/pclimg",
        "inst_name"       : "FBF",
        "ct_kpt_nm"       : "Center",
        "model_path"      : "/home/"+os.getlogin()+"/yolo/model_fbf/fbf.pt",
        "window_size"     : 15,
        "mad_thr"         : 1.5,
    }

    app = pub_kpt_raw.PUB_KPT_RAW(params)
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()