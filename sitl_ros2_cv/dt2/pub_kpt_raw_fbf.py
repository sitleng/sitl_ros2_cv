import rclpy

import os

from nodes.dt2 import pub_kpt_raw

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
        "model_path"      : "/home/"+os.getlogin()+"/dt2_dataset/kpts/fbf_results/model_final.pth",
        "model_score_thr" : 0.5,
        "window_size"     : 25,
        "mad_thr"         : 1.5,
        "kpt_score_thr"   : 0.1,
    }

    app = pub_kpt_raw.PUB_KPT_RAW(params)
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()