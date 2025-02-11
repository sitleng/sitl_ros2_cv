import rclpy

import os
from nodes.yolo import pub_seg_lv_gb_node

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"        : "pub_seg_lv_gb_yolo",
        "queue_size"       : 5,
        "slop"             : 0.05,
        # "refimg_topic"    : "/video/left/rect/image_color",
        # "pclimg_topic"    : "/video/recon3d/pclimg",
        "refimg_topic"     : "/ecm/left/rect/image_color",
        "pclimg_topic"     : "/ecm/recon3d/pclimg",
        "model_path"       : "/home/"+os.getlogin()+"/yolo/liver_seg/v11l_crcd_temp.pt",
        "conf_thr"         : 0.5,
        "cnt_area_thr"     : 10000,
        "dist_upp_bnd_2d"  : 15,
        "window_size"      : 25,
        "mad_thr"          : 1.5,
        "num_cnt_ratio"    : 0.05,
        "num_skel_ratio"   : 0.2,
        "skel_ang_thr"     : 25,
        "adj_cnt_seg_thr"  : 0.005,
    }

    app = pub_seg_lv_gb_node.PUB_SEG_LV_GB_YOLO(params)
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()