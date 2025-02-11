import rclpy

import os
from nodes.detect import pub_seg_lv_gb_node

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"        : "pub_seg_lv_gb",
        "queue_size"       : 5,
        "slop"             : 0.05,
        # "refimg_topic"    : "/video/left/rect/image_color",
        # "pclimg_topic"    : "/video/recon3d/pclimg",
        "refimg_topic"    : "/ecm/left/rect/image_color",
        "pclimg_topic"    : "/ecm/recon3d/pclimg",
        # "model_type"      : "MaskDINO",
        # "model_weights"   : "/home/"+os.getlogin()+"/MaskDINO/output_crcd/model_final.pth",
        # "config_file"     : "/home/"+os.getlogin()+"/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_custom.yaml",
        "model_type"       : "Detectron2",
        "model_weights"    : "/home/"+os.getlogin()+"/dt2_dataset/liver/crcd_res/model_final.pth",
        "pred_score_thr"   : 0.7,
        "cnt_area_thr"     : 10000,
        "dist_upp_bnd_2d"  : 15,
        "window_size"      : 25,
        "mad_thr"          : 1.5,
        "num_cnt_ratio"    : 0.05,
        "num_skel_ratio"   : 0.2,
        "skel_ang_thr"     : 30,
        "adj_cnt_seg_thr"  : 0.01,
    }

    app = pub_seg_lv_gb_node.PUB_SEG_LV_GB(params)
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()