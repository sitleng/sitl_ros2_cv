import rclpy

import os
from nodes.detect import pub_seg_lv_gb_node

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"        : "pub_seg_lv_gb",
        "queue_size"       : 5,
        "slop"             : 0.05,
        "pclimg_topic"     : "/video/recon3d/pclimg",
        "refimg_topic"     : "/video/left/rect/image_color",
        # "model_type"      : "MaskDINO",
        # "model_weights"   : "/home/"+os.getlogin()+"/MaskDINO/output_v1/model_final.pth",
        # "config_file"     : "/home/"+os.getlogin()+"/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_custom.yaml",
        "model_type"       : "Detectron2",
        "model_weights"    : "/home/"+os.getlogin()+"/dt2_dataset/liver/results_v5_re/model_final.pth",
        "pred_score_thr"   : 0.1,
        "cnt_area_thr"     : 10000,
        "dist_upp_bnd_2d"  : 20,
        "window_size"      : 15,
        "mad_thr"          : 3,
        "bnd_angle_thr"    : 20,
        "dist_upp_bnd_3d"  : 0.1,
        "num_cnt_ratio"    : 0.05,
        "num_skel_ratio"   : 0.1,
        "num_interp_ratio" : 0.1,
        "skel_ang_thr"     : 30,
        "depth_scale"      : 1000,
    }

    app = pub_seg_lv_gb_node.PUB_SEG_LV_GB(params)
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()