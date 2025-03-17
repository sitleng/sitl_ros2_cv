import rclpy

import os
from nodes.dt2 import pub_seg_lv_gb_raw

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"              : "pub_seg_lv_gb_dt2",
        "queue_size"             : 5,
        # "refimg_topic"    : "/video/left/rect/image_color",
        "refimg_topic"           : "/ecm/left/rect/image_color",

        # "model_type"             : "MaskDINO",
        # # "model_weights"          : "/home/"+os.getlogin()+"/MaskDINO/output_crcd/model_final.pth",
        # "model_weights"          : "/home/"+os.getlogin()+"/MaskDINO/output_crcd_temp/model_final.pth",
        # "config_file"            : "/home/"+os.getlogin()+"/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_custom.yaml",

        "model_type"             : "Detectron2",
        # "model_weights"          : "/home/"+os.getlogin()+"/dt2_dataset/liver/crcd_res/model_final_v2.pth",
        "model_weights"          : "/home/"+os.getlogin()+"/dt2_dataset/liver/results_v5_re/model_final.pth", # BioRob 2024

        "pred_score_thr"         : 0.5,
        "cnt_area_thr"           : 10000,
        "downsample_pixel_dist"  : 20,
    }

    app = pub_seg_lv_gb_raw.PUB_SEG_LV_GB_RAW(params)
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()