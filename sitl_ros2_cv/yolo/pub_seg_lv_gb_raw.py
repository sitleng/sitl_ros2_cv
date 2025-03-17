import rclpy

import os
from nodes.yolo import pub_seg_lv_gb_raw

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"             : "pub_seg_lv_gb_2d_yolo",
        "queue_size"            : 5,
        # "refimg_topic"         : "/video/left/rect/image_color",
        "refimg_topic"          : "/ecm/left/rect/image_color",
        "model_path"            : "/home/"+os.getlogin()+"/yolo/liver_seg/v11l_crcd_temp2.pt",
        # "model_path"            : "/home/"+os.getlogin()+"/yolo/liver_seg/v11l_crcd_final.pt",
        "conf_thr"              : 1e-3,
        "cnt_area_thr"          : 5000,
        "downsample_pixel_dist" : 20,
    }

    app = pub_seg_lv_gb_raw.PUB_SEG_LV_GB_RAW(params)
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()