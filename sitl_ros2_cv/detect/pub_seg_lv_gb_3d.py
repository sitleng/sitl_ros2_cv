import rclpy

from nodes.detect import pub_seg_lv_gb_3d

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"             : "pub_seg_lv_gb_3d",
        "queue_size"            : 5,
        # "pclimg_topic"         : "/video/recon3d/pclimg",
        "pclimg_topic"          : "/ecm/recon3d/pclimg",
        "window_size"           : 15,
        "mad_thr"               : 1.5,
        "downsample_pixel_dist" : 15,
    }

    app = pub_seg_lv_gb_3d.PUB_SEG_LV_GB_3D(params)
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()