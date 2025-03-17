import rclpy

from nodes.detect import pub_seg_lv_gb_post

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"       : "pub_seg_lv_gb_post",
        "queue_size"      : 5,
        "slop"            : 0.3,
        "adj_dub"         : 0.005, # dub for kdtree to find adjacent 3D points
        "adj_segs_thr"    : 0.005, # threshold to segment the adjacent trajectory
        "frame_id"        : 'ecm_left',
    }

    app = pub_seg_lv_gb_post.PUB_SEG_LV_GB_POST(params)
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()