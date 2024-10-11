import rclpy

import os
from nodes.video import pub_video_node

def main(args=None):
    rclpy.init(args=args)

    params = {
        "node_name"  : "pub_video_left",
        "video_path" : "/home/"+os.getlogin()+"/sitl_recs/left.mp4",
        "queue_size" : 5,
    }

    app = pub_video_node.PUB_VIDEO(params)
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()