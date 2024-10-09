import rclpy

from nodes import image_view_node

import cv2

def main(args=None):
    rclpy.init(args=args)

    params = {
        # "topic_name": "/ecm/right_rect/image_color",
        # "img_type"  : "compressed",
        "topic_name": "/ecm/recon3d/pclimg",
        "img_type"  : "compressed",
    }

    image_view = image_view_node.IMAGE_VIEW(params)

    rclpy.spin(image_view)
    image_view.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
