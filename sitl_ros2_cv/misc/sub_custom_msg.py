import rclpy
from rclpy.node import Node

from sitl_dvrk_ros2_interfaces.msg import Float32Stamped

class SUB_CUSTOM_MSG(Node):
    def __init__(self):
        super().__init__("sub_custom_msg")
        self.sub_msg = self.create_subscription(
            Float32Stamped,
            "/custom_topic",
            self.callback,
            10
        )

    def callback(self, msg):
        self.get_logger().info(f"{msg.data}")

def main(args=None):
    rclpy.init(args=args)
    app = SUB_CUSTOM_MSG()
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()