#!/usr/bin/env python3

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

def custom_qos_profile(queue_size):
    return QoSProfile(
        history                   = HistoryPolicy.KEEP_LAST,
        depth                     = queue_size,
        reliability               = ReliabilityPolicy.RELIABLE,
        durability                = DurabilityPolicy.VOLATILE,
        # deadline                  = ,
        # lifespan                  = ,
        # liveliness                = ,
        # liveliness_lease_duration = 
    )

def now(node):
    return node.get_clock().now().to_msg()

def loginfo(node, str, once_flag=False):
    node.get_logger().info(str, once=once_flag)