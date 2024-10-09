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