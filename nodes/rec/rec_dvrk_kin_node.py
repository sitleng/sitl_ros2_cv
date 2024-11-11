#!/usr/bin/env python3

# ros libraries
from rclpy.node import Node
from rclpy.serialization import serialize_message
import rosbag2_py
import message_filters
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped

# custom libraries
from utils import ros2_utils

class REC_DVRK_KIN(Node):
    def __init__(self, params, topic_names):
        super().__init__(params['node_name'])

        # initialize bag
        self.bag_writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(
            uri=params["save_path"] + "/dvrk_kinematics",
            storage_id='sqlite3'
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        self.bag_writer.open(storage_options, converter_options)

        self.topics = topic_names
        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.gen_subs(topic_names),
            queue_size=params["queue_size"],
            slop=params["slop"]
        )
        self.ts.registerCallback(self.callback)

    def gen_subs(self, topic_names):
        subs = []
        for topic_name in topic_names:
            if 'cp' in topic_name:
                subs.append(
                    message_filters.Subscriber(self, topic_name, TransformStamped),
                )
            elif 'js' in topic_name:
                subs.append(
                    message_filters.Subscriber(self, topic_name, JointState)
                )
        return subs
    
    def close_bag(self):
        self.bag_writer.close()

    def callback(self, *msgs):
        for i, msg in enumerate(msgs):
            self.bag_writer.write(
                self.topics[i],
                serialize_message(msg),
                ros2_utils.to_sec(msg)
            )
