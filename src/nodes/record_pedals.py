#!/usr/bin/env python3

import rospy
import rosbag
import message_filters
from sitl_dvrk.msg import BoolStamped

class RECORD_PEDALS():
    def __init__(self,params,topic_names):
        self.bag = rosbag.Bag(params["save_dir"]+"/dvrk_pedals.bag",'w',allow_unindexed=True,skip_index=True)
        self.topics = topic_names

    def callback(self,*msgs):
        for i, msg in enumerate(msgs):
            self.bag.write(self.topics[i],msg,msg.header.stamp)
        
if __name__ == "__main__":
    rospy.init_node("record_pedals")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)

    topic_names = [
        "/pedals/read/monopolar",
        # "/pedals/write/monopolar",
        # "/pedals/bipolar",
        "/pedals/clutch",
        "/pedals/camera"
    ]

    app = RECORD_PEDALS(params,topic_names)

    rospy.loginfo("Start recording pedals...")
    
    msgs = [
        message_filters.Subscriber(topic_names[0], BoolStamped),
        message_filters.Subscriber(topic_names[1], BoolStamped),
        message_filters.Subscriber(topic_names[2], BoolStamped)
        # message_filters.Subscriber(topic_names[3], BoolStamped)
    ]

    ts = message_filters.ApproximateTimeSynchronizer(msgs, queue_size=10, slop=0.01)
    ts.registerCallback(app.callback)
    try:
        rospy.spin()
    finally:
        print("Saving bag")
        app.bag.close()
    