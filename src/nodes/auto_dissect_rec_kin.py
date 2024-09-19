#!/usr/bin/env python3

import rospy
import rosbag
import message_filters
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped

class RECORD_KINEMATICS():
    def __init__(self,params,topic_names):
        self.bag = rosbag.Bag(params["save_dir"]+"/auto_dissect_kin.bag",'w',allow_unindexed=True,skip_index=True)
        self.topics = topic_names

    def callback(self,*msgs):
        for i, msg in enumerate(msgs):
            self.bag.write(self.topics[i],msg,msg.header.stamp)
        
if __name__ == "__main__":
    rospy.init_node("auto_dis_rec_kin")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)

    topic_names = [
        "/dt2/PCH/instjaw",
        "/kf/dt2/PCH/instjaw",
        "/dt2/goals",
    ]

    app = RECORD_KINEMATICS(params,topic_names)

    rospy.loginfo("Start recording kinematics...")
    
    msgs = [
        message_filters.Subscriber(topic_names[0], TransformStamped),
        message_filters.Subscriber(topic_names[1], TransformStamped),
        message_filters.Subscriber(topic_names[2], PointCloud2     ),
    ]

    ts = message_filters.ApproximateTimeSynchronizer(msgs, queue_size=10, slop=0.1)
    ts.registerCallback(app.callback)
    try:
        rospy.spin()
    finally:
        print("Saving bag")
        app.bag.close()
    