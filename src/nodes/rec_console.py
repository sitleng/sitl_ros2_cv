#!/usr/bin/env python3

import rospy
import rosbag
import message_filters
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, PoseStamped
from sitl_dvrk.msg import BoolStamped

class RECORD_KINEMATICS():
    def __init__(self,params,topic_names):
        self.bag = rosbag.Bag(params["save_dir"]+"/glove_console.bag",'w',allow_unindexed=True,skip_index=True)
        self.topics = topic_names

    def callback(self,*msgs):
        for i, msg in enumerate(msgs):
            self.bag.write(self.topics[i],msg,msg.header.stamp)
        
if __name__ == "__main__":
    rospy.init_node("glove_console")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)

    topic_names = [
        "/pedals/clutch",
        "/MTML/gripper/measured_js",
        "/MTML/local/measured_cp",
        "/MTML/measured_js",
        "/MTML/measured_cp",
        "/PSM2/custom/setpoint_cp",
        "/PSM2/custom/local/setpoint_cp",
        "/PSM2/jaw/measured_js",
        "/PSM2/measured_js"
    ]

    app = RECORD_KINEMATICS(params,topic_names)

    rospy.loginfo("Start recording kinematics...")
    
    msgs = [
        message_filters.Subscriber(topic_names[0], BoolStamped),
        message_filters.Subscriber(topic_names[1], JointState ),
        message_filters.Subscriber(topic_names[2], PoseStamped),
        message_filters.Subscriber(topic_names[3], JointState ),
        message_filters.Subscriber(topic_names[4], PoseStamped),
        message_filters.Subscriber(topic_names[5], PoseStamped),
        message_filters.Subscriber(topic_names[6], PoseStamped),
        message_filters.Subscriber(topic_names[7], JointState ),
        message_filters.Subscriber(topic_names[8], JointState ),
    ]

    ts = message_filters.ApproximateTimeSynchronizer(msgs, queue_size=10, slop=0.01)
    ts.registerCallback(app.callback)
    try:
        rospy.spin()
    finally:
        print("Saving bag")
        app.bag.close()
    