#!/usr/bin/env python3

import rospy
from sitl_dvrk.msg import Float64Stamped

class PUB_TAPNET_PARAMS(object):
    def __init__(self):
        self.last_click_time = 0
        self.pub_lct = rospy.Publisher("/tapnet/last_click_time", Float64Stamped, queue_size=10)

    def __del__(self):
        print("Shutting down PUB_TAPNET_PARAMS...")

    def callback(self, lct_msg):
        self.last_click_time = lct_msg.data

    def run(self):
        t = rospy.Time.now()
        lct_msg = Float64Stamped()
        lct_msg.header.stamp = t
        lct_msg.data = self.last_click_time
        self.pub_lct.publish(lct_msg)

if __name__=="__main__":
    rospy.init_node("pub_tapnet_params")
    app = PUB_TAPNET_PARAMS()
    # The endoscope frequency is around 60Hz
    R = rospy.Rate(60)

    rospy.Subscriber("/tapnet/last_click_time", Float64Stamped, app.callback)

    try:
        rospy.loginfo("Publishing Params for Collecting Points used in TAP")
        while not rospy.is_shutdown():
            app.run()
            R.sleep()
    except Exception as e:
        rospy.logdebug(e)