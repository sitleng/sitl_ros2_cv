#! /usr/bin/env python3

import os
from readchar import readkey

import rospy
from sitl_dvrk.msg import StringStamped

if __name__ == "__main__":
    rospy.init_node("grasp_key_input")
    pub_key = rospy.Publisher("/keyboard/grasp", StringStamped, queue_size=10)
    rate = rospy.Rate(50)
    msg = StringStamped()
    while not rospy.is_shutdown():
        print("Press one of the following keys...")
        print("p: pre-locate instrument before grasping.")
        print("o: open the jaws of the instrument.")
        print("c: close the jaws of the instrument.")
        print("g: grasp and pull the target object.")
        key = readkey()
        msg.header.stamp = rospy.Time.now()
        msg.data = key
        pub_key.publish(msg)
        rate.sleep()
        os.system('clear')