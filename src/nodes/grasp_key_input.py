#! /usr/bin/env python3

from readchar import readkey

import rospy
from sitl_dvrk.msg import StringStamped



if __name__ == "__main__":
    rospy.init_node("grasp_key_input")
    pub_key = rospy.Publisher("/keyboard/grasp", StringStamped, queue_size=10)
    rate = rospy.Rate(30)
    msg = StringStamped()
    rospy.loginfo("Press one of the following keys...")
    rospy.loginfo("a: align forceps before grasping.")
    rospy.loginfo("g: grasp the target tissue.")
    rospy.loginfo("p: pull the target tissue.")
    rospy.loginfo("r: release the target tissue.")
    rospy.loginfo("o: open the jaws of the forceps.")
    rospy.loginfo("c: close the jaws of the forceps.")
    rospy.loginfo("i: go to the initial position.")
    try:
        while not rospy.is_shutdown():
            key = readkey()
            msg.header.stamp = rospy.Time.now()
            msg.data = key
            pub_key.publish(msg)
            rate.sleep()
    except Exception as e:
        rospy.logerr(e)
    finally:
        print("")