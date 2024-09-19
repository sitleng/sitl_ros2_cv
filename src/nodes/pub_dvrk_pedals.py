#! /usr/bin/env python3
import rospy
from sensor_msgs.msg import Joy
from sitl_dvrk.msg import BoolStamped

class PUB_DVRK_PEDALS(object):
    def __init__(self):
        self.camera_msg = BoolStamped()
        self.camera_msg.data = False
        self.clutch_msg = BoolStamped()
        self.clutch_msg.data = False
        self.pub_clutch = rospy.Publisher("/pedals/clutch",BoolStamped,queue_size=10)
        self.pub_camera = rospy.Publisher("/pedals/camera",BoolStamped,queue_size=10)
        
    def clutch_cb(self,clutch_joy):
        if clutch_joy.buttons[0] == 0:
            self.clutch_msg.data = False
        elif clutch_joy.buttons[0] == 1:
            self.clutch_msg.data = True

    def camera_cb(self,camera_joy):
        if camera_joy.buttons[0] == 0:
            self.camera_msg.data = False
        elif camera_joy.buttons[0] == 1:
            self.camera_msg.data = True

if __name__ == '__main__':
    rospy.init_node("pub_dvrk_pedals")

    app = PUB_DVRK_PEDALS()

    try:
        sub_clutch = rospy.Subscriber("/footpedals/clutch",Joy,app.clutch_cb)
        sub_camera = rospy.Subscriber("/footpedals/camera",Joy,app.camera_cb)

        r = rospy.Rate(100)

        while not rospy.is_shutdown():
            app.clutch_msg.header.stamp = rospy.Time.now()
            app.camera_msg.header.stamp = rospy.Time.now()
            app.pub_clutch.publish(app.clutch_msg)
            app.pub_camera.publish(app.camera_msg)
            r.sleep()
    finally:
        pass