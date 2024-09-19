#!/usr/bin/env python3

# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from sitl_dvrk.msg import BoolStamped, Float64Stamped
from geometry_msgs.msg import PointStamped
from utils import tapnet_utils

import cv2
import time
import traceback

class COLLECT_TAP_PTS(object):
    def __init__(self):
        self.br = CvBridge()
        self.frame = None
        # self.dsize = (256, 256)
        self.dsize = (480, 480)
        self.last_click_time = 0
        self.pub_pos = rospy.Publisher("/tapnet/pos",             PointStamped,   queue_size=10)
        self.pub_qf  = rospy.Publisher("/tapnet/query_frame",     BoolStamped,    queue_size=10)
        self.pub_lct = rospy.Publisher("/tapnet/last_click_time", Float64Stamped, queue_size=10)

    def __del__(self):
        print("Shutting down COLLECT_TAP_PTS...")

    def mouse_click(self, event, x, y, flags, param):
        del flags, param
        # event fires multiple times per click sometimes??
        if (time.time() - self.last_click_time) < 0.5:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            t = rospy.Time.now()

            pos_msg = PointStamped()
            pos_msg.header.stamp = t
            # pos_msg.point.x = self.frame.shape[1] - x
            pos_msg.point.x = x
            pos_msg.point.y = y
            self.pub_pos.publish(pos_msg)

            qf_msg = BoolStamped()
            qf_msg.header.stamp = t
            qf_msg.data = True
            self.pub_qf.publish(qf_msg)
            
            lct_msg = Float64Stamped()
            lct_msg.header.stamp = t
            lct_msg.data = time.time()
            self.pub_lct.publish(lct_msg)

    def frame_callback(self, frame_msg):
        # self.frame = cv_cuda_utils.cvmat_resize(
        #     image = self.br.compressed_imgmsg_to_cv2(frame_msg),
        #     dsize = (256, 256)
        # )
        self.frame = tapnet_utils.get_frame_ros(frame_msg, self.dsize)

    def lct_callback(self, lct_msg):
        self.last_click_time = lct_msg.data

if __name__ == "__main__":
    rospy.init_node("collect_tap_pts")
    R = rospy.Rate(60)
    try:
        app = COLLECT_TAP_PTS()
        # sub_frame = message_filters.Subscriber("/ecm/left_rect/image_color", CompressedImage)
        # sub_frame = message_filters.Subscriber("/loop_video/frame", CompressedImage)
        # sub_frame = message_filters.Subscriber("/lama/frame", CompressedImage)
        # sub_lct   = message_filters.Subscriber("/tapnet/last_click_time", Float64Stamped)
        # ts = message_filters.ApproximateTimeSynchronizer([sub_frame, sub_lct], slop=0.1, queue_size=10)
        # ts.registerCallback(app.callback)

        rospy.Subscriber("/dt2/masks", CompressedImage, app.frame_callback)
        rospy.Subscriber("/tapnet/last_click_time", Float64Stamped, app.lct_callback)
        cv2.namedWindow("Choose Points")
        cv2.setMouseCallback("Choose Points", app.mouse_click)
        while not rospy.is_shutdown():
            if app.frame is not None:
                cv2.imshow("Choose Points", app.frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            # else:
            #     print(app.frame)
            R.sleep()
    except Exception:
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()