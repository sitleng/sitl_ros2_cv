#!/usr/bin/env python3

# Import open source libraries

# Import ROS libraries
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
import message_filters
from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2

# Import custom libraries
from sitl_dvrk.msg import UInt16MultiArrayStamped
from utils import pcl_utils, ma_utils, tf_utils, misc_utils

class PUB_SEG_SAMPLE_3D():
    def __init__(self,params):
        self.params = params
        self.pub_sample_ctrd_3d = rospy.Publisher("/seg/sample/ctrd_3d", PointStamped, queue_size=10)
        self.pub_cnt_3d  = rospy.Publisher("/seg/sample/cnt_3d", PointCloud2, queue_size=10)
        self.pub_bnd_3d  = rospy.Publisher("/seg/sample/bnd_3d", PointCloud2, queue_size=10)

    def __del__(self):
        print("Destructing class PUB_SEG_SAMPLE_3D...")

    def callback(self, pcl_msg, cnt_2d_msg, bnd_2d_msg, sample_c_2d_msg):
        # Load subscribed messages
        pcl = pcl_utils.xyzarr_to_nparr(pointcloud2_to_array(pcl_msg))
        cnt_2d = ma_utils.ma2arr(cnt_2d_msg)
        bnd_2d = ma_utils.ma2arr(bnd_2d_msg)
        sample_c_2d = ma_utils.ma2arr(sample_c_2d_msg)

        # Publish boundary points (3D)
        cnt_3d = pcl_utils.win_avg_3d_pts(cnt_2d, pcl, self.params["window_size"], self.params["window_thr"], cnt_2d)
        cnt_3d = misc_utils.remove_outliers_iso(cnt_3d)
        cnt_3d = misc_utils.interp_3d(cnt_3d, cnt_3d.shape[0])
        if cnt_3d is not None:
            self.pub_cnt_3d.publish(
                array_to_pointcloud2(
                    pcl_utils.pts_array_3d_to_pcl(
                        cnt_3d,
                        color = (0,255,0)
                    ),
                    frame_id = pcl_msg.header.frame_id,
                    stamp = rospy.Time.now()
                )
            )

        # Publish boundary points (3D)
        bnd_3d = pcl_utils.win_avg_3d_pts(bnd_2d, pcl, self.params["window_size"], self.params["window_thr"], bnd_2d)
        bnd_3d = misc_utils.remove_outliers_iso(bnd_3d)
        bnd_3d = misc_utils.interp_3d(bnd_3d, bnd_2d.shape[0])
        if bnd_3d is not None:
            self.pub_bnd_3d.publish(
                array_to_pointcloud2(
                    pcl_utils.pts_array_3d_to_pcl(
                        bnd_3d,
                        color = (0,0,255)
                    ),
                    frame_id = pcl_msg.header.frame_id,
                    stamp = rospy.Time.now()
                )
            )

        # Publish gallbladder sample centroid (3D)
        sample_c_3d = pcl_utils.win_avg_3d_pt(
            sample_c_2d, pcl, self.params["window_size"], self.params["window_thr"], sample_c_2d
        )
        if sample_c_3d is not None:
            self.pub_sample_ctrd_3d.publish(
                tf_utils.pt3d2ptstamped(
                    sample_c_3d, rospy.Time.now(), pcl_msg.header.frame_id
                )
            )
        

if __name__ == '__main__':
    rospy.init_node("pub_dt2_sample_bnd_3d")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)
    try:
        app = PUB_SEG_SAMPLE_3D(params)
        rospy.loginfo("Start segmenting sample... ")
        topic_list = []
        topic_list.append(message_filters.Subscriber("/ecm/points2", PointCloud2))
        topic_list.append(message_filters.Subscriber("/seg/sample/cnt_2d", UInt16MultiArrayStamped))
        topic_list.append(message_filters.Subscriber("/seg/sample/bnd_2d", UInt16MultiArrayStamped))
        topic_list.append(message_filters.Subscriber("/seg/sample/ctrd_2d", UInt16MultiArrayStamped))
        ts = message_filters.ApproximateTimeSynchronizer(
            topic_list, slop=0.05, queue_size=10
        )
        ts.registerCallback(app.callback)
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)
    finally:
        del app