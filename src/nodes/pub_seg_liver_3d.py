#!/usr/bin/env python3

# Import open source libraries
import numpy as np
import traceback

# Import ROS libraries
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
import message_filters
from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2

# Import custom libraries
from sitl_dvrk.msg import UInt16MultiArrayStamped
from utils import pcl_utils, ma_utils, tf_utils, misc_utils

class PUB_SEG_LIVER_3D():
    def __init__(self,params):
        self.params = params
        self.pub_bnd_3d = rospy.Publisher("/seg/liver_gallb/bnd_3d", PointCloud2, queue_size=1)
        self.pub_gallb_cnt_3d = rospy.Publisher("/seg/gallbladder/cnt_3d", PointCloud2, queue_size=1)
        self.pub_gallb_skel_3d = rospy.Publisher("/seg/gallbladder/skel_3d", PointCloud2, queue_size=1)
        self.pub_gallb_ctrd_3d = rospy.Publisher("/seg/gallbladder/ctrd_3d", PointStamped, queue_size=1)
        self.pub_gallb_grasp_3d = rospy.Publisher("/seg/gallbladder/grasp_3d", PointStamped, queue_size=1)

    def __del__(self):
        print("Destructing class PUB_SEG_LIVER_3D...")

    def callback(self, pcl_msg, gallb_cnt_2d_msg, gallb_skel_2d_msg, gallb_c_2d_msg):
        try:
            # Load subscribed messages
            pcl = pcl_utils.xyzarr_to_nparr(pointcloud2_to_array(pcl_msg))
            
            gallb_cnt_2d = ma_utils.ma2arr(gallb_cnt_2d_msg)
            gallb_skel_2d = ma_utils.ma2arr(gallb_skel_2d_msg)
            gallb_c_2d = ma_utils.ma2arr(gallb_c_2d_msg)
            
            # Publish contour points (3D)
            cnt_3d = pcl_utils.win_avg_3d_pts(
                gallb_cnt_2d, pcl, self.params["window_size"], self.params["mad_thr"]
            )
            # cnt_3d = misc_utils.smooth_cnt(cnt_3d, 0.5)
            # cnt_3d = misc_utils.remove_outliers_dbs(cnt_3d)
            if cnt_3d is not None:
                self.pub_gallb_cnt_3d.publish(
                    array_to_pointcloud2(
                        pcl_utils.pts_array_3d_to_pcl(
                            cnt_3d,
                            color = (0, 0, 255)
                        ),
                        frame_id = pcl_msg.header.frame_id,
                        stamp = rospy.Time.now()
                    )
                )

            # Publish gallbladder skeleton (3D)
            gallb_skel_3d = pcl_utils.win_avg_3d_pts(
                gallb_skel_2d, pcl, self.params["window_size"], self.params["mad_thr"]
            )
            if gallb_skel_3d is not None:
                self.pub_gallb_skel_3d.publish(
                    array_to_pointcloud2(
                        pcl_utils.pts_array_3d_to_pcl(
                            gallb_skel_3d,
                            color = (0, 255, 0)
                        ),
                        frame_id = pcl_msg.header.frame_id,
                        stamp = rospy.Time.now()
                    )
                )

            # Publish gallbladder centroid (3D)
            gallb_c_3d = pcl_utils.win_avg_3d_pt(
                gallb_c_2d, pcl, self.params["window_size"], self.params["mad_thr"]
            )
            if gallb_c_3d is not None:
                self.pub_gallb_ctrd_3d.publish(
                    tf_utils.pt3d2ptstamped(
                        gallb_c_3d, rospy.Time.now(), pcl_msg.header.frame_id
                    )
                )

            # Publish Boundary points (3D)
            if cnt_3d is None or gallb_skel_3d is None:
                return
            nn_inds = misc_utils.nn_kdtree(cnt_3d, gallb_skel_3d, self.params["adj_dist_upp_bnd"])[-1]
            nn_inds = misc_utils.rm_outlier_mad(np.unique(nn_inds), thresh=2)
            if nn_inds.size < 3:
                return
            nn_inds = np.arange(nn_inds[0], nn_inds[-1]+1)
            bnd_3d = cnt_3d[nn_inds]
            bnd_3d = bnd_3d[np.argsort(bnd_3d[:,0])[::-1]]
            if bnd_3d is not None:
                self.pub_bnd_3d.publish(
                    array_to_pointcloud2(
                        pcl_utils.pts_array_3d_to_pcl(
                            bnd_3d,
                            color = (255,0,0)
                        ),
                        frame_id = pcl_msg.header.frame_id,
                        stamp = rospy.Time.now()
                    )
                )

            # Publish gallbladder grasping point (3D)
            grasp_3d = (gallb_skel_3d.mean(axis=0) + bnd_3d.mean(axis=0)) / 2
            if gallb_c_3d is not None:
                self.pub_gallb_grasp_3d.publish(
                    tf_utils.pt3d2ptstamped(
                        grasp_3d, rospy.Time.now(), pcl_msg.header.frame_id
                    )
                )
        except:
            traceback.print_exc()


if __name__ == '__main__':
    rospy.init_node("pub_seg_liver_3d")
    node_name = rospy.get_name()
    params = rospy.get_param(node_name)
    try:
        app = PUB_SEG_LIVER_3D(params)
        rospy.loginfo("Start detecting boundary points... ")
        topic_list = []
        topic_list.append(message_filters.Subscriber("/ecm/points2", PointCloud2))
        topic_list.append(message_filters.Subscriber("/seg/gallbladder/cnt_2d", UInt16MultiArrayStamped))
        topic_list.append(message_filters.Subscriber("/seg/gallbladder/skel_2d", UInt16MultiArrayStamped))
        topic_list.append(message_filters.Subscriber("/seg/gallbladder/ctrd_2d", UInt16MultiArrayStamped))
        ts = message_filters.ApproximateTimeSynchronizer(
            topic_list, slop=0.2, queue_size=1
        )
        ts.registerCallback(app.callback)
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()
    except Exception as e:
        rospy.logerr(e)
    finally:
        del app