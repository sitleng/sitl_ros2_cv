#!/usr/bin/env python3

# Import open source libraries
import numpy as np
from tqdm import tqdm
import os

# Import ROS libraries
import rospy
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
from cv_bridge import CvBridge
import message_filters
from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2

# Import custom libraries
from utils import dt2_utils, pcl_utils

class PUB_DT2_BNDPTS():
    def __init__(self,params):
        self.br = CvBridge()
        self.params = params
        self.pub_goal_pcl  = rospy.Publisher("/dt2/goals",PointCloud2,queue_size=10)
        self.seg_predictor = dt2_utils.load_seg_predictor(
            params['model_path'],params['score_thresh']
        )
        self.seg_metadata = dt2_utils.load_seg_metadata()
        self.bnd_pts_2d = None
        self.bnd_pts_3d = None
        self.N = 0
        self.pbar = tqdm(total=1000)

    def __del__(self):
        print("Destructing class PUB_DT2_BNDPTS...")

    def callback(self,cam1_rect_color_msg,disp_msg,pcl_msg):
        img = self.br.compressed_imgmsg_to_cv2(cam1_rect_color_msg)
        # disp = self.br.imgmsg_to_cv2(disp_msg)
        disp = None
        if self.bnd_pts_2d is None:
            rospy.loginfo_once("Detecting 2D boundary points...")
            self.bnd_pts_2d = dt2_utils.process_draw_img_seg(
                self.params["cnt_area_thr"],
                self.params["num_goals"],
                self.params["obj_type"],
                self.seg_predictor,
                self.seg_metadata,
                img, disp,
                self.params["save_path"]
            )[-1]
        elif self.bnd_pts_3d is None or self.N < 1000:
            rospy.loginfo_once("Recording 3D Boundary Points:...")
            pcl = pointcloud2_to_array(pcl_msg)
            temp = pcl[self.bnd_pts_2d[:,1],self.bnd_pts_2d[:,0]]
            if not(np.isinf(temp['x']).any() or np.isinf(temp['y']).any() or np.isinf(temp['z'])).any():
                temp = np.hstack(
                    (temp["x"].reshape(-1,1),temp["y"].reshape(-1,1),temp["z"].reshape(-1,1))
                )
                if self.bnd_pts_3d is None:
                    self.bnd_pts_3d = temp
                else:
                    self.bnd_pts_3d = np.dstack((self.bnd_pts_3d,temp))
                self.pbar.update()
                self.N += 1
        else:
            rospy.loginfo_once("Found Average 3D Boundary Points...")
            rospy.loginfo_once("Publishing the 3D Boundary Points...")
            if hasattr(self,"seg_predictor"):
                del self.seg_predictor
            if self.bnd_pts_3d.ndim == 3:
                self.bnd_pts_3d = np.mean(self.bnd_pts_3d,axis=2)
                self.bnd_pts_3d = np.concatenate(
                    [self.bnd_pts_3d[0]-np.array([0,0,0.01]).reshape(1,3),self.bnd_pts_3d]
                )
            bnd_pts_3d_pcl = array_to_pointcloud2(
                pcl_utils.pts_array_3d_to_pcl(
                    self.bnd_pts_3d,
                    color = (0,0,255)
                ),
                frame_id = self.params["frame_id"],
                stamp = rospy.Time.now()
            )
            self.pub_goal_pcl.publish(bnd_pts_3d_pcl)

if __name__ == '__main__':
    rospy.init_node("pub_dt2_bndpts")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)
    try:
        app = PUB_DT2_BNDPTS(params)
        rospy.loginfo("Start detecting boundary points... ")
        disp_msg = message_filters.Subscriber("/ecm/disparity",Image)
        pcl_msg  = message_filters.Subscriber("/ecm/points2",PointCloud2)
        if params["calib_dir"] == "L2R":
            left_rect_color_msg  = message_filters.Subscriber("/ecm/left_rect/image_color",CompressedImage)
            ts                   = message_filters.ApproximateTimeSynchronizer(
                [left_rect_color_msg,disp_msg,pcl_msg],
                slop=0.02, queue_size=10
            )
        elif params["calib_dir"] == "R2L":
            right_rect_color_msg = message_filters.Subscriber("/ecm/right_rect/image_color",CompressedImage)
            ts                   = message_filters.ApproximateTimeSynchronizer(
                [right_rect_color_msg,disp_msg,pcl_msg],
                slop=0.02, queue_size=10
            )
        ts.registerCallback(app.callback)
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)
    finally:
        del app