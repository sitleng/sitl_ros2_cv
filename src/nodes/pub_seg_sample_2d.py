#!/usr/bin/env python3

# Import open source libraries
import numpy as np
import cv2
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize

# Import ROS libraries
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

# Import custom libraries
from sitl_dvrk.msg import UInt16MultiArrayStamped
from utils import ma_utils, misc_utils, cv_utils

class PUB_SEG_SAMPLE_2D():
    def __init__(self,params):
        self.br = CvBridge()
        self.num_pts_cnt = params["num_pts_cnt"]
        self.num_pts_bnd = params["num_pts_bnd"]
        self.adj_cnst_list = params["adj_cnst_list"]
        self.adj_dist_upp_bnd = params["adj_dist_upp_bnd"]
        self.pub_sample_mask = rospy.Publisher("/seg/sample/mask", CompressedImage, queue_size=10)
        self.pub_sample_ctrd_2d = rospy.Publisher("/seg/sample/ctrd_2d", UInt16MultiArrayStamped, queue_size=10)
        self.pub_cnt_2d  = rospy.Publisher("/seg/sample/cnt_2d", UInt16MultiArrayStamped, queue_size=10)
        self.pub_bnd_2d  = rospy.Publisher("/seg/sample/bnd_2d", UInt16MultiArrayStamped, queue_size=10)

    def __del__(self):
        print("Destructing class PUB_SEG_SAMPLE_2D...")

    def get_sample_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #set the lower and upper bounds for the blue hue
        lower_thr = np.array([80, 55, 110])
        upper_thr = np.array([130, 255, 255])

        #create a mask for blue colour using inRange function
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
        mask = cv2.inRange(hsv, lower_thr, upper_thr)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=1)
        return mask
    
    def get_sample_bnd(self, img, cnt_2d):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #set the lower and upper bounds for the red hue
        lower_thr = np.array([0, 20, 10])
        upper_thr = np.array([10, 255, 255])

        #create a mask for red colour using inRange function
        mask = cv2.inRange(hsv, lower_thr, upper_thr)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))

        # Get the Skeleton of the masked red tube
        skel = skeletonize(mask, method='lee')
        temp = np.where(skel > 0)
        bnd_2d = np.column_stack((temp[1], temp[0]))
        bnd_2d = misc_utils.farthest_first_traversal(bnd_2d, self.num_pts_bnd)
        # bnd_tree = KDTree(bnd_2d)
        # c_dist, c_inds = bnd_tree.query(cnt_2d, k=1, distance_upper_bound=self.adj_dist_upp_bnd, workers=-1)
        # bnd_2d = bnd_2d[c_inds[~np.isinf(c_dist)]]
        if bnd_2d.ndim != 2:
            return bnd_2d
        cost_matrix = cdist(bnd_2d, cnt_2d)
        _, col_ind = linear_sum_assignment(cost_matrix)
        bnd_2d = cnt_2d[col_ind,:]
        bnd_2d = bnd_2d[
            np.logical_and(
                np.logical_and(bnd_2d[:,0]>= self.adj_cnst_list[0]*img.shape[1], bnd_2d[:,0]<=self.adj_cnst_list[1]*img.shape[1]),
                np.logical_and(bnd_2d[:,1]>= self.adj_cnst_list[2]*img.shape[0], bnd_2d[:,1]<=self.adj_cnst_list[3]*img.shape[0])
            )
        ]
        return bnd_2d

    def callback(self, cam1_rect_color_msg):
        # Load subscribed messages
        img = self.br.compressed_imgmsg_to_cv2(cam1_rect_color_msg)

        # Publish gallbladder sample mask
        sample_mask = self.get_sample_mask(img)
        mask_msg = self.br.cv2_to_compressed_imgmsg(sample_mask)
        mask_msg.header.stamp = rospy.Time.now()
        self.pub_sample_mask.publish(mask_msg)

        # Publish sample boundary points (2D)
        mask_cnts = cv_utils.mask_cnts(sample_mask)
        cnt_2d = cv_utils.max_contour(mask_cnts)[1]
        if cnt_2d is not None:
            cnt_2d = misc_utils.farthest_first_traversal(cnt_2d, self.num_pts_cnt)
            cnt_2d_msg = ma_utils.arr2uint16ma(cnt_2d, rospy.Time.now())
            self.pub_cnt_2d.publish(cnt_2d_msg)
            bnd_2d = self.get_sample_bnd(img, cnt_2d)
            bnd_2d_msg = ma_utils.arr2uint16ma(bnd_2d, rospy.Time.now())
            self.pub_bnd_2d.publish(bnd_2d_msg)

        # Publish mask centroid (2D)
        sample_c_2d = cv_utils.cnt_centroid(cnt_2d)
        if sample_c_2d is not None:
            self.pub_sample_ctrd_2d.publish(
                ma_utils.arr2uint16ma(
                    sample_c_2d, rospy.Time.now()
                )
            )


if __name__ == '__main__':
    rospy.init_node("pub_dt2_sample_bnd_2d")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)
    try:
        app = PUB_SEG_SAMPLE_2D(params)
        if params["calib_dir"] == "L2R":
            rospy.Subscriber("/ecm/left_rect/image_color", CompressedImage, app.callback)
        elif params["calib_dir"] == "R2L":
            rospy.Subscriber("/ecm/right_rect/image_color", CompressedImage, app.callback)
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)
    finally:
        del app