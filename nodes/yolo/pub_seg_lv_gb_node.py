# Import ROS libraries
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import message_filters

# Import open source libraries
import numpy as np
import cv2
import time

# Import custom libraries
from utils import cv_utils, misc_utils, pcl_utils, ros2_utils, tf_utils
from utils import seg_utils, yolo_utils

class PUB_SEG_LV_GB_YOLO(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        self.br = CvBridge()
        self.load_params(params)
        self.seg_predictor = yolo_utils.load_model(params['model_path'])

        qos_profile = ros2_utils.custom_qos_profile(params["queue_size"])
        self.pub_lv_gb_cnts  = self.create_publisher(PointCloud2 , "cnts" , qos_profile)
        self.pub_lv_gb_adj_cnts = self.create_publisher(PointCloud2, "adj_cnts" , qos_profile)
        self.pub_lv_gb_skel  = self.create_publisher(PointCloud2 , "skel" , qos_profile)
        self.pub_lv_gb_bnd   = self.create_publisher(PointCloud2 , "bnd"  , qos_profile)
        self.pub_lv_gb_ctrd  = self.create_publisher(PointStamped, "ctrd" , qos_profile)
        self.pub_lv_gb_grasp = self.create_publisher(PointStamped, "grasp", qos_profile)

        ts = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(self, CompressedImage, params["refimg_topic"], qos_profile=qos_profile),
                message_filters.Subscriber(self, Image, params["pclimg_topic"], qos_profile=qos_profile)
            ],
            queue_size=params["queue_size"], slop=params["slop"]
        )
        ts.registerCallback(self.callback)

    def load_params(self, params):
        self.conf_thr = params['conf_thr']
        self.cnt_area_thr = params['cnt_area_thr']
        self.dist_upp_bnd_2d = params['dist_upp_bnd_2d']
        self.num_cnt_ratio = params['num_cnt_ratio']
        self.num_skel_ratio = params['num_skel_ratio']
        self.skel_ang_thr = params['skel_ang_thr']
        self.window_size = params["window_size"]
        self.mad_thr = params["mad_thr"]
        self.adj_cnt_seg_thr = params["adj_cnt_seg_thr"]

    def callback(self, img_msg, pclimg_msg):
        start = time.time()

        # Convert the subscribed img msgs
        img = self.br.compressed_imgmsg_to_cv2(img_msg)
        pclimg = self.br.imgmsg_to_cv2(pclimg_msg)
        if img is None or pclimg is None:
            return

        # Initial Detection
        seg_scores, seg_labels, seg_cnts = yolo_utils.get_segs_2d(
            img, self.seg_predictor, self.conf_thr
        )
        if "liver" not in seg_labels or "gallbladder" not in seg_labels:
            return
        
        # Some post processing of the detected masks
        liver_idx = seg_labels.index("liver")
        gallb_idx = seg_labels.index("gallbladder")
        gallb_cnt = yolo_utils.process_gallb_cnt(
            seg_cnts[gallb_idx],
            img.shape[:2],
            self.cnt_area_thr
        )
        adj_cnt, new_gallb_cnts, new_gallb_mask = seg_utils.gen_gallb_mask(
            seg_scores[gallb_idx],
            gallb_cnt,
            seg_scores[liver_idx],
            seg_cnts[liver_idx],
            img.shape[:2],
            self.dist_upp_bnd_2d
        )
        if new_gallb_mask is None:
            return
        
        # Adjacent 3D contour points
        adj_cnt = misc_utils.farthest_first_traversal(adj_cnt, int(adj_cnt.shape[0]*self.num_cnt_ratio))
        adj_cnt_3d = pcl_utils.win_avg_3d_pts(
            adj_cnt, pclimg, self.window_size, self.mad_thr
        )
        if adj_cnt_3d is not None:
            self.pub_lv_gb_adj_cnts.publish(
                pcl_utils.pts3d_to_pcl2(
                    adj_cnt_3d,
                    pclimg_msg.header.frame_id,
                    ros2_utils.now(self),
                    color = (0, 255, 255)
                )
            )

        # Extract the Contour Points
        new_gallb_cnts = misc_utils.farthest_first_traversal(new_gallb_cnts, int(new_gallb_cnts.shape[0]*self.num_cnt_ratio))
        new_gallb_cnts = misc_utils.smooth_cnt(new_gallb_cnts, 0.1)

        # Publish contour points (3D)
        cnt_3d = pcl_utils.win_avg_3d_pts(
            new_gallb_cnts, pclimg, self.window_size, self.mad_thr
        )
        if cnt_3d is not None:
            self.pub_lv_gb_cnts.publish(
                pcl_utils.pts3d_to_pcl2(
                    cnt_3d,
                    pclimg_msg.header.frame_id,
                    ros2_utils.now(self),
                    color = (0, 0, 255)
                )
            )
        
        # Extract the Skeleton
        skel_inds, ctrd = cv_utils.mask_skeleton(new_gallb_mask, 0.7)
        skel_inds = cv_utils.prune_skeleton(skel_inds, self.skel_ang_thr)
        skel_inds = misc_utils.farthest_first_traversal(skel_inds, int(skel_inds.shape[0]*self.num_skel_ratio))

        # Publish gallbladder skeleton (3D)
        skel_3d = pcl_utils.win_avg_3d_pts(
            skel_inds, pclimg, self.window_size, self.mad_thr
        )
        if skel_3d is not None:
            self.pub_lv_gb_skel.publish(
                pcl_utils.pts3d_to_pcl2(
                    skel_3d,
                    pclimg_msg.header.frame_id,
                    ros2_utils.now(self),
                    color = (0, 255, 0)
                )
            )
        
        # Publish gallbladder centroid (3D)
        ctrd = pcl_utils.win_avg_3d_pt(
            ctrd, pclimg, self.window_size, self.mad_thr
        )
        if ctrd is not None:
            self.pub_lv_gb_ctrd.publish(
                tf_utils.pt3d2ptstamped(
                    ctrd,
                    ros2_utils.now(self),
                    pclimg_msg.header.frame_id
                )
            )

        # Publish Boundary points (3D)
        if adj_cnt_3d is None:
            return
        adj_cnt_3d_segments = misc_utils.find_segments(adj_cnt_3d, self.adj_cnt_seg_thr)
        bnd_3d = seg_utils.right_bottom_segment(adj_cnt_3d_segments)
        if bnd_3d is not None or bnd_3d.size != 0:
            self.pub_lv_gb_bnd.publish(
                pcl_utils.pts3d_to_pcl2(
                    bnd_3d,
                    pclimg_msg.header.frame_id,
                    ros2_utils.now(self),
                    color = (255,0,0)
                )
            )

        # Publish gallbladder grasping point (3D)
        if skel_3d is None or bnd_3d is None:
            return
        grasp_3d = (skel_3d.mean(axis=0) + bnd_3d.mean(axis=0)) / 2
        if grasp_3d is not None:
            self.pub_lv_gb_grasp.publish(
                tf_utils.pt3d2ptstamped(
                    grasp_3d,
                    ros2_utils.now(self),
                    pclimg_msg.header.frame_id
                )
            )