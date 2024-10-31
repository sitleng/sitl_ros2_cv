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
from utils import seg_utils, dt2_utils #, mdino_utils

class PUB_SEG_LV_GB(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        self.br = CvBridge()
        self.load_params(params)
        if self.model_type == "Detectron2":
            self.seg_predictor = dt2_utils.load_seg_predictor(
                params['model_weights'], self.pred_score_thr
            )
        # elif self.model_type == "MaskDINO":
        #     self.seg_predictor = mdino_utils.load_seg_predictor(
        #         params['config_file'], params['model_weights']
        #     )
        self.seg_metadata = seg_utils.load_seg_metadata()

        qos_profile = ros2_utils.custom_qos_profile(params["queue_size"])
        self.pub_lv_gb_cnts  = self.create_publisher(PointCloud2 , "cnts" , qos_profile)
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
        self.depth_scale = params["depth_scale"]
        self.gray_mode = True if "gray" in params["model_weights"] else False
        self.model_type = params['model_type']
        self.pred_score_thr = params['pred_score_thr']
        self.cnt_area_thr = params['cnt_area_thr']
        self.dist_upp_bnd_2d = params['dist_upp_bnd_2d']
        self.num_cnt_ratio = params['num_cnt_ratio']
        self.num_skel_ratio = params['num_skel_ratio']
        self.skel_ang_thr = params['skel_ang_thr']
        self.window_size = params["window_size"]
        self.mad_thr = params["mad_thr"]
        self.dist_upp_bnd_3d = params["dist_upp_bnd_3d"]

    def callback(self, img_msg, pclimg_msg):
        start = time.time()

        # Convert the subscribed img msgs
        img = self.br.compressed_imgmsg_to_cv2(img_msg)
        pclimg = self.br.imgmsg_to_cv2(pclimg_msg)/self.depth_scale
        if img is None or pclimg is None:
            return
        if self.gray_mode:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Initial Detection
        if self.model_type == "Detectron2":
            _, seg_labels, _, seg_masks = dt2_utils.run_dt2_seg(
                img, self.seg_metadata, self.seg_predictor
            )
        # elif self.model_type == "MaskDINO":
        #     _, seg_labels, _, seg_masks = mdino_utils.mask_dino_seg(
        #         frame, self.seg_predictor, self.seg_metadata, self.pred_score_thr
        #     )
        if "Liver" not in seg_labels:
            return
        
        # Remove the liver from the image to enhance the gallbladder detection
        # The model tries to predict with the background removed image
        liver_mask = cv_utils.mask_convex_hull(seg_masks[seg_labels.index("Liver")])
        bg_rm_img = cv2.bitwise_and(
            img, cv2.cvtColor(liver_mask, cv2.COLOR_GRAY2BGR)
        )
        if self.model_type == "Detectron2":
            seg_probs, seg_labels, _, seg_masks = dt2_utils.run_dt2_seg(
                bg_rm_img, self.seg_metadata, self.seg_predictor
            )
        # elif self.model_type == "MaskDINO":
        #     seg_probs, seg_labels, _, seg_masks = mdino_utils.mask_dino_seg(
        #         bg_rm_img, self.seg_predictor, self.seg_metadata, self.pred_score_thr
        #     )
        if "Gallbladder" not in seg_labels:
            return
        
        # Some post processing of the detected masks
        liver_idx = seg_labels.index("Liver")
        gallb_idx = seg_labels.index("Gallbladder")
        gallb_mask = seg_utils.gen_gallb_mask(
            seg_utils.seg_score(seg_probs[gallb_idx]),
            seg_masks[gallb_idx],
            seg_utils.seg_score(seg_probs[liver_idx]),
            seg_masks[liver_idx],
            self.cnt_area_thr,
            self.dist_upp_bnd_2d
        )
        if gallb_mask is None:
            return

        # Extract the Contour Points
        gallb_cnts = seg_utils.get_obj_seg(gallb_mask, self.cnt_area_thr)
        if gallb_cnts is None:
            return
        gallb_cnts = misc_utils.farthest_first_traversal(gallb_cnts, int(gallb_cnts.shape[0]*self.num_cnt_ratio))
        gallb_cnts = misc_utils.smooth_cnt(gallb_cnts, 0.1)

        # Publish contour points (3D)
        cnt_3d = pcl_utils.win_avg_3d_pts(
            gallb_cnts, pclimg, self.window_size, self.mad_thr
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
        skel_inds, ctrd = cv_utils.mask_skeleton(gallb_mask, 0.7)
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
        if cnt_3d is None or skel_3d is None or skel_3d.size == 0:
            return
        nn_inds = misc_utils.nn_kdtree(cnt_3d, skel_3d, self.dist_upp_bnd_3d)[-1]
        nn_inds = misc_utils.rm_outlier_mad(np.unique(nn_inds), thresh=2)
        if nn_inds.size < 3:
            return
        nn_inds = np.arange(nn_inds[0], nn_inds[-1]+1)
        bnd_3d = cnt_3d[nn_inds]
        bnd_3d = bnd_3d[np.argsort(bnd_3d[:,0])[::-1]]
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
        grasp_3d = (skel_3d.mean(axis=0) + bnd_3d.mean(axis=0)) / 2
        if grasp_3d is not None:
            self.pub_lv_gb_grasp.publish(
                tf_utils.pt3d2ptstamped(
                    grasp_3d,
                    ros2_utils.now(self),
                    pclimg_msg.header.frame_id
                )
            )