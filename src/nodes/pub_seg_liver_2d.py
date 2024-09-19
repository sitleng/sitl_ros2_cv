#!/usr/bin/env python3

# Import open source libraries
import cv2
import traceback

# Import ROS libraries
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

# Import custom libraries
from sitl_dvrk.msg import UInt16MultiArrayStamped
from utils import dt2_utils, ma_utils, cv_utils, misc_utils, mdino_utils

class PUB_SEG_LIVER_2D():
    def __init__(self,params):
        self.br = CvBridge()
        self.load_params(params)
        self.pub_gallb_cnt_2d = rospy.Publisher("/seg/gallbladder/cnt_2d", UInt16MultiArrayStamped, queue_size=10)
        self.pub_gallb_skel_2d = rospy.Publisher("/seg/gallbladder/skel_2d", UInt16MultiArrayStamped, queue_size=10)
        self.pub_gallb_ctrd_2d = rospy.Publisher("/seg/gallbladder/ctrd_2d", UInt16MultiArrayStamped, queue_size=10)
        if self.model_type == "Detectron2":
            self.seg_predictor = dt2_utils.load_seg_predictor(
                params['model_weights'], self.pred_score_thr
            )
        elif self.model_type == "MaskDINO":
            self.seg_predictor = mdino_utils.load_seg_predictor(
                params['config_file'], params['model_weights']
            )
        self.seg_metadata = dt2_utils.load_seg_metadata()

    def __del__(self):
        print("Destructing class PUB_DT2_LIVER_MASK...")

    def load_params(self, params):
        self.gray_mode = True if "gray" in params["model_weights"] else False
        self.model_type = params['model_type']
        self.pred_score_thr = params['pred_score_thr']
        self.obj_type = params["obj_type"]
        self.cnt_area_thr = params['cnt_area_thr']
        self.adj_dist_upp_bnd = params['adj_dist_upp_bnd']
        self.num_cnt_ratio = params['num_cnt_ratio']
        self.num_skel_ratio = params['num_skel_ratio']
        self.skel_ang_thr = params['skel_ang_thr']

    def callback(self, cam1_rect_msg):
        try:
            img = self.br.compressed_imgmsg_to_cv2(cam1_rect_msg)
            if img is None:
                return
            if self.gray_mode:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if self.model_type == "Detectron2":
                _, seg_labels, _, seg_masks = dt2_utils.run_dt2_seg(
                    img, self.seg_metadata, self.seg_predictor
                )
            elif self.model_type == "MaskDINO":
                _, seg_labels, _, seg_masks = mdino_utils.mask_dino_seg(
                    img, self.seg_predictor, self.seg_metadata, self.pred_score_thr
                )
            if "Liver" not in seg_labels:
                return
            # Fill the hole due to the gallbladder
            liver_mask = cv_utils.mask_convex_hull(seg_masks[seg_labels.index("Liver")])
            bg_rm_img = cv2.bitwise_and(
                img, cv2.cvtColor(liver_mask, cv2.COLOR_GRAY2BGR)
            )
            if self.model_type == "Detectron2":
                seg_probs, seg_labels, _, seg_masks = dt2_utils.run_dt2_seg(
                    bg_rm_img, self.seg_metadata, self.seg_predictor
                )
            elif self.model_type == "MaskDINO":
                seg_probs, seg_labels, _, seg_masks = mdino_utils.mask_dino_seg(
                    bg_rm_img, self.seg_predictor, self.seg_metadata, self.pred_score_thr
                )
            if "Gallbladder" not in seg_labels:
                return
            liver_idx = seg_labels.index("Liver")
            gallb_idx = seg_labels.index("Gallbladder")
            liver_mask = seg_masks[liver_idx]
            gallb_mask = seg_masks[gallb_idx]
            liver_score = dt2_utils.seg_score(seg_probs[liver_idx])
            gallb_score = dt2_utils.seg_score(seg_probs[gallb_idx])
            gallb_mask = dt2_utils.gen_gallb_mask(
                gallb_score, gallb_mask, liver_score, liver_mask, self.cnt_area_thr, self.adj_dist_upp_bnd
            )
            gallb_cnts = dt2_utils.get_obj_seg(gallb_mask, self.cnt_area_thr)
            if gallb_cnts is None:
                return
            gallb_cnts = misc_utils.farthest_first_traversal(gallb_cnts, int(gallb_cnts.shape[0]*self.num_cnt_ratio))
            gallb_cnts = misc_utils.smooth_cnt(gallb_cnts, 0.1)
            self.pub_gallb_cnt_2d.publish(ma_utils.arr2uint16ma(gallb_cnts, rospy.Time.now()))
            skel_inds, ctrd = cv_utils.mask_skeleton(gallb_mask, 0.7)
            skel_inds = cv_utils.prune_skeleton(skel_inds, self.skel_ang_thr)
            skel_inds = misc_utils.farthest_first_traversal(skel_inds, int(skel_inds.shape[0]*self.num_skel_ratio))
            self.pub_gallb_skel_2d.publish(ma_utils.arr2uint16ma(skel_inds, rospy.Time.now()))
            self.pub_gallb_ctrd_2d.publish(ma_utils.arr2uint16ma(ctrd, rospy.Time.now()))
        except:
            traceback.print_exc()
            print(gallb_cnts)
            print(skel_inds)
            print(ctrd)

if __name__ == '__main__':
    rospy.init_node("pub_dt2_seg_liver_2d")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)
    try:
        app = PUB_SEG_LIVER_2D(params)
        rospy.loginfo("Start segmenting liver... ")
        if params["calib_dir"] == "L2R":
            if "gray" in params["model_weights"]:
                rospy.Subscriber("/ecm/left_rect/image_mono", CompressedImage, app.callback)
            else:
                rospy.Subscriber("/ecm/left_rect/image_color", CompressedImage, app.callback)
        elif params["calib_dir"] == "R2L":
            if "gray" in params["model_weights"]:
                rospy.Subscriber("/ecm/right_rect/image_mono", CompressedImage, app.callback)
            else:
                rospy.Subscriber("/ecm/right_rect/image_color", CompressedImage, app.callback)
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()
    except Exception as e:
        rospy.logerr(e)
    finally:
        del app