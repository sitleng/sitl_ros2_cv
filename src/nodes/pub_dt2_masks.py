#!/usr/bin/env python3

# Import open source libraries
import traceback
import numpy as np

# Import ROS libraries
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

# Import custom libraries
from utils import dt2_utils, cv_utils

class PUB_DT2_MASKS():
    def __init__(self,params):
        self.br = CvBridge()
        self.img = None
        self.cur_traj_pt = None
        self.reach_traj_flag = False
        self.kernel = np.ones((9,9),np.uint8)
        self.pub_masks = rospy.Publisher("/dt2/masks", CompressedImage, queue_size=10)
        self.obj_type = params["obj_type"]
        self.cnt_area_thr = params["cnt_area_thr"]
        self.seg_predictor = dt2_utils.load_seg_predictor(
            params['seg_model_path'], params['seg_score_thresh']
        )
        self.seg_metadata = dt2_utils.load_seg_metadata()
        self.kpt_predictor = dt2_utils.load_pch_predictor(
            params['kpt_model_path'], params['kpt_score_thresh']
        )
        self.kpt_metadata = dt2_utils.load_pch_metadata()
        self.kpt_nms = self.kpt_metadata.get("keypoint_names")

    def __del__(self):
        print("Destructing class PUB_DT2_BNDPTS...")

    def callback(self,cam1_rect_color_msg):
        self.img = self.br.compressed_imgmsg_to_cv2(cam1_rect_color_msg)

    def run(self):
        while not rospy.is_shutdown():
            img = self.img
            if img is not None:
                traj_pts = None
                final_traj = None
                seg_img = img.copy()
                # In case you need glare removal
                # img = cv_utils.remove_glare(img, 245, self.kernel, 3, 9)

                dt2_labels, dt2_masks = dt2_utils.run_dt2_seg(img, self.seg_metadata, self.seg_predictor)
                dt2_kpt_boxes, dt2_kpt_classes, _, dt2_kpts = dt2_utils.extract_kpt_res(
                    img, self.kpt_predictor, self.kpt_metadata
                )
                if dt2_kpts.size == 0:
                    continue

                obj1_idx, obj2_idx = dt2_utils.extract_obj_inds(self.obj_type, dt2_labels)
                # masks = dt2_utils.extract_seg_masks(self.cnt_area_thr, dt2_masks)
                # mask_img = dt2_utils.gen_mask_img(img, obj1_idx, obj2_idx, masks)
                # seg_img = dt2_utils.gen_seg_img(img, mask_img, alpha=0.3)
                
                cnts = dt2_utils.extract_full_seg(self.cnt_area_thr, dt2_masks)
                if obj1_idx is not None and obj2_idx is not None:
                    adj_obj1_obj2 = dt2_utils.extract_adj_pts(self.img, obj1_idx, obj2_idx, cnts)
                    traj_pts = dt2_utils.extract_traj(adj_obj1_obj2, None)
                    if traj_pts is not None and len(traj_pts) > 50:
                        traj_cnds = dt2_utils.split_traj(traj_pts, step=10, min_len=50)
                        th_pt = dt2_kpts[0][self.kpt_nms.index("TipHook")][:2]
                        final_traj = dt2_utils.get_adj_traj(traj_cnds, th_pt)
                # seg_img = cv_utils.scatter(seg_img, traj_pts, 5, (255,0,0))

                if final_traj is not None:
                    if self.cur_traj_pt is None:
                        self.cur_traj_pt = final_traj[0]
                    else:
                        self.cur_traj_pt = dt2_utils.get_cur_traj_pt(self.cur_traj_pt, final_traj, dt2_kpts, self.kpt_nms)
                    seg_img = cv_utils.scatter(img, final_traj, 5, (255,0,0))
                    if self.cur_traj_pt is not None:
                        seg_img = cv_utils.scatter(seg_img, self.cur_traj_pt, 10, (0,0,255))

                box_c = self.kpt_metadata.get("thing_colors")[dt2_kpt_classes[0]]
                seg_img = dt2_utils.draw_and_connect_keypoints(seg_img, self.kpt_metadata, dt2_kpts[0], dt2_kpt_boxes[0], box_c, 0.1)

                out_img_msg = self.br.cv2_to_compressed_imgmsg(seg_img)
                self.pub_masks.publish(out_img_msg)

if __name__ == '__main__':
    rospy.init_node("pub_dt2_bndpts")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)
    try:
        app = PUB_DT2_MASKS(params)
        rospy.loginfo("Start detecting boundary points... ")
        # if params["calib_dir"] == "L2R":
        #     rospy.Subscriber("/ecm/left_rect/image_color", CompressedImage, app.callback)
        # elif params["calib_dir"] == "R2L":
        #     rospy.Subscriber("/ecm/right_rect/image_color", CompressedImage, app.callback)
        rospy.Subscriber("/loop_video/frame", CompressedImage, app.callback)
        app.run()
    except:
        traceback.print_exc()
    finally:
        del app