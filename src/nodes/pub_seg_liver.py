#!/usr/bin/env python3

# Import open source libraries
import cv2
import numpy as np
import traceback

# Import ROS libraries
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
import message_filters
from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2

# Import custom libraries
# from sitl_dvrk.msg import UInt16MultiArrayStamped
from utils import dt2_utils, tf_utils, cv_utils, misc_utils, mdino_utils, pcl_utils

class PUB_SEG_LIVER_2D():
    def __init__(self,params):
        self.br = CvBridge()
        self.load_params(params)
        if self.model_type == "Detectron2":
            self.seg_predictor = dt2_utils.load_seg_predictor(
                params['model_weights'], self.pred_score_thr
            )
        elif self.model_type == "MaskDINO":
            self.seg_predictor = mdino_utils.load_seg_predictor(
                params['config_file'], params['model_weights']
            )
        self.seg_metadata = dt2_utils.load_seg_metadata()
        self.pub_bnd_3d = rospy.Publisher("/seg/liver_gallb/bnd_3d", PointCloud2, queue_size=1)
        self.pub_gallb_cnt_3d = rospy.Publisher("/seg/gallbladder/cnt_3d", PointCloud2, queue_size=1)
        self.pub_gallb_skel_3d = rospy.Publisher("/seg/gallbladder/skel_3d", PointCloud2, queue_size=1)
        self.pub_gallb_ctrd_3d = rospy.Publisher("/seg/gallbladder/ctrd_3d", PointStamped, queue_size=1)
        self.pub_gallb_grasp_3d = rospy.Publisher("/seg/gallbladder/grasp_3d", PointStamped, queue_size=1)
        self.img = None
        self.pcl = None

    def __del__(self):
        print("Destructing class PUB_DT2_LIVER_MASK...")

    def load_params(self, params):
        self.gray_mode = True if "gray" in params["model_weights"] else False
        self.model_type = params['model_type']
        self.pred_score_thr = params['pred_score_thr']
        self.obj_type = params["obj_type"]
        self.cnt_area_thr = params['cnt_area_thr']
        self.dist_upp_bnd_2d = params['dist_upp_bnd_2d']
        self.num_cnt_ratio = params['num_cnt_ratio']
        self.num_skel_ratio = params['num_skel_ratio']
        self.skel_ang_thr = params['skel_ang_thr']
        self.window_size = params["window_size"]
        self.mad_thr = params["mad_thr"]
        self.dist_upp_bnd_3d = params["dist_upp_bnd_3d"]

    def run(self):
        try:
            img = self.img
            pcl = self.pcl
            if img is None or pcl is None:
                return
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
                gallb_score, gallb_mask, liver_score, liver_mask, self.cnt_area_thr, self.dist_upp_bnd_2d
            )
            gallb_cnts = dt2_utils.get_obj_seg(gallb_mask, self.cnt_area_thr)
            if gallb_cnts is None:
                return
            gallb_cnts = misc_utils.farthest_first_traversal(gallb_cnts, int(gallb_cnts.shape[0]*self.num_cnt_ratio))
            gallb_cnts = misc_utils.smooth_cnt(gallb_cnts, 0.1)
            # Publish contour points (3D)
            cnt_3d = pcl_utils.win_avg_3d_pts(
                gallb_cnts, pcl, self.window_size, self.mad_thr
            )
            if cnt_3d is not None:
                self.pub_gallb_cnt_3d.publish(
                    array_to_pointcloud2(
                        pcl_utils.pts_array_3d_to_pcl(
                            cnt_3d,
                            color = (0, 0, 255)
                        ),
                        frame_id = self.frame_id,
                        stamp = rospy.Time.now()
                    )
                )
            skel_inds, ctrd = cv_utils.mask_skeleton(gallb_mask, 0.7)
            skel_inds = cv_utils.prune_skeleton(skel_inds, self.skel_ang_thr)
            skel_inds = misc_utils.farthest_first_traversal(skel_inds, int(skel_inds.shape[0]*self.num_skel_ratio))
            # Publish gallbladder skeleton (3D)
            gallb_skel_3d = pcl_utils.win_avg_3d_pts(
                skel_inds, pcl, self.window_size, self.mad_thr
            )
            if gallb_skel_3d is not None:
                self.pub_gallb_skel_3d.publish(
                    array_to_pointcloud2(
                        pcl_utils.pts_array_3d_to_pcl(
                            gallb_skel_3d,
                            color = (0, 255, 0)
                        ),
                        frame_id = self.frame_id,
                        stamp = rospy.Time.now()
                    )
                )
            # Publish gallbladder centroid (3D)
            gallb_c_3d = pcl_utils.win_avg_3d_pt(
                ctrd, pcl, self.window_size, self.mad_thr
            )
            if gallb_c_3d is not None:
                self.pub_gallb_ctrd_3d.publish(
                    tf_utils.pt3d2ptstamped(
                        gallb_c_3d, rospy.Time.now(), self.frame_id
                    )
                )

            # Publish Boundary points (3D)
            if cnt_3d is None or gallb_skel_3d is None:
                return
            nn_inds = misc_utils.nn_kdtree(cnt_3d, gallb_skel_3d, self.dist_upp_bnd_3d)[-1]
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
                        frame_id = self.frame_id,
                        stamp = rospy.Time.now()
                    )
                )

            # Publish gallbladder grasping point (3D)
            grasp_3d = (gallb_skel_3d.mean(axis=0) + bnd_3d.mean(axis=0)) / 2
            if gallb_c_3d is not None:
                self.pub_gallb_grasp_3d.publish(
                    tf_utils.pt3d2ptstamped(
                        grasp_3d, rospy.Time.now(), self.frame_id
                    )
                )
        except:
            traceback.print_exc()
            print(self.img.shape)

    def callback(self, cam1_rect_msg, pcl_msg):
        self.img = self.br.compressed_imgmsg_to_cv2(cam1_rect_msg)
        if self.gray_mode:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        self.pcl = pcl_utils.xyzarr_to_nparr(pointcloud2_to_array(pcl_msg))
        self.frame_id = pcl_msg.header.frame_id
            

if __name__ == '__main__':
    rospy.init_node("pub_dt2_seg_liver_2d")

    node_name = rospy.get_name()

    params = rospy.get_param(node_name)
    try:
        app = PUB_SEG_LIVER_2D(params)
        rospy.loginfo("Start segmenting liver... ")
        if params["calib_dir"] == "L2R":
            p1 = "left"
        else:
            p1 = "right"
        if "gray" in params["model_weights"]:
            p2 = "mono"
        else:
            p2 = "color"
        topic_list = []
        topic_list.append(message_filters.Subscriber("/ecm/{}_rect/image_{}".format(p1, p2), CompressedImage))
        topic_list.append(message_filters.Subscriber("/ecm/points2", PointCloud2))
        ts = message_filters.ApproximateTimeSynchronizer(
            topic_list, slop=0.1, queue_size=1
        )
        ts.registerCallback(app.callback)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            app.run()
            rate.sleep()
    except Exception as e:
        rospy.logerr(e)
    finally:
        del app