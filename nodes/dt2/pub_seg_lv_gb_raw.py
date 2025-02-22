# Import ROS libraries
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

# Import custom libraries
from sitl_ros2_interfaces.msg import SegStamped
from utils import misc_utils, ros2_utils
from utils import seg_utils, dt2_utils, mdino_utils

class PUB_SEG_LV_GB_RAW(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
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

        qos_profile = ros2_utils.custom_qos_profile(params["queue_size"])

        self.pub_lv_cnt     = self.create_publisher(SegStamped, 'liver/cnts_2d', qos_profile)
        self.pub_gb_cnt     = self.create_publisher(SegStamped, 'gallb/cnts_2d', qos_profile)
        self.pub_lv_bed_cnt = self.create_publisher(SegStamped, 'liverbed/cnts_2d', qos_profile)

        self.sub_img = self.create_subscription(
            CompressedImage,
            params["refimg_topic"],
            self.callback,
            qos_profile
        )

    def load_params(self, params):
        self.model_type = params['model_type']
        self.pred_score_thr = params['pred_score_thr']
        self.cnt_area_thr = params['cnt_area_thr']
        self.downsample_pixel_dist = params['downsample_pixel_dist']

    def pub_gallb_cnt(self, gallb_cnt):
        if gallb_cnt is None:
            return
        # gallb_cnt = seg_utils.get_obj_seg(
        #     cv_utils.mask_convex_hull(
        #         seg_utils.get_cnt_mask(seg_cnts[gallb_idx], img.shape[:2])
        #     ),
        #     self.cnt_area_thr
        # )
        gallb_cnt = misc_utils.farthest_first_traversal(
            gallb_cnt, self.downsample_pixel_dist
        )
        gallb_cnt = seg_utils.align_cnt(gallb_cnt)
        self.pub_gb_cnt.publish(
            seg_utils.gen_segstamped(
                ["gallbladder"],
                gallb_cnt,
                ros2_utils.now(self)
            )
        )

    def pub_liver_cnt(self, liver_cnt):
        if liver_cnt is None:
            return
        liver_cnt = misc_utils.farthest_first_traversal(
            liver_cnt, self.downsample_pixel_dist
        )
        self.pub_lv_cnt.publish(
            seg_utils.gen_segstamped(
                ["liver"],
                liver_cnt,
                ros2_utils.now(self)
            )
        )

    def pub_liver_bed_cnt(self, liver_bed_cnt):
        if liver_bed_cnt is None:
            return
        liver_bed_cnt = misc_utils.farthest_first_traversal(
            liver_bed_cnt, self.downsample_pixel_dist
        )
        self.pub_lv_bed_cnt.publish(
            seg_utils.gen_segstamped(
                ["liver bed"],
                liver_bed_cnt,
                ros2_utils.now(self)
            )
        )

    def callback(self, img_msg):
        # Convert the subscribed img msgs
        img = self.br.compressed_imgmsg_to_cv2(img_msg)
        if img is None:
            return

        # Initial Detection
        if self.model_type == "Detectron2":
            seg_probs, seg_labels, _, seg_masks = dt2_utils.run_dt2_seg(
                img, self.seg_metadata, self.seg_predictor
            )
        elif self.model_type == "MaskDINO":
            seg_probs, seg_labels, _, seg_masks = mdino_utils.mask_dino_seg(
                img, self.seg_predictor, self.seg_metadata, self.pred_score_thr
            )
        if "Liver" not in seg_labels or "Gallbladder" not in seg_labels:
            return
        
        # Some post processing of the detected masks
        gallb_cnt = seg_utils.get_obj_seg(seg_masks[seg_labels.index("Gallbladder")], self.cnt_area_thr)
        self.pub_gallb_cnt(gallb_cnt)
        liver_cnt = seg_utils.get_obj_seg(seg_masks[seg_labels.index("Liver")], self.cnt_area_thr)
        self.pub_liver_cnt(liver_cnt)
        if "LiverBed" in seg_labels:
            liver_bed_cnt = seg_utils.get_obj_seg(seg_masks[seg_labels.index("LiverBed")], self.cnt_area_thr)
            self.pub_liver_bed_cnt(liver_bed_cnt)
