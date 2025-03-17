# Import ROS libraries
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

# Import open source libraries

# Import custom libraries
from sitl_ros2_interfaces.msg import SegStamped
from utils import misc_utils, ros2_utils, cv_utils
from utils import seg_utils, yolo_utils

class PUB_SEG_LV_GB_RAW(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        self.br = CvBridge()
        self.load_params(params)
        self.seg_predictor = yolo_utils.load_model(params['model_path'])

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
        self.conf_thr = params['conf_thr']
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
        seg_scores, seg_labels, seg_masks = yolo_utils.get_segs_2d(
            img, self.seg_predictor, self.conf_thr
        )
        if seg_labels is None:
            return
        if "liver" not in seg_labels or "gallbladder" not in seg_labels:
            return
        if "liver bed" in seg_labels:
            seg_masks[seg_labels.index('liver')] = cv_utils.merge_mask(
                seg_masks[seg_labels.index('liver')],
                seg_masks[seg_labels.index('liver bed')]
            )
        # Some post processing of the detected masks
        seg_cnts = yolo_utils.process_masks(
            seg_masks,
            self.cnt_area_thr
        )
        self.pub_gallb_cnt(seg_cnts[seg_labels.index("gallbladder")])
        self.pub_liver_cnt(seg_cnts[seg_labels.index("liver")])
        if "liver bed" in seg_labels:
            self.pub_liver_bed_cnt(seg_cnts[seg_labels.index("liver bed")])
            