# Import ROS libraries
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import message_filters

# Import open source libraries
import numpy as np

# Import custom libraries
from sitl_ros2_interfaces.msg import SegStamped
from utils import misc_utils, pcl_utils, ros2_utils
from utils import seg_utils, tf_utils, cv_utils, ma_utils

class PUB_SEG_LV_GB_3D(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        self.br = CvBridge()
        self.load_params(params)
        self.pcl_img = None
        qos_profile = ros2_utils.custom_qos_profile(params["queue_size"])
        
        self.pub_lv_cnt     = self.create_publisher(PointCloud2,  'liver/cnts_3d',    qos_profile)
        self.pub_gb_cnt     = self.create_publisher(PointCloud2,  'gallb/cnts_3d',    qos_profile)
        self.pub_lv_bed_cnt = self.create_publisher(PointCloud2,  'liverbed/cnts_3d', qos_profile)
        self.pub_gb_skel    = self.create_publisher(PointCloud2,  'gallb/skel_3d',    qos_profile)
        self.pub_gb_ctrd    = self.create_publisher(PointStamped, 'gallb/ctrd_3d',    qos_profile)

        self.sub_lv_cnt = self.create_subscription(
            Image, params["pclimg_topic"], self.pclimg_cb, qos_profile
        )
        self.sub_lv_cnt = self.create_subscription(
            SegStamped, 'liver/cnts_2d', self.liver_cb, qos_profile
        )
        self.sub_gb_cnt = self.create_subscription(
            SegStamped, 'gallb/cnts_2d', self.gallb_cb, qos_profile
        )
        self.sub_lv_bed_cnt = self.create_subscription(
            SegStamped, 'liverbed/cnts_2d', self.liver_bed_cb, qos_profile
        )

    def load_params(self, params):
        self.downsample_pixel_dist = params['downsample_pixel_dist']
        self.window_size = params["window_size"]
        self.mad_thr = params["mad_thr"]

    def pclimg_cb(self, pclimg_msg):
        self.pcl_img = self.br.imgmsg_to_cv2(pclimg_msg)
        self.frame_id = pclimg_msg.header.frame_id

    def gallb_cb(self, gb_cnt_msg):
        if self.pcl_img is None:
            return
        pcl_img = np.copy(self.pcl_img)
        gallb_cnt = ma_utils.ma2arr(gb_cnt_msg.cnt2d)
        gallb_cnt_3d = pcl_utils.win_avg_3d_pts(
            gallb_cnt, pcl_img, self.window_size, self.mad_thr
        )
        if gallb_cnt_3d is None:
            return
        self.pub_gb_cnt.publish(
            pcl_utils.pts3d_to_pcl2(
                gallb_cnt_3d,
                self.frame_id,
                ros2_utils.now(self),
                color = (0, 255, 0)
            )
        )
        self.pub_ctrd_3d(gallb_cnt, pcl_img, self.frame_id)
        self.pub_skel_3d(gallb_cnt, pcl_img, self.frame_id)
        
    def liver_cb(self, lv_cnt_msg):
        if self.pcl_img is None:
            return
        liver_cnt = ma_utils.ma2arr(lv_cnt_msg.cnt2d)
        liver_cnt_3d = pcl_utils.win_avg_3d_pts(
            liver_cnt, self.pcl_img, self.window_size, self.mad_thr
        )
        if liver_cnt_3d is None:
            return
        self.pub_lv_cnt.publish(
            pcl_utils.pts3d_to_pcl2(
                liver_cnt_3d,
                self.frame_id,
                ros2_utils.now(self),
                color = (0, 0, 255)
            )
        )

    def liver_bed_cb(self, lv_bed_cnt_msg):
        if self.pcl_img is None:
            return
        liver_bed_cnt = ma_utils.ma2arr(lv_bed_cnt_msg.cnt2d)
        liver_bed_cnt_3d = pcl_utils.win_avg_3d_pts(
            liver_bed_cnt, self.pcl_img, self.window_size, self.mad_thr
        )
        if liver_bed_cnt_3d is None:
            return
        self.pub_lv_bed_cnt.publish(
            pcl_utils.pts3d_to_pcl2(
                liver_bed_cnt_3d,
                self.frame_id,
                ros2_utils.now(self),
                color = (0, 0, 255)
            )
        )

    def pub_ctrd_3d(self, gallb_cnt, pcl_img, frame_id):
        ctrd = cv_utils.cnt_centroid(np.int32(gallb_cnt))
        ctrd_3d = pcl_utils.win_avg_3d_pt(
            ctrd, pcl_img, self.window_size, self.mad_thr
        )
        if ctrd_3d is None:
            return
        self.pub_gb_ctrd.publish(
            tf_utils.pt3d2ptstamped(
                ctrd_3d,
                ros2_utils.now(self),
                frame_id
            )
        )

    def pub_skel_3d(self, gallb_cnt, pcl_img, frame_id):
        skel = misc_utils.farthest_first_traversal(
            cv_utils.mask_skeleton(
                cv_utils.mask_convex_hull(
                    seg_utils.get_cnt_mask(gallb_cnt, pcl_img.shape[:2])
                )
                # seg_utils.get_cnt_mask(
                #     gallb_cnt, pcl_img.shape[:2]
                # )
            ),
            self.downsample_pixel_dist
        )
        skel_3d = pcl_utils.win_avg_3d_pts(
            skel, pcl_img, self.window_size, self.mad_thr
        )
        if skel_3d is None:
            return
        self.pub_gb_skel.publish(
            pcl_utils.pts3d_to_pcl2(
                skel_3d,
                frame_id,
                ros2_utils.now(self),
                color = (255, 255, 0)
            )
        )

        

        
