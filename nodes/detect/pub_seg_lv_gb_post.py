# Import ROS libraries
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import message_filters

# Import custom libraries
from utils import pcl_utils, ros2_utils
from utils import seg_utils, misc_utils

class PUB_SEG_LV_GB_POST(Node):
    def __init__(self, params):
        super().__init__(params["node_name"])
        self.load_params(params)

        qos_profile = ros2_utils.custom_qos_profile(params["queue_size"])

        self.pub_lv_gb_adj   = self.create_publisher(PointCloud2,  "adj_3d",   qos_profile)
        self.pub_lv_gb_bnd   = self.create_publisher(PointCloud2,  "bnd_3d",   qos_profile)

        ts = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(self, PointCloud2,  'liver/cnts_3d', qos_profile=qos_profile),
                message_filters.Subscriber(self, PointCloud2,  'gallb/cnts_3d', qos_profile=qos_profile),
                message_filters.Subscriber(self, PointCloud2,  'gallb/skel_3d', qos_profile=qos_profile),
            ],
            queue_size=params["queue_size"], slop=params["slop"]
        )
        ts.registerCallback(self.callback)

    def load_params(self, params):
        self.adj_dub = params['adj_dub']
        self.adj_segs_thr = params["adj_segs_thr"]
        self.frame_id = params['frame_id']

    def callback(self, lv_msg, gb_msg, skel_msg):
        lv_cnt_3d = pcl_utils.pcl2nparray(lv_msg)
        gb_cnt_3d = pcl_utils.pcl2nparray(gb_msg)
        gb_skel_3d = pcl_utils.pcl2nparray(skel_msg)

        adj_3d = seg_utils.adj_cnts_3d(lv_cnt_3d, gb_cnt_3d, self.adj_dub)
        # ros2_utils.loginfo(self, f'Hellooooo')
        if adj_3d.shape[0] < 5:
            return
        self.pub_lv_gb_adj.publish(
            pcl_utils.pts3d_to_pcl2(
                adj_3d,
                self.frame_id,
                ros2_utils.now(self),
                color = (0, 255, 255)
            )
        )

        bnd_3d = seg_utils.get_bnd(gb_skel_3d, adj_3d)
        # bnd_3d = misc_utils.smooth_cnt(bnd_3d, 0.2)
        # bnd_3d = misc_utils.filter_bnd_3d(bnd_3d)
        # bnd_3d = misc_utils.filt_bnd_pts(bnd_3d)
        # bnd_3d = misc_utils.interp_3d(bnd_3d, is_closed=False)
        # ros2_utils.loginfo(self, f'bnd_3d shape: {bnd_3d.shape}')
        if bnd_3d is None or bnd_3d.shape[0] < 5:
            return
        self.pub_lv_gb_bnd.publish(
            pcl_utils.pts3d_to_pcl2(
                bnd_3d,
                self.frame_id,
                ros2_utils.now(self),
                color = (255,0,255)
            )
        )

