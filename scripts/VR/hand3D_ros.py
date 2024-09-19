#!/usr/bin/env python3
import rospy
import cv2
from utils_mediapipe import MediaPipeHand
from utils_display import DisplayHand
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Float64
from sensor_msgs.msg import PointCloud2
from ros_numpy.point_cloud2 import array_to_pointcloud2
from sitl_dvrk.msg import param_3D
import geometry_msgs.msg  
from utils import pcl_utils

class VR_HAND_3D():

    def __init__(self):

        self.pub_left = rospy.Publisher('hand_3D/left', param_3D, queue_size=10)
        self.pub_left_pcl = rospy.Publisher('hand_3D/left_pcl', PointCloud2, queue_size=10)
        self.pub_right = rospy.Publisher('hand_3D/right', param_3D, queue_size=10)
        self.angle_norm = 0
        self.image = None
        self.param = None

        self.img_width = 1920
        self.img_height = 1080

        self.media_pipe_hand_msg = param_3D()
        self.br = CvBridge()

        self.intrin = {
            'fx': self.img_width * 0.9,
            'fy': self.img_width * 0.9,
            'cx': self.img_width * 0.5,
            'cy': self.img_height * 0.5,
            'width': self.img_width,
            'height': self.img_height,
        }

        self.pipe = MediaPipeHand(static_image_mode=True, max_num_hands=2, intrin=self.intrin)
        self.disp = DisplayHand(draw3d=True, draw_camera=True, max_num_hands=2, intrin=self.intrin)

    def callback(self, imgmsg):

        self.image = self.br.compressed_imgmsg_to_cv2(imgmsg)

    

    def publish_param(self, param):
        

        media_pipe_hand_msg_left = param_3D()
        media_pipe_hand_msg_right = param_3D()

        #media_pipe_hand_msg_left.hand_side = param[0]['class']
        media_pipe_hand_msg_left.score = [param[0]['score']]
        media_pipe_hand_msg_left.keypt = [geometry_msgs.msg.Point(x=param[0]['keypt'][i,0], y=param[0]['keypt'][i, 1], z=0) for i in range(21)]
        media_pipe_hand_msg_left.joint = [geometry_msgs.msg.Point(x=param[0]['joint'][i, 0], y=param[0]['joint'][i, 1], z=param[0]['joint'][i, 2]) for i in range(21)]
        self.pub_left.publish(media_pipe_hand_msg_left)
 

        #media_pipe_hand_msg_right.hand_side = param[1]['class']
        media_pipe_hand_msg_right.score = [param[1]['score']]
        media_pipe_hand_msg_right.keypt = [geometry_msgs.msg.Point(x=param[1]['keypt'][i,0], y=param[1]['keypt'][i, 1], z=0) for i in range(21)]
        media_pipe_hand_msg_right.joint = [geometry_msgs.msg.Point(x=param[1]['joint'][i, 0], y=param[1]['joint'][i, 1], z=param[1]['joint'][i, 2]) for i in range(21)]
        self.pub_right.publish(media_pipe_hand_msg_right)

    def publish_pcl(self, param):
        pcl_msg = array_to_pointcloud2(pcl_utils.pts_array_3d_to_pcl(param[0]['joint']),frame_id="camera")
        self.pub_left_pcl.publish(pcl_msg)

if __name__ == '__main__':

    rospy.init_node('hand_gesture_node', anonymous=True)
    hand_app = VR_HAND_3D()
    
   
    try:
        rate = rospy.Rate(30)  # 10 Hz
        rospy.Subscriber("/vr/left/image_color", CompressedImage, hand_app.callback)

        while not rospy.is_shutdown():

            if hand_app.image is not None:

                hand_app.image.flags.writeable = False
                param = hand_app.pipe.forward(hand_app.image)
                hand_app.image.flags.writeable = True
                hand_app.disp.draw3d(param, hand_app.image)

                hand_app.disp.vis.update_geometry(None)
                hand_app.disp.vis.poll_events()
                hand_app.disp.vis.update_renderer()
                
                hand_app.publish_param(param)  # Call the function to publish the parameter data
                hand_app.publish_pcl(param)
            
                if cv2.waitKey(1) & 0xFF == ord("q"):  # Break the loop on 'Esc' key
                    break
                rate.sleep()




    except rospy.ROSInterruptException:
        pass

    finally:
        cv2.destroyAllWindows()
