#!/usr/bin/env python

import cv2
import rospy
import tf
from geometry_msgs.msg import Vector3, PointStamped
import numpy as np
import math
import open3d as o3d

import time
from utils_display import DisplayCamera
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Quaternion
from sitl_dvrk.msg import param_3D
from utils import tf_utils
from utils import dvrk_utils
import tf_conversions.posemath as pm
from scipy.linalg import expm
from filterpy.kalman import KalmanFilter

import ctypes
import time
import xr
from pprint import pprint


class publisher_controller():

    def __init__(self):
        self.pub_R = rospy.Publisher('controller/2/R', Quaternion, queue_size=10)
        self.pub_T = rospy.Publisher('controller/2/T', Vector3, queue_size=10)
        
        
    
if __name__ == '__main__':

        pub=publisher_controller()
        rospy.init_node('pub_controller', anonymous=True)
        rate = rospy.Rate(30)  # 10 Hz
        try:
            while not rospy.is_shutdown():
                # ContextObject is a high level pythonic class meant to keep simple cases simple.
                with xr.ContextObject(
                    instance_create_info=xr.InstanceCreateInfo(
                        enabled_extension_names=[
                            # A graphics extension is mandatory (without a headless extension)
                            xr.KHR_OPENGL_ENABLE_EXTENSION_NAME,
                        ],
                    ),
                ) as context:
                    # Set up the controller pose action
                    controller_paths = (xr.Path * 2)(
                        xr.string_to_path(context.instance, "/user/hand/left"),
                        xr.string_to_path(context.instance, "/user/hand/right"),
                    )
                    controller_pose_action = xr.create_action(
                        action_set=context.default_action_set,
                        create_info=xr.ActionCreateInfo(
                            action_type=xr.ActionType.POSE_INPUT,
                            action_name="hand_pose",
                            localized_action_name="Hand Pose",
                            count_subaction_paths=len(controller_paths),
                            subaction_paths=controller_paths,
                        ),
                    )
                    suggested_bindings = (xr.ActionSuggestedBinding * 2)(
                        xr.ActionSuggestedBinding(
                            action=controller_pose_action,
                            binding=xr.string_to_path(
                                instance=context.instance,
                                path_string="/user/hand/left/input/grip/pose",
                            ),
                        ),
                        xr.ActionSuggestedBinding(
                            action=controller_pose_action,
                            binding=xr.string_to_path(
                                instance=context.instance,
                                path_string="/user/hand/right/input/grip/pose",
                            ),
                        ),
                    )
                    xr.suggest_interaction_profile_bindings(
                        instance=context.instance,
                        suggested_bindings=xr.InteractionProfileSuggestedBinding(
                            interaction_profile=xr.string_to_path(
                                context.instance,
                                "/interaction_profiles/khr/simple_controller",
                            ),
                            count_suggested_bindings=len(suggested_bindings),
                            suggested_bindings=suggested_bindings,
                        ),
                    )
                    xr.suggest_interaction_profile_bindings(
                        instance=context.instance,
                        suggested_bindings=xr.InteractionProfileSuggestedBinding(
                            interaction_profile=xr.string_to_path(
                                context.instance,
                                "/interaction_profiles/htc/vive_controller",
                            ),
                            count_suggested_bindings=len(suggested_bindings),
                            suggested_bindings=suggested_bindings,
                        ),
                    )

                    action_spaces = [
                        xr.create_action_space(
                            session=context.session,
                            create_info=xr.ActionSpaceCreateInfo(
                                action=controller_pose_action,
                                subaction_path=controller_paths[0],
                            ),
                        ),
                        xr.create_action_space(
                            session=context.session,
                            create_info=xr.ActionSpaceCreateInfo(
                                action=controller_pose_action,
                                subaction_path=controller_paths[1],
                            ),
                        ),
                    ]

                    # Loop over the render frames
                    for frame_index, frame_state in enumerate(context.frame_loop()):

                        if context.session_state == xr.SessionState.FOCUSED:
                            active_action_set = xr.ActiveActionSet(
                                action_set=context.default_action_set,
                                subaction_path=xr.NULL_PATH,
                            )
                            xr.sync_actions(
                                session=context.session,
                                sync_info=xr.ActionsSyncInfo(
                                    count_active_action_sets=1,
                                    active_action_sets=ctypes.pointer(active_action_set),
                                ),
                            )
                            found_count = 0
                            for index, space in enumerate(action_spaces):
                                space_location = xr.locate_space(
                                    space=space,
                                    base_space=context.space,
                                    time=frame_state.predicted_display_time,
                                )
                                if space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:

                                    pub.pub_R.publish(space_location.pose.orientation.x,
                                                        space_location.pose.orientation.y,
                                                        space_location.pose.orientation.z,
                                                        space_location.pose.orientation.w)
                                    
                                    pub.pub_T.publish(space_location.pose.position.x,
                                                      space_location.pose.position.y,
                                                      space_location.pose.position.z)
                                    

                                    print(space_location.pose)
                                    found_count += 1
                            if found_count == 0:
                                print("no controllers active")

                        rate.sleep()
    
        except rospy.ROSInterruptException:
            pass

        finally:   
           del pub
                    