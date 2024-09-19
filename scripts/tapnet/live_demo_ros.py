#!/usr/bin/env python3

# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Live Demo for Online TAPIR with ROS.

import jax
import jax.numpy as jnp

import functools
import time

import cv2
import haiku as hk

import numpy as np
from tapnet import tapir_model

import traceback

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PointStamped

from sitl_dvrk.msg import BoolStamped
from utils import tapnet_utils

NUM_POINTS = 8

# --------------------
# Load checkpoint and initialize
class TAPIR_ROS_DEMO(object):
    def __init__(self):
        rospy.loginfo(jax.devices())
        rospy.loginfo("Welcome to the TAPIR live demo.")
        rospy.loginfo("Please note that if the framerate is low (<construct_initial_causal_state~12 fps), TAPIR performance")
        rospy.loginfo("may degrade and you may need a more powerful GPU.")
        rospy.loginfo("Loading checkpoint...")
        self.params, self.state = tapnet_utils.load_checkpoint(
            "tapnet/checkpoints/causal_tapir_checkpoint.npy"
        )
        self.frame = None
        # self.dsize = (256, 256)
        self.dsize = (480, 480)
        self.pos = tuple()
        self.query_frame = True
        self.online_init_apply, self.online_predict_apply = self.init_model()
        self.have_point = [False] * NUM_POINTS
        self.next_query_idx = 0
        self.query_points, self.query_features, self.causal_state = self.compile_jax_fns()
        self.t = time.time()
        self.step_counter = 0
    
    def __del__(self):
        print("Shutting Down TAPIR_ROS_DEMO...")

    def init_model(self):
        rospy.loginfo("Creating model...")
        online_init = hk.transform_with_state(tapnet_utils.build_online_model_init)
        online_init_apply = jax.jit(online_init.apply)

        online_predict = hk.transform_with_state(tapnet_utils.build_online_model_predict)
        online_predict_apply = jax.jit(online_predict.apply)

        rng = jax.random.PRNGKey(42)
        online_init_apply = functools.partial(
            online_init_apply, params=self.params, state=self.state, rng=rng
        )
        online_predict_apply = functools.partial(
            online_predict_apply, params=self.params, state=self.state, rng=rng
        )
        return online_init_apply, online_predict_apply

    def compile_jax_fns(self):
        self.frame = tapnet_utils.get_frame_ros(
            # rospy.wait_for_message("/ecm/left_rect/image_color", CompressedImage),
            # rospy.wait_for_message("/loop_video/frame", CompressedImage),
            rospy.wait_for_message("/dt2/masks", CompressedImage),
            # rospy.wait_for_message("/lama/frame", CompressedImage),
            self.dsize
        )
        rospy.loginfo("Compiling jax functions (this may take a while...)")
        # --------------------
        # Call one time to compile
        query_points = jnp.zeros([NUM_POINTS, 3], dtype=jnp.float32)
        query_features, _ = self.online_init_apply(
            frames=tapnet_utils.preprocess_frames(self.frame[None, None]),
            points=query_points[None, 0:1],
        )
        jax.block_until_ready(query_features)

        query_features, _ = self.online_init_apply(
            frames=tapnet_utils.preprocess_frames(self.frame[None, None]),
            points=query_points[None],
        )
        causal_state = tapnet_utils.construct_initial_causal_state(
            NUM_POINTS, len(query_features.resolutions) - 1
        )
        (prediction, causal_state), _ = self.online_predict_apply(
            frames=tapnet_utils.preprocess_frames(self.frame[None, None]),
            features=query_features,
            causal_context=causal_state,
        )
        jax.block_until_ready(prediction["tracks"])
        rospy.loginfo("Compiling jax functions Finished!")
        return query_points, query_features, causal_state
    
    def upd(self, s1, s2):
        return s1.at[:, self.next_query_idx : self.next_query_idx + 1].set(s2)

    def pos_cb(self, pos_msg):
        self.pos = (pos_msg.point.y, pos_msg.point.x)

    def qf_cb(self, qf_msg):
        self.query_frame = qf_msg.data
    
    def frame_cb(self, frame_msg):
        self.frame = tapnet_utils.get_frame_ros(frame_msg, self.dsize)
        if self.frame is not None:
            if self.query_frame:
                self.query_points = jnp.array((0,) + self.pos, dtype=jnp.float32)
                init_query_features, _ = self.online_init_apply(
                    frames=tapnet_utils.preprocess_frames(self.frame[None, None]),
                    points=self.query_points[None, None],
                )
                init_causal_state = tapnet_utils.construct_initial_causal_state(
                    1, len(self.query_features.resolutions) - 1
                )

                # cv2.circle(frame, (pos[0], pos[1]), 5, (255,0,0), -1)
                self.query_frame = False

                self.causal_state = jax.tree_map(self.upd, self.causal_state, init_causal_state)
                self.query_features = tapir_model.QueryFeatures(
                    lowres=jax.tree_map(
                        self.upd, self.query_features.lowres, init_query_features.lowres
                    ),
                    hires=jax.tree_map(
                        self.upd, self.query_features.hires, init_query_features.hires
                    ),
                    resolutions=self.query_features.resolutions,
                )
                self.have_point[self.next_query_idx] = True
                self.next_query_idx = (self.next_query_idx + 1) % NUM_POINTS
            if self.pos:
                (prediction, self.causal_state), _ = self.online_predict_apply(
                    frames=tapnet_utils.preprocess_frames(self.frame[None, None]),
                    features=self.query_features,
                    causal_context=self.causal_state,
                )
                track = prediction["tracks"][0, :, 0]
                occlusion = prediction["occlusion"][0, :, 0]
                expected_dist = prediction["expected_dist"][0, :, 0]
                visibles = tapnet_utils.postprocess_occlusions(occlusion, expected_dist)
                track = np.round(track)
                for i in range(len(self.have_point)):
                    if visibles[i] and self.have_point[i]:
                        cv2.circle(
                            self.frame, (int(track[i, 0]), int(track[i, 1])), 5, (255, 0, 0), -1
                        )
                        # if track[i, 0] < 16 and track[i, 1] < 16:
                        #     rospy.loginfo((i, self.next_query_idx))
                self.step_counter += 1
                if time.time() - self.t > 5:
                    rospy.loginfo(f"{self.step_counter/(time.time()-self.t)} frames per second")
                    self.t = time.time()
                    self.step_counter = 0
            else:
                self.t = time.time()
            cv2.imshow("Point Tracking", app.frame)
            cv2.waitKey(1)

if __name__ == "__main__":
    rospy.init_node("tapir_ros_demo")
    try:
        app = TAPIR_ROS_DEMO()
        rospy.Subscriber("/tapnet/pos",                PointStamped,    app.pos_cb  )
        rospy.Subscriber("/tapnet/query_frame",        BoolStamped,     app.qf_cb   )
        # rospy.Subscriber("/ecm/left_rect/image_color", CompressedImage, app.frame_cb)
        # rospy.Subscriber("/loop_video/frame",          CompressedImage, app.frame_cb)
        rospy.Subscriber("/dt2/masks",                 CompressedImage, app.frame_cb)
        # rospy.Subscriber("/lama/frame",          CompressedImage, app.frame_cb)
        rospy.loginfo_once("Press Ctrl + C to shutdown...")
        rospy.spin()
    except Exception:
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
