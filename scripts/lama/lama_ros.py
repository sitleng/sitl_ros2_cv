#!/usr/bin/env python3

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers
from saicinpainting.evaluation.data import pad_tensor_to_modulo

LOGGER = logging.getLogger(__name__)

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

class LAMA_ROS(object):
    def __init__(self):
        self.br       = CvBridge()
        self.img      = None
        self.mask     = None
        self.kernel   = np.ones((11,11),np.uint8)
        self.lama_pub = rospy.Publisher("/lama/frame", CompressedImage, queue_size=10)

    def callback(self, color_msg):
        self.img = self.br.compressed_imgmsg_to_cv2(color_msg)
        gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)[1]
        self.mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel, iterations=3)
        # self.mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, self.kernel, iterations=1)

@hydra.main(config_path='/home/leo/lama/configs/prediction', config_name='custom.yaml')
def main(predict_config: OmegaConf):
    app = LAMA_ROS()
    mod = 8
    try:
        rospy.Subscriber("/loop_video/frame",CompressedImage,app.callback)
        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log
        # Prep LaMa
        device = torch.device(predict_config.device)
        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        checkpoint_path = os.path.join(predict_config.model.path, 
                                        'models',
                                        predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)
        while not rospy.is_shutdown():
            if app.img is not None and app.mask is not None:
                img  = app.img
                mask = app.mask
                assert len(mask.shape) == 2
                if np.max(mask) == 1:
                    mask = mask * 255
                img = torch.from_numpy(img).float().div(255.)
                mask = torch.from_numpy(mask).float()
                batch = {}
                batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
                batch['mask'] = mask[None, None]
                unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
                batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
                batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
                batch = move_to_device(batch, device)
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = model(batch)
                cur_res = batch["inpainted"][0].permute(1, 2, 0)
                cur_res = cur_res.detach().cpu().numpy()
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]
                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                msg = app.br.cv2_to_compressed_imgmsg(cur_res)
                msg.header.stamp = rospy.Time.now()
                app.lama_pub.publish(msg)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)

if __name__ == '__main__':
    rospy.init_node("lama_ros")
    main()
