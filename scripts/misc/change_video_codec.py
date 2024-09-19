#!/usr/bin/env python3

import cv2
import argparse
from utils import ecm_utils
from tqdm import tqdm

class CHANGE_VIDEO_CODEC():
    def __init__(self,args):
        fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
        self.orig_video = cv2.VideoCapture(args.video_path)
        fps    = self.orig_video.get(cv2.CAP_PROP_FPS)
        width  = int(self.orig_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.orig_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.orig_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pbar = tqdm(total = total_frames)
        self.out_video  = cv2.VideoWriter(
            self.get_out_video_path(args.video_path),
            cv2.CAP_FFMPEG,
            fourcc, fps, (width,height), True
        )
    
    def get_out_video_path(self,orig_video_path):
        out_video_path = orig_video_path.split("/")
        temp = out_video_path[-1].split(".")
        temp[0] += "_copy"
        temp[1] = "mp4"
        out_video_path[-1] = ".".join(temp)
        out_video_path = "/".join(out_video_path)
        return out_video_path

    def run(self):
        while self.orig_video.isOpened():
            ret, frame = self.orig_video.read() 
            if ret:
                self.out_video.write(frame)
                self.pbar.update(1)
        self.orig_video.release()
        self.out_video.release()
        self.pbar.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Change Video Codec',
        description='Capture Images for Endoscope Calibration',
        epilog='Text at the bottom of help')
    parser.add_argument(
        '-vp', '--video_path', type=str, 
        help='Path to the video...'
    )
    parser.add_argument(
        '-f', '--fourcc', type=str, 
        help='Desired video codec: e.g. MJPG'
    )
    args = parser.parse_args()
    app = CHANGE_VIDEO_CODEC(args)
    try:
        app.run()
    except Exception as e:
        print(e)