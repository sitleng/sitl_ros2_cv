# SITL ROS2 CV - Computer Vision for da Vinci Surgical Robotics

A ROS2 package for real-time computer vision on da Vinci stereo endoscopic cameras, developed by the Surgical Innovation and Training Lab at the University of Illinois Chicago.

## Overview

This package provides computer vision capabilities for surgical robotics applications, specifically designed for da Vinci surgical systems. It includes stereo vision processing, anatomical structure segmentation, surgical instrument detection, and 3D reconstruction.

## Features

### Vision Capabilities
- Stereo vision processing with real-time disparity mapping
- Anatomical structure segmentation (liver, gallbladder)
- Surgical instrument keypoint detection (Fenestrated Bipolar Forceps, Permanent Cautery Hook)
- 3D point cloud generation from stereo cameras

### Supported AI Models
- YOLO (YOLOv11) for object detection and segmentation
- Detectron2 for instance segmentation
- MaskDINO for advanced segmentation tasks

### Camera Systems
- ECM (Endoscopic Camera Manipulator) stereo pipeline
- Recorded stereo video processing
- Live camera streaming with calibration and rectification

## Installation

### Prerequisites
```bash
# ROS2 Humble
sudo apt install ros-humble-desktop ros-humble-cv-bridge ros-humble-image-transport

# Python dependencies
pip install opencv-python scipy scikit-learn numpy torch torchvision
pip install ultralytics
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

### Build Package
```bash
cd ~/ros2_ws/src
git clone https://github.com/sitleng/sitl_ros2_cv.git
cd ~/ros2_ws
colcon build --packages-select sitl_ros2_cv
source install/setup.bash
```

## Usage

### Stereo ECM System
```bash
# Basic stereo camera processing
ros2 launch sitl_ros2_cv stereo_ecm_base.xml

# Stereo system with 3D point cloud generation
ros2 launch sitl_ros2_cv stereo_ecm_pcl.xml
```

### Anatomical Structure Segmentation
```bash
# Liver-gallbladder segmentation using YOLO
ros2 launch sitl_ros2_cv yolo_seg_lv_gb.xml

# Liver-gallbladder segmentation using Detectron2
ros2 launch sitl_ros2_cv dt2_seg_lv_gb.xml
```

### Surgical Instrument Keypoint Detection
```bash
# Prograsp Forceps keypoint detection
ros2 launch sitl_ros2_cv yolo_kpt_fbf.xml
ros2 launch sitl_ros2_cv dt2_kpt_fbf.xml

# Patch Grasper keypoint detection
ros2 launch sitl_ros2_cv yolo_kpt_pch.xml
ros2 launch sitl_ros2_cv dt2_kpt_pch.xml
```

### Video Playback System
```bash
# Process recorded stereo videos
ros2 launch sitl_ros2_cv stereo_video_base.xml

# Video processing with 3D reconstruction
ros2 launch sitl_ros2_cv stereo_video_pcl.xml
```

### Data Recording
```bash
# Record stereo camera data
ros2 launch sitl_ros2_cv dvrk_record.xml
```

## Architecture

### Core Modules

#### Camera Management
- `sitl_ros2_cv/ecm/`: ECM-specific implementations for da Vinci cameras
- `sitl_ros2_cv/video/`: Video file processing and playback nodes
- `nodes/camera/`: Camera interfacing, calibration, rectification, disparity calculation, and 3D reconstruction

#### Detection and Segmentation
- `sitl_ros2_cv/detect/`: High-level detection pipelines with post-processing
- `sitl_ros2_cv/dt2/` and `nodes/dt2/`: Detectron2-based segmentation
- `sitl_ros2_cv/yolo/` and `nodes/yolo/`: YOLO-based detection and segmentation

#### Utilities
- `utils/pcl_utils.py`: Point cloud processing and CUDA acceleration
- `utils/cv_cuda_utils.py`: CUDA-accelerated computer vision operations
- `utils/seg_utils.py`: Segmentation post-processing and contour analysis
- `utils/kpt_utils.py`: Keypoint detection and tracking
- `utils/tf_utils.py`: 3D transformations and coordinate conversions
- `utils/ecm_utils.py`: ECM camera-specific utilities
- `utils/misc_utils.py`: General utilities

### Key Algorithms

#### Stereo 3D Reconstruction
```python
# CUDA-accelerated stereo matching with WLS filtering
disp = cv_cuda_utils.cuda_sgm_wls_filter(cam1_sgm, cam2_sgm, left_img, right_img, wls_filter)

# Convert disparity to 3D point cloud
pclimg = pcl_utils.disp2pclimg_cuda(disp, Q, depth_scale, pcl_scale, depth_trunc)
```

#### Segmentation Post-Processing
```python
# Outlier removal and contour smoothing
clean_cnt = seg_utils.rm_cnt_outliers_pca(cnt, pca_var_ratio=0.95)
smooth_cnt = misc_utils.smooth_cnt(clean_cnt, win_r=0.1)

# 3D contour projection
cnt_3d = seg_utils.cnt_2d_to_3d(smooth_cnt, pclimg)
```

## ROS2 Messages and Data Formats

### ROS2 Messages
- `sensor_msgs/Image`: Raw and processed camera images
- `sensor_msgs/CompressedImage`: Compressed video streams
- `sensor_msgs/PointCloud2`: 3D point clouds from stereo reconstruction
- `geometry_msgs/PointStamped`: 3D keypoint locations
- `sitl_ros2_interfaces/SegStamped`: Segmentation masks with timestamps
- `sitl_ros2_interfaces/Dt2KptState`: Keypoint detection results

### Supported Formats
- Images: PNG, JPG, BMP
- Videos: MP4, AVI for stereo pairs
- Point clouds: PCL-compatible formats
- Models: PyTorch (.pt), Detectron2 (.pth)

## Configuration

### Camera Calibration
Place stereo calibration files in the home directory:
```
~/ecm_si_calib_data/
├── left_camera_matrix.yaml
├── right_camera_matrix.yaml
├── stereo_params.yaml
└── rectification_maps/
    ├── left_rectification_map.yaml
    └── right_rectification_map.yaml
```

### Model Configuration
Update model paths in launch files:
```python
# YOLO models
"model_path": "/home/"+os.getlogin()+"/yolo_models/liver_gb_best.pt"
"fbf_model_path": "/home/"+os.getlogin()+"/yolo_models/fbf_keypoints_best.pt"

# Detectron2 models
"model_weights": "/home/"+os.getlogin()+"/dt2_models/liver_model_final.pth"
"model_config": "/home/"+os.getlogin()+"/dt2_models/liver_config.yaml"
```

### Performance Parameters
Adjust parameters in configuration files:
```yaml
# Point cloud parameters
depth_scale: 1000.0
pcl_scale: 1.0
depth_trunc: 2000.0

# Detection thresholds
detection_confidence: 0.5
nms_threshold: 0.4
```

## Applications

### Surgical Automation
- Tissue boundary detection for automated organ identification
- Instrument tracking for real-time surgical tool pose estimation
- Depth perception for 3D spatial awareness in robotic manipulation

### Research and Development
- Algorithm validation and benchmarking
- Annotated surgical dataset generation
- Performance analysis of detection and segmentation methods

## About

**Surgical Innovation and Training Lab**  
University of Illinois Chicago  
Department of Surgery

Research focused on advancing surgical robotics through computer vision and machine learning techniques.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@INPROCEEDINGS{koh2024autodissect,
  author={Oh, Ki-Hwan and Borgioli, Leonardo and Žefran, Miloš and Chen, Liaohai and Giulianotti, Pier Cristoforo},
  booktitle={2024 10th IEEE RAS/EMBS International Conference for Biomedical Robotics and Biomechatronics (BioRob)}, 
  title={A Framework for Automated Dissection Along Tissue Boundary}, 
  year={2024},
  pages={1427-1433},
  keywords={Three-dimensional displays;Medical robotics;Automation;Tracking;Endoscopes;Instruments;Surgery;Liver;Gallbladder;Real-time systems},
  doi={10.1109/BioRob60516.2024.10719948}
}
```

## Disclaimer

This software is for research purposes only and is not intended for clinical use without proper validation and regulatory approval.
