import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sitl_ros2_cv'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.xml')),
    ],
    install_requires=[
        'setuptools',
        'sitl_ros2_interfaces'
    ],
    zip_safe=True,
    maintainer='koh',
    maintainer_email='sktlgt93@gmail.com',
    description='SITL ROS2 Package for Image Processing and Computer Vision Applications',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            # ecm
            'pub_ecm_left_info    = sitl_ros2_cv.ecm.pub_ecm_left_info:main',
            'pub_ecm_left_raw     = sitl_ros2_cv.ecm.pub_ecm_left_raw:main',
            'pub_ecm_left_rect    = sitl_ros2_cv.ecm.pub_ecm_left_rect:main',
            'pub_ecm_right_info   = sitl_ros2_cv.ecm.pub_ecm_right_info:main',
            'pub_ecm_right_raw    = sitl_ros2_cv.ecm.pub_ecm_right_raw:main',
            'pub_ecm_right_rect   = sitl_ros2_cv.ecm.pub_ecm_right_rect:main',
            'pub_ecm_disp         = sitl_ros2_cv.ecm.pub_ecm_disp:main',
            'pub_ecm_pclimg       = sitl_ros2_cv.ecm.pub_ecm_pclimg:main',
            'pub_ecm_pcl          = sitl_ros2_cv.ecm.pub_ecm_pcl:main',
            'pub_ecm_recon3d      = sitl_ros2_cv.ecm.pub_ecm_recon3d:main',
            # video
            'pub_video_left       = sitl_ros2_cv.video.pub_video_left:main',
            'pub_video_right      = sitl_ros2_cv.video.pub_video_right:main',
            'pub_video_disp       = sitl_ros2_cv.video.pub_video_disp:main',
            'pub_video_pclimg     = sitl_ros2_cv.video.pub_video_pclimg:main',
            'pub_video_pcl        = sitl_ros2_cv.video.pub_video_pcl:main',
            # misc
            'sub_custom_msg       = sitl_ros2_cv.misc.sub_custom_msg:main',
            'image_view           = sitl_ros2_cv.misc.image_view:main',
            # dvrk record
            'rec_stereo_ecm       = sitl_ros2_cv.rec.rec_stereo_ecm:main',
            'rec_dvrk_kin         = sitl_ros2_cv.rec.rec_dvrk_kin:main',
            # detectron2
            'pub_seg_lv_gb_dt2    = sitl_ros2_cv.dt2.pub_seg_lv_gb:main',
            'pub_kpt_raw_fbf_dt2  = sitl_ros2_cv.dt2.pub_kpt_raw_fbf:main',
            'pub_kpt_raw_pch_dt2  = sitl_ros2_cv.dt2.pub_kpt_raw_pch:main',
            'pub_kpt_cp_fbf_dt2   = sitl_ros2_cv.dt2.pub_kpt_cp_fbf:main',
            'pub_kpt_cp_pch_dt2   = sitl_ros2_cv.dt2.pub_kpt_cp_pch:main',
            # YOLOv11
            'pub_kpt_raw_fbf_yolo = sitl_ros2_cv.yolo.pub_kpt_raw_fbf:main',
            'pub_kpt_raw_pch_yolo = sitl_ros2_cv.yolo.pub_kpt_raw_pch:main',
            'pub_kpt_cp_fbf_yolo  = sitl_ros2_cv.yolo.pub_kpt_cp_fbf:main',
            'pub_kpt_cp_pch_yolo  = sitl_ros2_cv.yolo.pub_kpt_cp_pch:main',
            'pub_seg_lv_gb_yolo   = sitl_ros2_cv.yolo.pub_seg_lv_gb:main',
        ],
    },
)
