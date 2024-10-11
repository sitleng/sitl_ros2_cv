import os
from glob import glob
from setuptools import setup

package_name = 'sitl_ros2_cv'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name, 'utils', 'nodes'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.xml')),
    ],
    install_requires=[
        'setuptools',
        'sitl_dvrk_ros2_interfaces'
    ],
    zip_safe=True,
    maintainer='koh',
    maintainer_email='sktlgt93@gmail.com',
    description='SITL ROS2 Package for Image Processing and Computer Vision Applications',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            # ecm
            'pub_ecm_left_info  = sitl_ros2_cv.ecm.pub_ecm_left_info:main',
            'pub_ecm_left_raw   = sitl_ros2_cv.ecm.pub_ecm_left_raw:main',
            'pub_ecm_left_rect  = sitl_ros2_cv.ecm.pub_ecm_left_rect:main',
            'pub_ecm_right_info = sitl_ros2_cv.ecm.pub_ecm_right_info:main',
            'pub_ecm_right_raw  = sitl_ros2_cv.ecm.pub_ecm_right_raw:main',
            'pub_ecm_right_rect = sitl_ros2_cv.ecm.pub_ecm_right_rect:main',
            'pub_ecm_disp       = sitl_ros2_cv.ecm.pub_ecm_disp:main',
            'pub_ecm_pclimg     = sitl_ros2_cv.ecm.pub_ecm_pclimg:main',
            'pub_ecm_pcl        = sitl_ros2_cv.ecm.pub_ecm_pcl:main',
            'pub_ecm_recon3d    = sitl_ros2_cv.ecm.pub_ecm_recon3d:main',
            # video
            'pub_video_left     = sitl_ros2_cv.video.pub_video_left:main',
            'pub_video_right    = sitl_ros2_cv.video.pub_video_right:main',
            'pub_video_disp     = sitl_ros2_cv.video.pub_video_disp:main',
            'pub_video_pclimg   = sitl_ros2_cv.video.pub_video_pclimg:main',
            'pub_video_pcl      = sitl_ros2_cv.video.pub_video_pcl:main',
            # detection models
            'pub_seg_lv_gb      = sitl_ros2_cv.detect.pub_seg_lv_gb:main',
            'pub_kpt_raw_fbf    = sitl_ros2_cv.detect.pub_kpt_raw_fbf:main',
            'pub_kpt_raw_pch    = sitl_ros2_cv.detect.pub_kpt_raw_pch:main',
            'pub_kpt_cp_fbf     = sitl_ros2_cv.detect.pub_kpt_cp_fbf:main',
            'pub_kpt_cp_pch     = sitl_ros2_cv.detect.pub_kpt_cp_pch:main',
            # misc
            'sub_custom_msg     = sitl_ros2_cv.misc.sub_custom_msg:main',
            'image_view         = sitl_ros2_cv.misc.image_view:main',
        ],
    },
)
