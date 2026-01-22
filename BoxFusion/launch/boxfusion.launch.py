#!/usr/bin/env python
__author__ = "Joris Verhagen"
__contact__ = "jorisv@kth.se"

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from scipy.spatial.transform import Rotation as R
import numpy as np

def generate_launch_description():
    # launch rviz2 with a specific config file
    rviz_config_file = os.path.join(
        '/home/none/manipulation_ws/src/BoxFusion/BoxFusion/config/realsense_rviz.rviz'
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    # # # tf static transform for azure kinect on the panda arm
    # tf_broadcaster_node_0 = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='static_transform_publisher_align_frame_8',
    #     arguments=['0.0', '0.0', '0.0', '0.0', str(np.deg2rad(-90)), str(np.deg2rad(180-45)), 'panda/panda_link8', 'link8_aligned'],
    #     output='screen'
    # )

    # tf_broadcaster_node_2 = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='static_transform_publisher_frame_8_to_kinect_base',
    #     arguments=['0.035', '0.0', '0.11', str(np.deg2rad(2)), str(np.deg2rad(-17)), '0.0', 'link8_aligned', 'camera_base'],
    #     output='screen'
    # )

    # read ~/c_space_results/eye_on_hand_calibration.json
    # load T_ee_cam from the json file and use it as static transform from
    # panda_link8 to camera_base
    with open(os.path.expanduser('~/c_space_stl_results/eye_on_hand_calibration.json'), 'r') as f:
        import json
        calib_data = json.load(f)
        R_cam2ee = calib_data['R_cam2ee']  # 3x3 list
        t_cam2ee = calib_data['t_cam2ee']  # list of 3
    R_cam2ee = np.array(R_cam2ee)
    t_cam2ee = np.array(t_cam2ee)
    # extract translation and euler angles
    q = R.from_matrix(R_cam2ee).as_quat(scalar_first=False)#('xyz', degrees=False)
    # some additional rotations
    R_new = R.from_quat(q) * R.from_euler('x', np.pi/2)
    q = R_new.as_quat(scalar_first=False)
    R_new = R.from_quat(q) * R.from_euler('z', np.pi/2)
    q = R_new.as_quat(scalar_first=False)
    
    tf_broadcaster_node_1 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_align_frame_8',
        arguments=[str(t_cam2ee[0]), str(t_cam2ee[1]), str(t_cam2ee[2]),
                   str(q[0]), str(q[1]), str(q[2]), str(q[3]),
                   'panda/panda_link8', 'camera_base'],
        output='screen'
    )

    # static transform because boxfusion has different camera frame convention than rviz/azure kinect
    # -90 around x, then 90 around z
    tf_broadcaster_node_3 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_kinect_adjustment',
        arguments=['0', '0', '0', '-1.57079632679', '0', '-1.57079632679', 'camera_base', 'camera_base_boxfusion'],
        output='screen'
    )

    # ros2 launch azure_kinect_ros_driver driver.launch.py
    kinect_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('azure_kinect_ros_driver'),
                'launch',
                'driver.launch.py'
            )
        ),
        launch_arguments={'fps': '15', 
                          'rgb_point_cloud': 'true',
                          'point_cloud_in_depth_frame': 'true',
                          }.items()
    )

    return LaunchDescription([
        rviz_node,
        tf_broadcaster_node_1,
        tf_broadcaster_node_3,
        kinect_launch,
    ])