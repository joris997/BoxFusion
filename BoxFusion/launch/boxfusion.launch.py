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

    # # launch tf
    # # static transform to align world to panda base
    # tf_broadcaster_node = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='static_transform_publisher_world_to_panda',
    #     arguments=['0', '0', '0', '-0.825', '-1.57079632679', '0', 'panda/panda_link8', 'camera_panda'],
    #     output='screen'
    # )
    # # static transform because boxfusion has different camera frame convention than rviz/realsense
    # # -90 around x, then 90 around z
    # tf_broadcaster_node_2 = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='static_transform_publisher_camera_adjustment',
    #     arguments=['0', '0', '0', '-1.57079632679', '0', '-1.37079632679', 'camera_panda', 'camera_panda_boxfusion'],
    #     output='screen'
    # )

    # # ros2 launch realsense2_camera rs_launch.py publish_tf:=true
    # realsense_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(
    #             get_package_share_directory('realsense2_camera'),
    #             'launch',
    #             'rs_launch.py'
    #         )
    #     ),
    #     launch_arguments={'publish_tf': 'true',
    #                       'camera_frame_id': 'panda/panda_link8',  # make camera child of panda link
    #                       'base_frame_id': 'panda'
    #                       }.items()
    # )

    # # tf static transfrom for azure kinect on the tripod
    # tf_broadcaster_node = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='static_transform_publisher_world_to_kinect',
    #     arguments=['1.25', '0.08', '0.67', '3.14', '0.78', '0', 'panda/panda_link0', 'camera_base'],
    #     output='screen'
    # )
    # tf static transform for azure kinect on the panda arm
    tf_broadcaster_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_world_to_kinect',
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0', 'panda/panda_link8', 'camera_base'],
        output='screen'
    )

    # static transform because boxfusion has different camera frame convention than rviz/azure kinect
    # -90 around x, then 90 around z
    tf_broadcaster_node_2 = Node(
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
        launch_arguments={'rgb_point_cloud': 'true',
                          'point_cloud_in_depth_frame': 'true',
                          }.items()
    )

    return LaunchDescription([
        rviz_node,
        tf_broadcaster_node,
        tf_broadcaster_node_2,
        # realsense_launch,
        kinect_launch,
    ])