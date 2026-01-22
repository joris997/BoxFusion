# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, Bool
from sensor_msgs_py import point_cloud2
import os, sys
from rclpy import Parameter
import time
import threading

import yaml

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

from boxfusion.capture_stream import ROSDataset
from tests.open3d_example import read_trajectory


# read ~/c_space_stl_results/eye_on_hand_calibration.json
with open(os.path.expanduser('~/c_space_stl_results/eye_on_hand_calibration.json'), 'r') as f:
    import json 
    calib_data = json.load(f)
    camera_matrix = calib_data['camera_matrix']  # 3x3 list
print("Original camera matrix:", camera_matrix)
camera_matrix = np.array(camera_matrix)

# The camera matrix is for the original high-resolution image
# We need to scale it to match the 640x480 resolution used by ROSDataset
# Estimate original resolution from principal point (cx, cy should be ~half of width, height)
original_width = camera_matrix[0, 2] * 2  # cx * 2
original_height = camera_matrix[1, 2] * 2  # cy * 2
target_width = 640
target_height = 480

scale_x = target_width / original_width
scale_y = target_height / original_height

print(f"Original resolution estimate: {original_width:.0f}x{original_height:.0f}")
print(f"Target resolution: {target_width}x{target_height}")
print(f"Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")

# Scale the intrinsics
fx_scaled = camera_matrix[0, 0] * scale_x
fy_scaled = camera_matrix[1, 1] * scale_y
cx_scaled = camera_matrix[0, 2] * scale_x
cy_scaled = camera_matrix[1, 2] * scale_y

print(f"Scaled intrinsics: fx={fx_scaled:.2f}, fy={fy_scaled:.2f}, cx={cx_scaled:.2f}, cy={cy_scaled:.2f}")

camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsics.set_intrinsics(
    width=target_width,
    height=target_height,
    fx=fx_scaled,
    fy=fy_scaled,
    cx=cx_scaled,
    cy=cy_scaled,
)

# create ros dataset 
if not os.path.exists('./config/online.yaml'):
    raise ValueError("Missing config path")
else:
    with open('./config/online.yaml', 'r') as  f:
        cfg = yaml.full_load(f)
print(cfg)

class OnlineOpen3DNode(Node):
    def __init__(self):
        super().__init__('online_open3d_node',
                         parameter_overrides=[Parameter('use_sim_time',Parameter.Type.BOOL, True)])
        print("done init")
        self.cfg = cfg
        print("Creating ROSDataset for online Open3D integration")
        self.dataset = ROSDataset(cfg)
        self.dataset.load_arkit_depth = True

        # pointcloud ros2 publisher
        self.point_cloud_pub = self.create_publisher(
            PointCloud2,
            '/open3d/merged_point_cloud',
            10
        )

        # next subscriber
        self.next_sub = self.create_subscription(
            Bool,
            '/open3d/next_frame',
            self.next_frame_callback,
            10
        )
        self.cnt = 0
        self.next_cnt = 0
        self.start_time = time.time()
        self.max_time = 10000  # seconds

        # Define workspace bounds (meters)
        self.workspace_min = np.array([-0.5, 0.0, -0.1])
        self.workspace_max = np.array([0.5, 1.2, 0.5])

        # Calculate workspace center and dimensions
        self.workspace_center = (self.workspace_min + self.workspace_max) / 2.0
        workspace_size = self.workspace_max - self.workspace_min
        max_dim = np.max(workspace_size)

        self.T_restore = np.eye(4)
        self.T_restore[0:3, 0:3] = np.array([[0, 1, 0],      # swap x and y axis
                                             [1, 0, 0],
                                             [0, 0, 1]])
        self.T_restore[:3, 3] = self.workspace_center        # translate back to original center            
        
        print(f"Workspace center: {self.workspace_center}")
        print(f"Workspace dimensions: {workspace_size}")
        print(f"Max dimension: {max_dim:.3f}m")
        
        # Create TSDF volume centered at origin
        # Volume will be centered at workspace_center via pose transformation
        # Use smallest cube that fits the workspace
        volume_length = max_dim * 1.1  # 10% margin for safety
        voxel_size = volume_length / 512
        print(f"Volume length: {volume_length:.3f}m")
        print(f"Voxel size: {voxel_size*1000:.2f}mm")
        
        self.volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=volume_length,
            resolution=512,
            sdf_trunc=voxel_size * 5,  # 5 voxels truncation
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        print("Starting online integration loop")
        # while True:
        #     self.run_once()
        #     elapsed_time = time.time() - self.start_time
        #     if elapsed_time > self.max_time:
        #         print(f"Reached max time of {self.max_time}s, stopping integration")
        #         break
        # print("\n=== Extracting point cloud ===")
        # pcd = self.volume.extract_point_cloud()
        # print(f"Point cloud: {len(pcd.points)} points")
        # pcd.transform(self.T_restore)

    def publish_point_cloud(self, point_cloud:np.ndarray):
        # Convert to float32 and extract xyz and rgb
        xyz = point_cloud[:, :3].astype(np.float32)
        rgb = point_cloud[:, 3:6].astype(np.float32)
        
        # Pack RGB into a single uint32 (as required by PointCloud2 RGB convention)
        # RGB values should be in [0, 1] range from point_cloud
        rgb_uint8 = np.clip((rgb * 255), 0, 255).astype(np.uint8)
        rgb_packed = (rgb_uint8[:, 0].astype(np.uint32) << 16 | 
                      rgb_uint8[:, 1].astype(np.uint32) << 8 | 
                      rgb_uint8[:, 2].astype(np.uint32))
        
        # Create structured array with the correct dtypes
        cloud_arr = np.zeros(len(xyz), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32),
        ])
        
        cloud_arr['x'] = xyz[:, 0]
        cloud_arr['y'] = xyz[:, 1]
        cloud_arr['z'] = xyz[:, 2]
        cloud_arr['rgb'] = rgb_packed
        
        fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32,  count=1),
        ]

        header = Header()
        header.frame_id = "panda/panda_link0"
        header.stamp = self.get_clock().now().to_msg()
        pc2 = point_cloud2.create_cloud(header, fields, cloud_arr)
        self.point_cloud_pub.publish(pc2)

    def next_frame_callback(self, msg:Bool):
        print("Received next frame signal")
        if msg.data:
            self.cnt += 1
            self.run_once()

    def run_once(self):
        for sample in self.dataset:
            print(f"\n=== Frame {self.cnt} ===")
            pose = sample['sensor_info'].gt.RT.numpy()[0]  # camera-to-world
            rgb = sample['wide']['image'][-1].numpy()
            depth = sample['wide']['depth'][-1].numpy()

            # reorder channels
            rgb = np.transpose(rgb, (1,2,0))
            rgb = rgb.astype(np.uint8)
            depth = depth.astype(np.float32)
            
            camera_pos = pose[:3, 3]
            print(f"Camera position: {camera_pos}")
            print(f"Depth range: [{np.min(depth):.3f}, {np.max(depth):.3f}]m")
            
            # Apply to camera pose: new_pose = pose @ T_recenter^-1
            # (shift world by -center, which shifts camera by +center in world frame)
            T_recenter_inv = np.eye(4)
            T_recenter_inv[:3, 3] = self.workspace_center
            pose_recentered = T_recenter_inv @ pose
            
            # Open3D expects world-to-camera (extrinsic matrix)
            pose_inv = np.linalg.inv(pose_recentered)

            # convert to open3d rgbd image
            color_o3d = o3d.geometry.Image(rgb)
            depth_o3d = o3d.geometry.Image(depth)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
            
            self.volume.integrate(
                rgbd_image,
                camera_intrinsics,
                pose_inv,
            )

            xyzrgb = self.volume.extract_point_cloud()
            xyzrgb.transform(self.T_restore)
            xyzrgb = np.hstack((np.asarray(xyzrgb.points), np.asarray(xyzrgb.colors)))
            self.publish_point_cloud(xyzrgb)

            break
        

def main(args=None):
    rclpy.init(args=args)
    node = OnlineOpen3DNode()
    
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down online 3D node")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    print("Starting online Open3D node")
    main(args=None)
