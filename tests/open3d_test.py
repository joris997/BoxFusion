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
import os, sys
from rclpy import Parameter
import time

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

        # Define workspace bounds (meters)
        self.workspace_min = np.array([-0.5, 0.0, -0.1])
        self.workspace_max = np.array([0.5, 1.2, 0.5])
        
        # Calculate workspace center and dimensions
        self.workspace_center = (self.workspace_min + self.workspace_max) / 2.0
        workspace_size = self.workspace_max - self.workspace_min
        max_dim = np.max(workspace_size)
        
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
        self.run()

    def run(self):
        cnt = 0
        max_cnt = 7
        for sample in self.dataset:
            print(f"\n=== Frame {cnt} ===")
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
            
            # Transform pose to center volume on workspace
            # Create translation matrix to shift workspace center to origin
            T_recenter = np.eye(4)
            T_recenter[:3, 3] = -self.workspace_center
            
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

            cnt += 1
            if cnt >= max_cnt:
                break

        print("\n=== Extracting point cloud ===")
        pcd = self.volume.extract_point_cloud()
        print(f"Point cloud: {len(pcd.points)} points")
        
        # Transform point cloud back to original world coordinates
        T_restore = np.eye(4)
        T_restore[:3, 3] = self.workspace_center
        pcd.transform(T_restore)
        
        # Verify points are in expected workspace
        points = np.asarray(pcd.points)
        if len(points) > 0:
            print(f"Point cloud bounds:")
            print(f"  X: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
            print(f"  Y: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
            print(f"  Z: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
        
        o3d.visualization.draw_geometries([pcd])

        # xyz = np.asarray(xyz.points)
        # rgb = np.asarray(xyz.colors)
        # xyzrgb = np.hstack((xyz, rgb))
        # print("Extracted point cloud from volume")
        # print(f"number of points: {xyzrgb.shape}")

        


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

# if __name__ == "__main__":
#     rgbd_data = o3d.data.SampleRedwoodRGBDImages()
#     camera_poses = read_trajectory(rgbd_data.odometry_log_path)
#     camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
#         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
#     volume = o3d.pipelines.integration.UniformTSDFVolume(
#         length=4.0,
#         resolution=512,
#         sdf_trunc=0.04,
#         color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
#     )

#     for i in range(len(camera_poses)):
#         print("Integrate {:d}-th image into the volume.".format(i))
#         color = o3d.io.read_image(rgbd_data.color_paths[i])
#         depth = o3d.io.read_image(rgbd_data.depth_paths[i])
#         print(f"color: {color}, depth: {depth}")

#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
#         volume.integrate(
#             rgbd,
#             camera_intrinsics,
#             "boop",#np.linalg.inv(camera_poses[i].pose),
#         )

#     # print("Extract triangle mesh")
#     # mesh = volume.extract_triangle_mesh()
#     # mesh.compute_vertex_normals()
#     # o3d.visualization.draw_geometries([mesh])

#     print("Extract voxel-aligned debugging point cloud")
#     voxel_pcd = volume.extract_voxel_point_cloud()
#     o3d.visualization.draw_geometries([voxel_pcd])

#     print("Extract voxel-aligned debugging voxel grid")
#     voxel_grid = volume.extract_voxel_grid()
#     # o3d.visualization.draw_geometries([voxel_grid])

#     # print("Extract point cloud")
#     # pcd = volume.extract_point_cloud()
#     # o3d.visualization.draw_geometries([pcd])