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
print(camera_matrix)
camera_matrix = np.array(camera_matrix)
camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsics.set_intrinsics(
    width=640,
    height=480,
    fx=camera_matrix[0,0],
    fy=camera_matrix[1,1],
    cx=camera_matrix[0,2],
    cy=camera_matrix[1,2],
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
        super().__init__('online_open3d_node')
        print("done init")
        self.cfg = cfg
        print("Creating ROSDataset for online Open3D integration")
        dataset = ROSDataset(cfg)
        dataset.load_arkit_depth = True

        # volume object
        print("Creating TSDF volume")
        volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=4.0,
            resolution=512,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for sample in dataset:
            print(f"Integrate frame into the volume.")
            pose = sample['sensor_info'].gt.RT.numpy()
            rgb = sample['sensor_info'].wide.image.K[-1].numpy()
            depth = sample['sensor_info'].wide.depth.K[-1].numpy()
            print(f"pose: {pose}")
            print(f"rgb shape: {rgb.shape}, depth shape: {depth.shape}")
            print(f"rgb: {rgb}, depth: {depth}")

            # reorder channels
            rgb = np.transpose(rgb, (1,2,0))
            rgb = rgb.astype(np.uint8)
            depth = depth.astype(np.float32)

            # convert to open3d rgbd image
            color_o3d = o3d.geometry.Image(rgb)
            depth_o3d = o3d.geometry.Image(depth)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_trunc=4.0, convert_rgb_to_intensity=False)
            
            volume.integrate(
                rgbd_image,
                camera_intrinsics,
                np.linalg.inv(pose),
            )

            voxel_pcd = volume.extract_voxel_point_cloud()
            o3d.visualization.draw_geometries([voxel_pcd])


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