# Save boxes manually for different scenarios
import argparse
import pickle
from typing import List
import os
import numpy as np
import torch
from c_space_stl.c_space_stl.helpers.sets import Polytope
import matplotlib.pyplot as plt
from tools.utils import get_objects_from_predicate_file, points_in_box

 
def main(args=None, recipe:str=None, plot:bool=False):
    assert recipe is not None, "Please provide a recipe name."

    results_folder = os.path.expanduser('~/c_space_stl_results')
    # extra objects, predicates from STL, to be detected!
    objects = get_objects_from_predicate_file(recipe,results_folder)
    # get all box_*_<object>.pkl files for the recipe and check against objects
    # if any object is missing, create a manual box for it
    existing_boxes = []
    response_dir = os.path.join(results_folder,recipe)

    # load original point cloud from BoxFusion in xyzrgb.npy
    point_cloud_path = os.path.join(results_folder, recipe, 'xyzrgb.npy')
    with open(point_cloud_path, "rb") as f:
        xyzrgb = np.load(f)

    # load open3d point cloud as xyzrgb_o3d.npy
    point_cloud_o3d_path = os.path.join(results_folder, recipe, 'xyzrgb_o3d.npy')
    with open(point_cloud_o3d_path, "rb") as f:
        xyzrgb_o3d = np.load(f)

    # loop through all the existing boxes and change the point cloud points
    # to use the open3d point cloud
    for file in os.listdir(response_dir):
        if file.startswith("box_") and file.endswith(".pkl"):
            # load the pkl file
            with open(os.path.join(response_dir, file), "rb") as f:
                box_obj = pickle.load(f)
                old_box_obj = box_obj.copy()
                
                poly = Polytope(box_obj['polygon_points'])
                box_obj['cloud_points'] = points_in_box(poly, xyzrgb=xyzrgb_o3d)

                if plot:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(np.array(old_box_obj['cloud_points'])[:,0],
                               np.array(old_box_obj['cloud_points'])[:,1],
                               np.array(old_box_obj['cloud_points'])[:,2],
                               c='red', s=5, label='Old Box Cloud Points')
                    ax.scatter(np.array(box_obj['cloud_points'])[:,0],
                               np.array(box_obj['cloud_points'])[:,1],
                               np.array(box_obj['cloud_points'])[:,2],
                               c='green', s=5, label='New Box Cloud Points')
                    poly.plot(ax, color='blue', alpha=1.0)
                    ax.set_title(f'Box: {box_obj["label"]}')
                    ax.legend()
                    plt.savefig(os.path.join(response_dir, f'box_{box_obj["label"]}_comparison.png'))

                # save back to pkl
            with open(os.path.join(response_dir, file), "wb") as f:
                pickle.dump(box_obj, f)

    print("Existing boxes:", existing_boxes)
    missing_objects = [obj for obj in objects if obj not in existing_boxes]
    print("Missing objects for manual box creation:", missing_objects)
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online 3D Node")
    parser.add_argument('--recipe', default='none', help='recipe/instruction, loads the response file')
    parser.add_argument('--plot', default=False, action='store_true', help='whether to plot the boxes')
    args = parser.parse_args()

    main(args=None, recipe=args.recipe, plot=args.plot)