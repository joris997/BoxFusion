# Save boxes manually for different scenarios
import argparse
import pickle
from typing import List
import os
import numpy as np
from tools.utils import get_objects_from_predicate_file

def create_box_c_s(center:List[float], size:List[float]) -> np.ndarray:
    return np.array([
        [center[0]-size[0]/2, center[1]-size[1]/2, center[2]-size[2]/2],
        [center[0]+size[0]/2, center[1]-size[1]/2, center[2]-size[2]/2],
        [center[0]+size[0]/2, center[1]+size[1]/2, center[2]-size[2]/2],
        [center[0]-size[0]/2, center[1]+size[1]/2, center[2]-size[2]/2],
        [center[0]-size[0]/2, center[1]-size[1]/2, center[2]+size[2]/2],
        [center[0]+size[0]/2, center[1]-size[1]/2, center[2]+size[2]/2],
        [center[0]+size[0]/2, center[1]+size[1]/2, center[2]+size[2]/2],
        [center[0]-size[0]/2, center[1]+size[1]/2, center[2]+size[2]/2],
    ])

def main(args=None, recipe:str=None):
    assert recipe is not None, "Please provide a recipe name."

    results_folder = os.path.expanduser('~/c_space_stl_results')
    # extra objects, predicates from STL, to be detected!
    objects = get_objects_from_predicate_file(recipe,results_folder)
    # get all box_*_<object>.pkl files for the recipe and check against objects
    # if any object is missing, create a manual box for it
    existing_boxes = []
    response_dir = os.path.join(results_folder,recipe)
    for file in os.listdir(response_dir):
        if file.startswith("box_") and file.endswith(".pkl"):
            parts = file.split("_")
            if len(parts) >= 3:
                label = "_".join(parts[2:])[:-4]  # remove .pkl
                existing_boxes.append(label)
    print("Existing boxes:", existing_boxes)
    missing_objects = [obj for obj in objects if obj not in existing_boxes]
    print("Missing objects for manual box creation:", missing_objects)
    

    #! manually created for the different test scenarios
    if recipe == "test":
        # need to create 'box_1_red_square'
        # state-space points
        polygon_points = create_box_c_s(center=[0.5,-0.25,0.1],
                                        size=[0.1,0.1,0.2])
        # point cloud points, we don't have any so we copy polygon points
        cloud_points = polygon_points.copy()
        # label
        label = 'red_square'

        # create the box dict
        box_obj = {
            'polygon_points': polygon_points.tolist(),
            'cloud_points': cloud_points.tolist(),
            'label': label
        }
        
        # save to file
        response_dir = os.path.join(results_folder,recipe)
        file_path = os.path.join(response_dir,f"box_1_{label}.pkl")
        os.makedirs(response_dir, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(box_obj, f)
    
    else:
        print(f"No manual boxes defined for recipe: {recipe}.")

    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online 3D Node")
    parser.add_argument('--recipe', default='none', help='recipe/instruction, loads the response file')
    args = parser.parse_args()

    main(args=None, recipe=args.recipe)