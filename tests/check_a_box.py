import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

# load box_0_cup.pkl
# results_path = os.path.expanduser('~/manipulation_ws/src/BoxFusion/results/test2')
results_path = os.path.expanduser('~/c_space_stl_results/test2')
with open(os.path.join(results_path, "box_1_lipstick.pkl"), 'rb') as f:
    box = pickle.load(f)
    cloud_points = box['cloud_points']
    polygon_points = box['polygon_points']

# load entire point cloud in xyzrgb.pt file
pcd_path = os.path.join(results_path,'xyzrgb.pt')
xyzrgb = torch.load(pcd_path).numpy()

print(polygon_points)

# visualize the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyzrgb[::50,0], xyzrgb[::50,1], xyzrgb[::50,2], c='g', s=1)
# ax.scatter(cloud_points[:,0], cloud_points[:,1], cloud_points[:,2], s=1)
ax.scatter(polygon_points[:,0], polygon_points[:,1], polygon_points[:,2], c='r', s=10)

ax.set_title(f"Label: {box['label']}, Points: {cloud_points.shape[0]}")
plt.show()

#     # and save it back
# with open(os.path.join(results_path, "box_0_cup.pkl"), 'wb') as f:
#     pickle.dump(box, f)