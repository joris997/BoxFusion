import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# load box_0_cup.pkl
results_path = os.path.expanduser('~/manipulation_ws/src/BoxFusion/results/test2')
with open(os.path.join(results_path, "box_2_soap.pkl"), 'rb') as f:
    box = pickle.load(f)
    box['cloud_points'] = box['cloud_points']

    # visualize the points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(box['cloud_points'][:,0], box['cloud_points'][:,1], box['cloud_points'][:,2], s=1)
    ax.set_title(f"Label: {box['label']}, Points: {box['cloud_points'].shape[0]}")
    plt.show()

#     # and save it back
# with open(os.path.join(results_path, "box_0_cup.pkl"), 'wb') as f:
#     pickle.dump(box, f)