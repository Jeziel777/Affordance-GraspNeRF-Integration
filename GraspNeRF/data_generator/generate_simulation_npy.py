import os
import numpy as np
import random

'''
This configuration may change depending on your local configuation
'''

# === CONFIG ===
URDF_DIR = "~Documents/GraspNerf2/GraspNeRF/data/assets/data/urdfs/vrb_experiments/test"
OUTPUT_DIR = "~Documents/GraspNerf2/GraspNeRF/data/assets/data/mesh_pose_list/vrb_experiment"
NUM_SCENES = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect all URDF files
urdf_paths = sorted([
    os.path.join(URDF_DIR, f)
    for f in os.listdir(URDF_DIR)
    if f.endswith(".urdf")
])

# Correct relative path builder
def to_relative_path(full_path):
    idx = full_path.find("data/urdfs")
    if idx != -1:
        return full_path[idx:]
    else:
        raise ValueError("Could not extract relative URDF path from: " + full_path)

for scene_id in range(NUM_SCENES):
    chosen_urdf = random.choice(urdf_paths)

    scale = round(random.uniform(1.0, 1.5), 3)
    angle = round(random.uniform(0, 2 * np.pi), 3)
    x = 0.15
    y = 0.15
    rel_urdf = to_relative_path(chosen_urdf)

    scene_dict = {
        0: [scale, angle, x, y, rel_urdf]
    }

    out_path = os.path.join(OUTPUT_DIR, f"{scene_id:04d}.npy")
    np.save(out_path, scene_dict)

print(f"Saved {NUM_SCENES} simulation-ready .npy files in: {OUTPUT_DIR}")

