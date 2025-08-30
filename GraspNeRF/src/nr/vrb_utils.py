import subprocess
import numpy as np
from pathlib import Path
import os
import scipy

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from src.gd.grasp import from_voxel_coordinates
from gd.grasp import Grasp
from gd.utils.transform import Transform

# Function for connecting comunication from both models
def call_vrb_bridge(image_id, data_dir, output_dir, round_id):
    # Get ~/Documents from src/nr/vrb_utils.py
    docs_dir = Path(__file__).resolve().parents[4]
    vrb_venv = Path("/home/jeziel_a/miniconda3/envs/vrbVenv/bin/python")
    vrb_script = docs_dir / "VRB" / "vrb" / "vrb_bridge.py"

    env = os.environ.copy()  # Retain system vars
    env["PYTHONPATH"] = ""   # Clear Blender overrides only

    subprocess.run([
        str(vrb_venv),
        str(vrb_script),
        "--image_id", image_id,
        "--data_dir", str(data_dir),
        "--output_dir", str(output_dir),
        "--round_id", str(round_id)
    ], env=env)


"""
Increase (boost) the grasp quality values in a local region around 
VRB-predicted contact points in the 3D quality volume.

Args:
    qual_vol (np.ndarray): 3D volume of grasp quality scores.
    points_world (list or np.ndarray): One or more contact points in world coordinates.
    voxel_size (float): Size of each voxel in meters.
    radius (int): Radius around each voxel to boost.
    boost_value (float): The value to assign or max into the boosted region.
    
Returns:
    qual_vol (np.ndarray): The updated quality volume.
"""

def boost_vrb_region(qual_vol, points_world, voxel_size, radius=3, boost_value=0.95, tsdf_vol=None, tsdf_thresh=0.1):

    # Ensure we always have a list of 3D points
    if not isinstance(points_world, list):
        points_world = [points_world]

    for point in points_world:
        # Convert world coordinates to voxel indices
        voxel = np.round(np.array(point) / voxel_size).astype(int)

        # Skip invalid voxel indices (outside volume bounds)
        if np.any(voxel < 0) or np.any(voxel >= np.array(qual_vol.shape) - radius):
            continue
        
        # OPTIONAL: skip voxels far from surface
        if tsdf_vol is not None:
            tsdf_val = tsdf_vol[tuple(voxel)]
            if tsdf_val > tsdf_thresh:  # too far from object surface
                continue

        # Get center voxel index
        x, y, z = voxel

        # Define 3D slicing bounds around the voxel, clipped to volume dimensions
        xs = slice(max(x - radius, 0), min(x + radius + 1, qual_vol.shape[0]))
        ys = slice(max(y - radius, 0), min(y + radius + 1, qual_vol.shape[1]))
        zs = slice(max(z - radius, 0), min(z + radius + 1, qual_vol.shape[2]))

        # Boost the values in the local region by taking the maximum of each voxel and boost_value
        qual_vol[xs, ys, zs] = np.maximum(qual_vol[xs, ys, zs], boost_value)

    return qual_vol



def boost_qual_near_vrb_affordances(
    qual_vol, contact_points, voxel_size=0.0075, distance_thresh=0.03
):
    """
    Boost grasp quality near predicted contact points.

    Args:
        qual_vol (np.ndarray): 3D volume of grasp quality scores.
        contact_points (np.ndarray): (N, 3) array of 3D world coordinates.
        voxel_size (float): Size of each voxel in meters.
        distance_thresh (float): Max distance (in meters) to apply boost.

    Returns:
        qual_vol_boosted (np.ndarray): Boosted quality volume.
    """

    qual_vol_boosted = qual_vol.copy()
    boosted_mask = np.zeros_like(qual_vol, dtype=bool)

    boost_value = 0.05
    quality_cap = 0.99
    offset = np.array([0.15, 0.15, 0.05])  # to align contact points to TSDF origin

    radius_vox = int(np.ceil(distance_thresh / voxel_size))

    for pt in contact_points:
        voxel_coords = (pt + offset) / voxel_size
        vi, vj, vk = np.floor(voxel_coords).astype(int)

        for di in range(-radius_vox, radius_vox + 1):
            for dj in range(-radius_vox, radius_vox + 1):
                for dk in range(-radius_vox, radius_vox + 1):
                    ii, jj, kk = vi + di, vj + dj, vk + dk

                    if (
                        0 <= ii < qual_vol.shape[0]
                        and 0 <= jj < qual_vol.shape[1]
                        and 0 <= kk < qual_vol.shape[2]
                        and not boosted_mask[ii, jj, kk]
                    ):
                        # Compute voxel center in world coordinates
                        voxel_center = np.array([ii + 0.5, jj + 0.5, kk + 0.5]) * voxel_size
                        if np.linalg.norm(voxel_center - pt) <= distance_thresh:
                            qual_vol_boosted[ii, jj, kk] = min(
                                qual_vol_boosted[ii, jj, kk] + boost_value, quality_cap
                            )
                            boosted_mask[ii, jj, kk] = True

    return qual_vol_boosted



def combine_contact_points(folder_path, top_k=150):
    """
    Loads and combines top contact points from multiple VRB .npz files in a folder.

    Args:
        folder_path (str): Path to the folder containing six *_3d_contacts.npz files.
        top_k (int): Number of closest contact points to keep from each file.

    Returns:
        np.ndarray: Combined (top_k * num_files, 3) array of contact point positions (x, y, z).
    """
    all_points = []

    # Collect all .npz files ending with '_3d_contacts.npz'
    files = [f for f in os.listdir(folder_path) if f.endswith('_3d_contacts.npz')]
    files = sorted(files)  # ensure consistent order

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        data = np.load(file_path)
        
        if "contacts" not in data:
            raise ValueError(f"File {file_name} does not contain 'contacts' key.")

        contacts = data["contacts"]  # shape: (N, 4) -> x, y, z, distance
        num_contacts = min(len(contacts), top_k)
        top_contacts = contacts[:num_contacts, :3]
        top_contacts = contacts[:top_k, :3]  # take top_k closest, discard distance column
        all_points.append(top_contacts)

    if not all_points:
        raise RuntimeError(f"No valid .npz files found in {folder_path}")
    
    combined_points = np.vstack(all_points)
    return combined_points


def contact_points_from_single_npz(file_path):
    """
    Loads a SINGLE *_3d_contacts.npz file.
    - Assumes 'contacts' array columns are [x, y, z, distance] and that rows
      are already ordered by increasing distance.
    """
    data = np.load(file_path)
    if "contacts" not in data:
        raise ValueError(f"File {os.path.basename(file_path)} missing 'contacts' key.")
    contacts = data["contacts"]  # shape: (N, 4) -> x, y, z, distance
    positions = contacts[:, :3]  # keep order (already sorted by distance)
    
    return positions