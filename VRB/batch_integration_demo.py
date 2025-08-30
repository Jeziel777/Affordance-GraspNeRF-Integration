from pathlib import Path
import numpy as np
import os
from utils import load_vrb_model, load_exr_depth, get_3d_contact_points, sort_points_by_centroid_distance

# Setup VRB model with fixed config 
class Args:
    num_heads = 8
    enc_depth = 6
    hidden_dim = 192
    hand_latent_dim = 4
    cond_dim = 256
    coord_dim = 64
    resnet_type = 'resnet18'
    attn_kp = 1
    attn_kp_fc = 1
    traj_len = 5
    encoder_time_embed_type = 'sin'

# Load pre-trained model
args = Args()
model = load_vrb_model("models/model_checkpoint_1249.pth.tar", args)

# Target image ids in simulation 
image_ids = ['0002', '0006', '0010', '0014', '0018', '0022']

# Blender uses Z-forward, Y-up; OpenCV uses Z-backward, Y-down
blender2opencv = np.diag([1, -1, -1, 1])

# Load common camera poses for computing extrinsics
poses_all = np.load('camera_pose.npy')  # (24, 4, 4)
final_poses = [(poses_all[i] @ blender2opencv)[:3, :] for i in range(24)]

# Intrinsics from GraspNeRF simulation
intrinsic = np.array([
    [450.0, 0.0, 320.0],
    [0.0, 450.0, 180.0],
    [0.0,   0.0,   1.0]
])

def process_dataset(data_dir: Path, results_dir: Path):
    """Process one dataset folder and save results."""
    os.makedirs(results_dir, exist_ok=True)

    # Load of intrinsics/extrinsics file (not used)
    npz_path = data_dir / 'intrinsics_extrinsics_round48.npz'
    if npz_path.exists():
        np.load(npz_path)

    for img_id in image_ids:
        print(f"\n[INFO] {data_dir.name}: Processing image {img_id}...")
        extrinsic = final_poses[int(img_id)]
        image_path = data_dir / 'rgb' / f'{img_id}.png'
        depth_path = data_dir / 'depth' / f'{img_id}_0.exr'

        if not depth_path.exists() or not image_path.exists():
            print(f"[WARN] Missing files for {img_id} in {data_dir}")
            continue

        try:
            depth_map = load_exr_depth(str(depth_path))
            if depth_map.ndim == 3:
                depth_map = depth_map[:, :, 0]
        except Exception as e:
            print(f"[ERROR] Could not load depth map for {img_id}: {e}")
            continue

        result = get_3d_contact_points(
            image_path=str(image_path),
            intrinsics=intrinsic,
            extrinsics=extrinsic,
            depth_image=depth_map,
            vrb_model=model,
            args=args,
        )

        if result[0] is None:
            print("[INFO] No contact point found.")
            continue

        point_world, contact_pixel, overlay_img, all_3d_points = result
        
        # Order contacts by distance to centroid
        sorted_pts, dists, centroid = sort_points_by_centroid_distance(all_3d_points)
        contacts_with_dist = np.hstack([sorted_pts, dists[:, None]])
        
        out_img = results_dir / f'{img_id}_vrb.png'
        overlay_img.save(out_img)
        # Save contacts and centroid information
        np.savez_compressed(results_dir / f'{img_id}_3d_contacts.npz',
                            contacts=contacts_with_dist,
                            centroid=centroid)
        print(f"[Saved] {out_img}")

'''
Runs the VRB inference on the number of rounds * number of images
that the corresponding dataset folder has.

This part changed depending on the three different 
datasets configuration:
  - single_orig_specular
  - single_orig_diffuse
  - single_orig_trans
'''

def main():
    root = Path('single_orig_diffuse') # this changes
    results_root = Path(f'{root.name}_results')
    os.makedirs(results_root, exist_ok=True)

    # Save the inference in results folder
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        results_dir = results_root / sub.name
        process_dataset(sub, results_dir)

if __name__ == '__main__':
    main()
