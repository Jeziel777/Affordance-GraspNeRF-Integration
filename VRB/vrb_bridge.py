from pathlib import Path
import torch
import argparse
import numpy as np
import os
from PIL import Image

from vrb_contact_module import load_vrb_model, get_3d_contact_points, load_exr_depth

# This gets ~/Documents/VRB for local configuration was ~/Documents/VRB/vrb/
ROOT_DIR = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_id", required=True, help="e.g., 0002")
    parser.add_argument("--data_dir", required=True, help="Folder with image/depth/intrinsics_extrinsics.npz")
    parser.add_argument("--output_dir", required=True, help="Where to save round-specific data like vrb_contact_XXXX.npz")
    parser.add_argument("--round_id", required=True, help="Round index used to find intrinsics_extrinsics_roundX.npz")

    args = parser.parse_args()

    #Load paths
    image_path = os.path.join(args.data_dir, "rgb", f"{args.image_id}.png")
    depth_path = os.path.join(args.data_dir, "depth", f"{args.image_id}_0.exr")
    meta_path = os.path.join(args.data_dir, "intrinsics_extrinsics_round0.npz")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    #Load extrinsics and instrinsics
    npz = np.load(meta_path)
    idx = ["0002", "0006", "0010", "0014", "0018", "0022"].index(args.image_id)
    intrinsics = npz["intrinsics"][idx]
    extrinsics = npz["extrinsics"][idx]

    # Load depth map
    depth = load_exr_depth(depth_path)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

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
    vrb_args = Args()
    
    # Load pre-trained model
    model_path = ROOT_DIR / "vrb" / "models" / "model_checkpoint_1249.pth.tar"
    model = load_vrb_model(str(model_path), vrb_args, device="cpu")

    # Run VRB inference
    result = get_3d_contact_points(
        image_path, intrinsics, extrinsics, depth, model, vrb_args
    )

    if result[0] is None:
        print("[VRB Bridge] No contact points found.")
        return

    point_world, contact_pixel, overlay_image, all_points = result

    out_file = output_path / f"vrb_contact_{args.image_id}.npz"
    np.savez_compressed(out_file,
                        point_world=point_world,
                        contact_pixel=contact_pixel,
                        all_points=np.array(all_points))
    print(f"[VRB Bridge] Saved contact to: {out_file}")
    
    # Saving Image
    try:
        name_wo_ext = Path(image_path).stem
        output_img_path = output_path / f"{name_wo_ext}_vrb.png"
        overlay_image.save(output_img_path)
        print(f"[Saved] {output_img_path}")
    except Exception as e:
        print(f"[VRB Bridge] Warning: Failed to save overlay image — {e}")
    

if __name__ == "__main__":
    main()

