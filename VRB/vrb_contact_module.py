# vrb_contact_module.py

from pathlib import Path
import numpy as np
from PIL import Image
import torch
import os
import sys
import OpenEXR
import Imath

# Add VRB path to import its model and inference function
VRB_PATH = Path(__file__).resolve().parent  # folder where this script lives
sys.path.append(str(VRB_PATH))

from inference import run_inference
from networks.model import VRBModel
from networks.traj import TrajAffCVAE


def load_vrb_model(model_path, args, device = "cpu"):
        
    # Setup hand trajectory head
    hand_head = TrajAffCVAE(in_dim=2*args.traj_len, hidden_dim=args.hidden_dim,
                            latent_dim=args.hand_latent_dim, condition_dim=args.cond_dim,
                            coord_dim=args.coord_dim, traj_len=args.traj_len)
    
    src_in_features = 2048 if args.resnet_type == 'resnet50' else 512

    net = VRBModel(src_in_features=src_in_features,
                   num_patches=1,
                   hidden_dim=args.hidden_dim, 
                   hand_head=hand_head,
                   encoder_time_embed_type=args.encoder_time_embed_type,
                   num_frames_input=10,
                   resnet_type=args.resnet_type, 
                   embed_dim=args.cond_dim, coord_dim=args.coord_dim,
                   num_heads=args.num_heads, enc_depth=args.enc_depth, 
                   attn_kp=args.attn_kp, attn_kp_fc=args.attn_kp_fc, n_maps=5)

    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint)
    net.to(device)
    net.eval()
    
    return net


def get_3d_contact_points(image_path, intrinsics, extrinsics, depth_image, vrb_model, args):
   
    # Load and resize RGB image to match model input
    img = Image.open(image_path).convert("RGB")
    img = img.resize((1008, 756))
    
    # Run model inference
    im_out, all_candidates_2d = run_inference(vrb_model, img, return_all_contacts=True)
    if all_candidates_2d is None or len(all_candidates_2d) == 0:
        print("[VRB] No contact point found — skipping image.")
        return None, None, im_out

    # Rescale contacts from model image size to depth image size
    scale_x = 640 / 1008
    scale_y = 360 / 756
    contact_points_scaled = np.copy(all_candidates_2d)
    contact_points_scaled[:, 0] *= scale_x  # u
    contact_points_scaled[:, 1] *= scale_y  # v
    
    # Extract intrinsics
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Extrinsic world-to-camera
    Rt4x4 = np.eye(4)
    Rt4x4[:3, :] = extrinsics

    contact_3d_points = []
    valid_pixels = []

    # Backprojection
    for (u, v) in contact_points_scaled.astype(int):
        if not (0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]):
            continue
        Z = depth_image[v, u]
        if Z <= 0 or np.isnan(Z):
            continue
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        point_camera = np.array([X, Y, Z, 1.0])
        point_world = Rt4x4 @ point_camera
        contact_3d_points.append(point_world[:3])
        valid_pixels.append((u, v))

    if len(contact_3d_points) == 0:
        print("[VRB] All candidate contacts were invalid in depth — skipping.")
        return None, None, im_out

    # Return first valid point 
    first_point_world = contact_3d_points[0]
    first_pixel = valid_pixels[0]

    return first_point_world, first_pixel, im_out, contact_3d_points


def load_exr_depth(exr_path):
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_data = exr_file.channel('R', pt)  # Depth stored in 'R' channel
    depth = np.frombuffer(depth_data, dtype=np.float32).reshape((height, width))

    return depth
