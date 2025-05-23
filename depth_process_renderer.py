# =============================================================================
#  Imports
# =============================================================================

import os
from pathlib import Path
import subprocess
import torch
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gs2mesh_utils.eval_utils import create_strings
from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.eval_utils import prepare_eval, write_to_csv
from third_party.DLNR.core.utils.frame_utils import readPFM, writePFM
from poc_run_dtu import run_DTU_mesh_creation, all_train_scans, test_scans_nums

device = "cuda" if torch.cuda.is_available() else "cpu"
base_dir = os.path.abspath(os.getcwd())

def create_paths(args, strings):
    """
    Creates and returns a dictionary of formatted paths used in the DS creation process.

    Parameters:
    args (Namespace): The arguments from the command line given to the reconstruction/evaluation function.

    Returns:
    dict: A dictionary containing the formatted paths.
    """

    gt_scan_dir = os.path.abspath(os.path.join(base_dir, "data", args.dataset_name, args.colmap_name))
    gt_depth_path = os.path.join(gt_scan_dir, "depths", f"depth_map_{str(args.cam_idx).zfill(4)}.pfm")
    generated_dir_path = strings["output_for_finetune_dir_root"]
    generated_cam_dir_path = os.path.join(generated_dir_path, f"{str(args.cam_idx).zfill(3)}")
    
    generated_render_left_path = os.path.join(generated_cam_dir_path, "left.png")
    generated_render_right_path = os.path.join(generated_cam_dir_path, "right.png")

    generated_disparity_path = os.path.join(generated_cam_dir_path, "out_DLNR_Middlebury", "disparity_LR.npy")
    segmentation_mask_path = os.path.join(generated_cam_dir_path, "left_mask.npy")
    occlusion_mask_path = os.path.join(generated_cam_dir_path, "out_DLNR_Middlebury", "occlusion_mask.npy")

    data_for_finetune_base_path = os.path.abspath(os.path.join(base_dir, args.data_for_finetune_root, args.dataset_name, args.colmap_name))
    data_for_finetune_left_path = os.path.join(data_for_finetune_base_path, "left")
    data_for_finetune_right_path = os.path.join(data_for_finetune_base_path, "right")
    data_for_finetune_disp_path = os.path.join(data_for_finetune_base_path, "disp")

    paths_dict = {
        "gt_scan_dir": gt_scan_dir,
        "gt_depth_path": gt_depth_path,
        "generated_dir_path": generated_dir_path,
        "generated_cam_dir_path": generated_cam_dir_path,
        "generated_render_left_path": generated_render_left_path,
        "generated_render_right_path": generated_render_right_path,
        "generated_disparity_path": generated_disparity_path,
        "segmentation_mask_path": segmentation_mask_path,
        "occlusion_mask_path": occlusion_mask_path,
        "data_for_finetune_base_path": data_for_finetune_base_path,
        "data_for_finetune_left_path": data_for_finetune_left_path,
        "data_for_finetune_right_path": data_for_finetune_right_path,
        "data_for_finetune_disp_path": data_for_finetune_disp_path,
    }

    return paths_dict

# =============================================================================
#  Convert GT depths to disparities
# =============================================================================
args = {
    "camera_data_file": "/home/itamarp/gs2mesh/output_for_finetune/DTU_test_nw_iterations30000_DLNR_Middlebury_baseline7_0p/scan105/camera_data.json",
    "cam_idx": 58
}
paths_dict = create_paths(args, strings)

with open(args["camera_data_file"]) as cam_data_file:
    camera_data = json.load(cam_data_file)

# set parameters
baseline = camera_data[args.cam_idx]["left"]["baseline"]
fx = camera_data[args.cam_idx]["left"]["fx"]

# Load the necessary files
gt_depth_map = readPFM(paths_dict["gt_depth_path"])
generated_disparity = np.load(paths_dict["generated_disparity_path"])
segmentation_mask = np.load(paths_dict["segmentation_mask_path"])
occlusion_mask = np.load(paths_dict["occlusion_mask_path"])

# Combine valid_mask with occlusion mask and segmentation_mask before calculating median depth
valid_mask = ((gt_depth_map > 0) & (occlusion_mask > 0)) & segmentation_mask

# Calculate scale factor for disparity using median values
median_gt_depth = np.median(gt_depth_map[valid_mask])
median_gt_disparity = (fx * baseline) / median_gt_depth
median_generated_disparity = np.median(generated_disparity[valid_mask])
scaling_factor = median_generated_disparity / median_gt_disparity

# Apply the scaling factor to the depth map
scaled_gt_disparity = np.zeros_like(gt_depth_map)
scaled_gt_disparity[valid_mask] = (fx * baseline * scaling_factor) / gt_depth_map[valid_mask]

# Mask out areas with large differences between scaled and generated disparities
difference_mask = np.abs(scaled_gt_disparity - generated_disparity) <= args.difference_mask_threshold
final_mask = valid_mask & difference_mask

# Apply final mask to outputs
masked_gt_disparity = np.where(final_mask, scaled_gt_disparity, 0)
masked_generated_disparity = np.where(final_mask, generated_disparity, 0)

save_disparity_for_ds(args, paths_dict, masked_gt_disparity)

if (args.cam_idx == 34):
    os.makedirs("plots", exist_ok=True)
    plt.imsave(("plots/gt_depth_map.png"), gt_depth_map)
    plt.imsave(("plots/generated_disparity.png"), generated_disparity)
    plt.imsave(("plots/segmentation_mask.png"), segmentation_mask)
    plt.imsave(("plots/occlusion_mask.png"), occlusion_mask)
    plt.imsave(("plots/valid_mask.png"), valid_mask)
    plt.imsave(("plots/scaled_gt_disparity.png"), scaled_gt_disparity)
    plt.imsave(("plots/difference_mask.png"), difference_mask)
    plt.imsave(("plots/final_mask.png"), final_mask)
    plt.imsave(("plots/masked_gt_disparity.png"), masked_gt_disparity)