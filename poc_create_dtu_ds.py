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

from gs2mesh_utils.eval_utils import create_strings
from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.eval_utils import prepare_eval, write_to_csv
from third_party.DLNR.core.utils.frame_utils import readPFM

device = "cuda" if torch.cuda.is_available() else "cpu"
base_dir = os.path.abspath(os.getcwd())

# =============================================================================
#  Run
# =============================================================================

def convert_gt_depth_to_disparities(args, strings, cam_idx):
    # =============================================================================
    #  Convert GT depths to disparities
    # =============================================================================
    #set paths
    gt_scan_dir = os.path.abspath(os.path.join(base_dir, "data", args.dataset_name, args.colmap_name))
    gt_depth_path = os.path.join(gt_scan_dir, "depths", f"depth_map_{str(cam_idx).zfill(4)}.pfm")
    output_dir_path = strings["output_dir_root"]
    cam_output_dir_path = os.path.join(output_dir_path, f"{str(cam_idx).zfill(3)}")
    camera_data = json.load(os.path.join(cam_output_dir_path,'camera_data.json'))

    # set parameters
    baseline = camera_data['left']['baseline']
    fx = camera_data['left']['fx']

    # Load the necessary files
    gt_depth_map = readPFM(gt_depth_path)
    generated_disparity = np.load(os.path.join(cam_output_dir_path, 'out_DLNR_Middlebury','disparity_LR.npy'))
    segmentation_mask = np.load(os.path.join(cam_output_dir_path, 'left_mask.npy'))
    occlusion_mask = np.load(os.path.join(cam_output_dir_path, 'out_DLNR_Middlebury','occlusion_mask.npy'))

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
    threshold = 5  # Set a threshold for large differences
    difference_mask = np.abs(scaled_gt_disparity - generated_disparity) <= threshold
    final_mask = valid_mask & difference_mask

    # Apply final mask to outputs
    masked_gt_disparity = np.where(final_mask, scaled_gt_disparity, 0)
    masked_generated_disparity = np.where(final_mask, generated_disparity, 0)




def create_dtu_gt_disparities(args):
    # =============================================================================
    #  Create GT disparities
    # =============================================================================

    args.dataset_name = os.path.join("DTU", "DTU_train")

    for scan_num in args.scans:
        # =============================================================================
        #  create GT disparity for single scan
        # =============================================================================
        args.colmap_name = f"scan{scan_num}"
        args.GS_port = GS_port_orig + scan_num
        print(f"----START PROCESSING SCAN {scan_num}----")
        strings = create_strings(args)
        convert_gt_depth_to_disparities(args, strings, 0)


# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser("DTU")
    args = parser.parse_args()
    GS_port_orig = args.GS_port

    create_dtu_gt_disparities(args)
