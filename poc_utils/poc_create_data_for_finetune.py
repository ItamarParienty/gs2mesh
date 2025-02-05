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

from gs2mesh_utils.eval_utils import create_strings
from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.eval_utils import prepare_eval, write_to_csv
from third_party.DLNR.core.utils.frame_utils import readPFM, writePFM
from poc_run_dtu import run_DTU_mesh_creation

device = "cuda" if torch.cuda.is_available() else "cpu"
base_dir = os.path.abspath(os.getcwd())


# =============================================================================
#  Run
# =============================================================================
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


def save_disparity_for_ds(args, paths_dict, gt_disparity):
    os.makedirs(paths_dict["data_for_finetune_disp_path"], exist_ok=True)
    out_file_path = os.path.join(paths_dict["data_for_finetune_disp_path"], f"img{str(args.cam_idx)}")
    np.save((out_file_path + ".npy"), gt_disparity)
    writePFM((out_file_path + ".pfm"), gt_disparity)

    gt_disparity_png = np.nan_to_num(gt_disparity, nan=0)

    # Normalize array to 0–255 for 8-bit PNG 
    gt_disparity_min, gt_disparity_max = np.min(gt_disparity_png), np.max(gt_disparity_png)
    normalized_gt_disparity = ((gt_disparity_png - gt_disparity_min) / (gt_disparity_max - gt_disparity_min) * 255).astype(np.uint8)
    plt.imsave((out_file_path + ".png"), normalized_gt_disparity)


def convert_gt_depth_to_disparities(args, strings):
    # =============================================================================
    #  Convert GT depths to disparities
    # =============================================================================
    paths_dict = create_paths(args, strings)

    with open(args.camera_data_files) as camera_data_file:
        camera_data = json.load(camera_data_file)

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
    threshold = 5  # Set a threshold for large differences
    difference_mask = np.abs(scaled_gt_disparity - generated_disparity) <= threshold
    final_mask = valid_mask & difference_mask
    
    # Apply final mask to outputs
    # masked_gt_disparity = np.where(final_mask, scaled_gt_disparity, np.NaN)
    # masked_generated_disparity = np.where(final_mask, generated_disparity, np.NaN)
    # CHANGED TO 0 BACKGROUND BECAUSE OF ISSUES WITH DLNR
    masked_gt_disparity = np.where(final_mask, scaled_gt_disparity, 0)
    masked_generated_disparity = np.where(final_mask, generated_disparity, 0)

    save_disparity_for_ds(args, paths_dict, masked_gt_disparity)

def copy_renders_to_ds_folder(args, strings):
    # =============================================================================
    #  Copy the left and right renders from the generated folder to the DS folder
    # =============================================================================
    paths_dict = create_paths(args, strings)
    
    os.makedirs(paths_dict["data_for_finetune_left_path"], exist_ok=True)
    os.makedirs(paths_dict["data_for_finetune_right_path"], exist_ok=True)

    left_out_path = os.path.join(paths_dict["data_for_finetune_left_path"], f"img{str(args.cam_idx)}.png")
    right_out_path = os.path.join(paths_dict["data_for_finetune_right_path"], f"img{str(args.cam_idx)}.png")

    copyfile(paths_dict['generated_render_left_path'], left_out_path)
    copyfile(paths_dict['generated_render_right_path'], right_out_path)



def create_DTU_data_for_finetune(args):
    # =============================================================================
    #  Create GT disparities
    # =============================================================================
    args.dataset_name = "DTU_test"
    args.data_for_finetune_root = "data_for_finetune"
    args.skip_create_mesh = False

    for scan_num in args.scans:
        # =============================================================================
        #  Create Data for Fintune for Single Scan
        # =============================================================================
        print(f"----START PROCESSING SCAN {scan_num}----")
        args.colmap_name = f"scan{scan_num}"
        strings = create_strings(args)
        strings["output_for_finetune_dir_root"] = strings["output_for_finetune_dir_root"].replace("output", "output_for_finetune")

        # =============================================================================
        #  Create Mesh from Scan for the Finetuning
        # =============================================================================
        if not args.skip_create_mesh:
            run_DTU_mesh_creation(args)

        #get number of cameras in this scan
        cameras_data_file = os.path.join(strings["output_for_finetune_dir_root"], "camera_data.json")
        with open(cameras_data_file) as camera_data_file:
            camera_data = json.load(camera_data_file)
        args.camera_data_file = cameras_data_file
        args.cams_num = len(camera_data)

        #process and copy data to final DS
        for cam_idx in range(args.cams_num):
            print(f"----START PROCESSING CAM {cam_idx}----")
            args.cam_idx = cam_idx
            convert_gt_depth_to_disparities(args, strings)
            copy_renders_to_ds_folder(args, strings)


# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser("DTU")
    args = parser.parse_args()
    create_DTU_data_for_finetune(args)
