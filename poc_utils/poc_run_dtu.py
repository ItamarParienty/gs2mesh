# =============================================================================
#  Imports
# =============================================================================

import os
from pathlib import Path
import subprocess
from itertools import chain
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from poc_run_single import run_single
from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.eval_utils import prepare_eval, write_to_csv, create_strings
base_dir = os.path.abspath(os.getcwd())

# =============================================================================
#  Run
# =============================================================================
test_scans_nums = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

specific_prompts = {
    15: "building",
    16: "building",
    17: "building",
    18: "building",
    21: "building",
    22: "building",
    43: "building",
    46: "building",
    103: "pig",
}

def all_train_scans():
    all_scans = chain(range(1, 78), range(82, 129))
    return [scan for scan in all_scans if scan not in test_scans_nums]

def move_dir(src_dir_path, dst_dir_path):
    print(f"---Moving dir {src_dir_path} to {dst_dir_path} ---")
    dst_dir_parent_path = os.path.dirname(dst_dir_path)

    if os.path.exists(dst_dir_path):
            shutil.rmtree(dst_dir_path)
    Path(dst_dir_parent_path).mkdir(parents=True, exist_ok=True)
    shutil.move(src_dir_path, dst_dir_parent_path)

def set_poc_args(args, scan_num):
    args.skip_colmap = False
    args.skip_TSDF = False
    args.stereo_warm = False
    args.renderer_save_json = True
    args.masker_automask = True
    args.masker_prompt = "main_object"
    args.masker_SAM2_local = False

    args.masker_prompt = specific_prompts[scan_num] if scan_num in specific_prompts else 'main_object'

    args.colmap_name = f"scan{scan_num}"
    args.GS_port = args.GS_port + scan_num

    args.dataset_name = "DTU_test" if scan_num in test_scans_nums else "DTU_train"

    # TODO: delete
    args.skip_colmap = True
    args.skip_GS = True
    # args.skip_rendering = True
    # args.skip_masking = True
    return args


def run_DTU_mesh_creation(args):
    # =============================================================================
    #  Create disparities, and masks
    # =============================================================================
    if args.scans == [0]:
        args.scans = all_train_scans()
    if args.scans == [-1]:
        args.scans = test_scans_nums
         

    for scan_num in args.scans:
        # =============================================================================
        #  create output disparities and masks
        # =============================================================================
        
        # =============================================================================
        #  Set Scan Args
        # =============================================================================
        args = set_poc_args(args, scan_num)
        
        print(f"----START DTU Mesh Creation SCAN {scan_num}----")
        print("args:")
        print(args)

        strings = create_strings(args)
        output_dir_root = strings["output_dir_root"]
        output_for_finetune_dir_root = output_dir_root.replace("output", "output_for_finetune")
        splatting_output_dir_root = os.path.join(base_dir, 'splatting_output', strings['splatting'], args.colmap_name)
        splatting_output_for_finetune_dir_root = splatting_output_dir_root.replace("output", "output_for_finetune")

        # =============================================================================
        #  Create mesh
        # =============================================================================

        # if didn't ran the GS, there will be no splatting_output dir, 
        # but there may be one from previous run in the splatting_output_for_finetune dir so we move it to the output dir
        if (not os.path.exists(splatting_output_dir_root)) and (args.skip_GS) and (os.path.exists(splatting_output_for_finetune_dir_root)):
            move_dir(splatting_output_for_finetune_dir_root, splatting_output_dir_root)

        run_single(args)

        # =============================================================================
        #  Move Outputs Folders to Avoid Clashes
        # =============================================================================

        move_dir(output_dir_root, output_for_finetune_dir_root)
        move_dir(splatting_output_dir_root, splatting_output_for_finetune_dir_root)


# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser('DTU')
    args = parser.parse_args()

    run_DTU_mesh_creation(args)
