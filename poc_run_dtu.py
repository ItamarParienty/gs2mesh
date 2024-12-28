# =============================================================================
#  Imports
# =============================================================================

import os
from pathlib import Path
import subprocess
from itertools import chain
import sys

from poc_run_single import run_single
from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.eval_utils import prepare_eval, write_to_csv

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

def set_poc_args(scan_num):
    args.skip_TSDF = True
    args.stereo_warm = False
    args.renderer_save_json = True
    args.masker_automask = True
    args.masker_SAM2_local = False

    args.skip_colmap = scan_num in specific_prompts
    args.skip_GS = scan_num in specific_prompts
    args.skip_rendering = scan_num in specific_prompts
    args.masker_prompt = specific_prompts[scan_num] if scan_num in specific_prompts else 'main_object'

    # TODO: delete
    args.skip_colmap = True
    args.skip_GS = True
    args.skip_rendering = True
    args.skip_masking = True


def run_DTU_POC(args):
    # =============================================================================
    #  Create disparities, and masks
    # =============================================================================
    args.dataset_name = os.path.join("DTU", "train")

    if args.scans == [0]:
        args.scans = all_train_scans()

    for scan_num in args.scans:

        # =============================================================================
        #  create output disparities and masks
        # =============================================================================
        if scan_num in test_scans_nums:
            continue
        
        set_poc_args(scan_num)
        args.colmap_name = f"scan{scan_num}"
        args.GS_port = GS_port_orig + scan_num
        print(f"----START PROCESSING SCAN {scan_num}----")
        print("args:")
        print(args)
        run_single(args)


# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser('DTU')
    args = parser.parse_args()
    GS_port_orig = args.GS_port

    run_DTU_POC(args)
