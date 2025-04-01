# =============================================================================
#  Imports
# =============================================================================

import os
from pathlib import Path
import subprocess
from itertools import chain
import sys
import shutil
import numpy as np
import open3d as o3d
import pymeshlab as ml
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', 'evaluation', 'DTU', 'eval_code')))
# sys.path.append(os.path.abspath(os.path.join(__file__, '..', 'evaluation', 'DTU', 'eval_code')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "gaussian-splatting")))


from poc_run_single import run_single
from poc_run_dtu import move_dir
from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.eval_utils import prepare_eval, write_to_csv, create_strings

from evaluate_single_scene import cull_scan
device = "cuda" if torch.cuda.is_available() else "cpu"
base_dir = os.path.abspath(os.getcwd())

# =============================================================================
#  Run
# =============================================================================
test_scans_nums = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

test_scans_by_class = {
    "DLNR_Finetuned_Full" : test_scans_nums,
    "DLNR_Finetuned_Figures" : [55, 69, 83, 105, 106, 110, 114, 118, 122],
    "DLNR_Finetuned_Food" : [63, 97],
    "DLNR_Finetuned_Buildings" : [24],
    "DLNR_Finetuned_Hardware_Materials" : [37, 40],
    "DLNR_Finetuned_Body_Parts" : [65],
}


def all_train_scans():
    all_scans = chain(range(1, 78), range(82, 129))
    return [scan for scan in all_scans if scan not in test_scans_nums]

def run_icp(src_mesh_file, gt_dataset_dir, scan, icp_result_mesh_file):
    print(f"---Start ICP for Scan {scan}---")
    ms = ml.MeshSet()

    # Load target mesh
    print(f"loading GT mesh")
    ms.load_new_mesh(f'{gt_dataset_dir}/Points/stl/stl{scan:03}_total.ply')
    
    # Load source and target point clouds
    print(f"loading cull scan result mesh")
    ms.load_new_mesh(src_mesh_file)
    
    # apply icp:
    # ms.apply_filter('icp_between_meshes')
    matrix = ms.apply_filter('compute_matrix_by_icp_between_meshes')

    # Extract the transformation matrix from the ICP computation
    matrix = ms[1].trasform_matrix()

    # Convert the matrix to a NumPy array
    icp_matrix = np.array(matrix).reshape(4, 4)  # 4x4 transformation matrix
    print("ICP Transformation Matrix:")
    print(icp_matrix)

    # Apply the transformation matrix to the source mesh
    # ms.apply_filter("transform_matrix",
    #     transformationmatrix=icp_matrix.flatten().tolist(),  # Convert 4x4 to 1D list
    #     applyto=1  # Apply to the source mesh
    # )
    ms.apply_filter("apply_matrix_freeze")

    # save aligned mesh
    ms.save_current_mesh(icp_result_mesh_file)

def create_mesh_and_eval(args, scan_num, exp_path, dataset_string, Offical_DTU_Dataset, csv_file):
    # =============================================================================
    #  Set Scan Args
    # =============================================================================

    print(f"----START PROCESSING SCAN {scan_num}----")
    args.colmap_name = f'scan{scan_num}'
    args.GS_port = GS_port_orig + scan_num
    print(args.colmap_name)

    strings = create_strings(args)
    output_dir_root = strings["output_dir_root"]
    output_for_eval_dir_root = output_dir_root.replace("output", "output_for_eval")
    splatting_output_dir_root = os.path.join(base_dir, 'splatting_output', strings['splatting'], args.colmap_name)
    splatting_output_for_eval_dir_root = splatting_output_dir_root.replace("output", "output_for_eval")

    # =============================================================================
    #  Create mesh
    # =============================================================================

    # if didn't ran the GS, there will be no splatting_output dir, 
    # but there may be one from previous run in the splatting_output_for_eval dir so we move it to the output dir
    if (not os.path.exists(splatting_output_dir_root)) and (args.skip_GS) and (os.path.exists(splatting_output_for_eval_dir_root)):
            move_dir(splatting_output_for_eval_dir_root, splatting_output_dir_root)

    ply_file = run_single(args)

    # =============================================================================
    #  Evaluate
    # =============================================================================
    
    # if didn't ran the create mesh, there will be no output dir, 
    # but there may be one from previous run in the output_for_eval dir so we move it to the output dir
    if (not os.path.exists(output_dir_root)) and (os.path.exists(output_for_eval_dir_root)):
            move_dir(output_for_eval_dir_root, output_dir_root)

    # =============================================================================
    #  Evaluate Before ICP
    # =============================================================================

    out_dir = os.path.join(exp_path, str(scan_num), "without_ICP")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    result_mesh_file = os.path.join(out_dir, f"{dataset_string}_scan{scan_num}.ply")
    cull_scan(scan_num, ply_file, result_mesh_file, Offical_DTU_Dataset)

    cmd = f"python {os.path.join(os.getcwd(), 'evaluation', 'DTU', 'eval_code', 'eval.py')} --data {result_mesh_file} --scan {scan_num} --mode mesh --dataset_dir {Offical_DTU_Dataset} --vis_out_dir {out_dir}"
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    output = output.replace(" ", ",").split(",")
    output[-1] = output[-1].strip()
    output = [scan_num] + output
    
    write_to_csv(args.dataset_name, csv_file, output)

    # =============================================================================
    #  Evaluate After ICP
    # =============================================================================
    icp_out_dir = os.path.join(exp_path, str(scan_num), "with_ICP")
    Path(icp_out_dir).mkdir(parents=True, exist_ok=True)
    icp_result_mesh_file = os.path.join(icp_out_dir, f"{dataset_string}_scan{scan_num}_ICP.ply")

    run_icp(result_mesh_file, Offical_DTU_Dataset, scan_num, icp_result_mesh_file)

    cmd = f"python {os.path.join(os.getcwd(), 'evaluation', 'DTU', 'eval_code', 'eval.py')} --data {icp_result_mesh_file} --scan {scan_num} --mode mesh --dataset_dir {Offical_DTU_Dataset} --vis_out_dir {icp_out_dir}"
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    output = output.replace(" ", ",").split(",")
    output[-1] = output[-1].strip()
    output = [f"{scan_num}_ICP"] + output
    
    write_to_csv(args.dataset_name, csv_file, output)

    # =============================================================================
    #  Move Outputs Folders to Avoid Clashes
    # =============================================================================
    
    move_dir(output_dir_root, output_for_eval_dir_root)
    move_dir(splatting_output_dir_root, splatting_output_for_eval_dir_root)


def run_DTU_eval(args):

    # =============================================================================
    #  Set arguments
    # =============================================================================

    Offical_DTU_Dataset = os.path.join(os.getcwd(), 'data_for_eval', 'DTU', 'SampleSet', 'MVS_Data')

    args.data_root_dir = "data_for_eval"
    args.dataset_name = "DTU"

    args.stereo_warm = False
    args.renderer_save_json = True
    args.masker_automask = True
    args.masker_SAM2_local = False
    args.masker_prompt = 'main_object'
    skip_GS = args.skip_GS

    # =============================================================================
    #  Create meshes and evaluate
    # =============================================================================
    for scan_num in args.scans:
        #run and eval original model
        args.stereo_model = "DLNR_Middlebury"
        args.skip_GS = skip_GS
        dataset_string, exp_path, csv_file = prepare_eval(args)
        create_mesh_and_eval(args, scan_num, exp_path, dataset_string, Offical_DTU_Dataset, csv_file)

        #run and eval finetuned model
        args.stereo_model = f"DLNR_Finetuned_scan{scan_num}"
        args.skip_GS = True
        dataset_string, exp_path, csv_file = prepare_eval(args)
        create_mesh_and_eval(args, scan_num, exp_path, dataset_string, Offical_DTU_Dataset, csv_file)
        


# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser('DTU')
    args = parser.parse_args()
    GS_port_orig = args.GS_port

    run_DTU_eval(args)
