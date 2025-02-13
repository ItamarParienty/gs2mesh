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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from poc_run_single import run_single
from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.eval_utils import prepare_eval, write_to_csv, create_strings

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', 'evaluation', 'DTU', 'eval_code')))
from evaluate_single_scene import cull_scan
base_dir = os.path.abspath(os.getcwd())

# =============================================================================
#  Run
# =============================================================================
test_scans_nums = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

def all_train_scans():
    all_scans = chain(range(1, 78), range(82, 129))
    return [scan for scan in all_scans if scan not in test_scans_nums]

def move_dir(src_dir_path, dst_dir_parent_path):
    if os.path.exists(dst_dir_parent_path):
            shutil.rmtree(dst_dir_parent_path)
    Path(dst_dir_parent_path).mkdir(parents=True, exist_ok=True)
    shutil.move(src_dir_path, dst_dir_parent_path)

def run_icp(src_mesh_file, gt_dataset_dir, scan, icp_result_mesh_file):
    print(f"---Start ICP for Scan {scan}---")

    # Load source and target point clouds
    print(f"loading cull scan result mesh")
    source_mesh = o3d.io.read_triangle_mesh(src_mesh_file)
    # source_pcd = source_mesh.sample_points_uniformly(number_of_points=10000)

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source_mesh.vertices))
    
    print(f"loading GT mesh")
    target_pcd = o3d.io.read_point_cloud(f'{gt_dataset_dir}/Points/stl/stl{scan:03}_total.ply')

    # Initial alignment
    trans_init = np.eye(4)

    # ICP Registration
    threshold = 0.005  # Distance threshold for point matching
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Apply transformation
    source_mesh.transform(icp_result.transformation)

    # Save output point cloud
    o3d.io.write_triangle_mesh(icp_result_mesh_file, source_mesh)



def run_DTU_eval(args):

    # =============================================================================
    #  Set arguments
    # =============================================================================

    Offical_DTU_Dataset = os.path.join(os.getcwd(), 'data_for_eval', 'DTU', 'SampleSet', 'MVS_Data')
    dataset_string, exp_path, csv_file = prepare_eval(args)

    args.data_root_dir = "data_for_eval"
    args.dataset_name = "DTU"
    # args.stereo_model = "DLNR_Finetuned"
    
    args.skip_colmap = True
    args.skip_GS = True
    # args.skip_rendering = True
    # args.skip_masking = True
    # args.skip_TSDF = True

    args.stereo_warm = False
    args.renderer_save_json = True
    args.masker_automask = True
    args.masker_SAM2_local = False
    args.masker_prompt = 'main_object'

    # =============================================================================
    #  Create meshes and evaluate
    # =============================================================================
    
    for scan_num in args.scans:
        # =============================================================================
        #  Set Scan Args
        # =============================================================================

        print(f"----START PROCESSING SCAN {scan_num}----")
        args.colmap_name = f'scan{scan_num}'
        args.GS_port = GS_port_orig + scan_num
        print(args.colmap_name)
        print(args)

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
             move_dir(splatting_output_for_eval_dir_root, os.path.dirname(splatting_output_dir_root))

        ply_file = run_single(args)

        # =============================================================================
        #  Evaluate
        # =============================================================================
        
        # if didn't ran the create mesh, there will be no output dir, 
        # but there may be one from previous run in the output_for_eval dir so we move it to the output dir
        if (not os.path.exists(output_dir_root)) and (os.path.exists(output_for_eval_dir_root)):
             move_dir(output_for_eval_dir_root, os.path.dirname(output_dir_root))

        # =============================================================================
        #  Evaluate Before ICP
        # =============================================================================

        out_dir = os.path.join(exp_path, str(scan_num), "without_ICP")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        vis_out_dir = os.path.join(exp_path, str(scan_num), "without_ICP")
        Path(vis_out_dir).mkdir(parents=True, exist_ok=True)
        result_mesh_file = os.path.join(out_dir, f"{dataset_string}_scan{scan_num}.ply")
        cull_scan(scan_num, ply_file, result_mesh_file, Offical_DTU_Dataset)

        cmd = f"python {os.path.join(os.getcwd(), 'evaluation', 'DTU', 'eval_code', 'eval.py')} --data {result_mesh_file} --scan {scan_num} --mode mesh --dataset_dir {Offical_DTU_Dataset} --vis_out_dir {vis_out_dir}"
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
        icp_vis_out_dir = os.path.join(exp_path, str(scan_num), "with_ICP")
        Path(icp_vis_out_dir).mkdir(parents=True, exist_ok=True)
        icp_result_mesh_file = os.path.join(icp_out_dir, f"{dataset_string}_scan{scan_num}_ICP.ply")

        #TODO: improve this function
        run_icp(result_mesh_file, Offical_DTU_Dataset, scan_num, icp_result_mesh_file)

        cmd = f"python {os.path.join(os.getcwd(), 'evaluation', 'DTU', 'eval_code', 'eval.py')} --data {icp_result_mesh_file} --scan {scan_num} --mode mesh --dataset_dir {Offical_DTU_Dataset} --vis_out_dir {icp_vis_out_dir}"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        output = output.replace(" ", ",").split(",")
        output[-1] = output[-1].strip()
        output = [f"{scan_num}_ICP"] + output
       
        write_to_csv(args.dataset_name, csv_file, output)

        # =============================================================================
        #  Move Outputs Folders to Avoid Clashes
        # =============================================================================
        
        move_dir(output_dir_root, os.path.dirname(output_for_eval_dir_root))
        move_dir(splatting_output_dir_root, os.path.dirname(splatting_output_for_eval_dir_root))


# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser('DTU')
    args = parser.parse_args()
    GS_port_orig = args.GS_port

    run_DTU_eval(args)
