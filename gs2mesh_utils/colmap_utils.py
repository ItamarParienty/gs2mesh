# =============================================================================
#  Imports
# =============================================================================

import torch
import numpy as np
import os
from tqdm import tqdm
import cv2
import shutil
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
import plotly.graph_objects as go
import trimesh

from gs2mesh_utils.io_utils import read_ply
import gs2mesh_utils.third_party.visualization.visualize as visualize
import gs2mesh_utils.third_party.visualization.camera_utils as camera_utils
from gs2mesh_utils.third_party.colmap_runner.utils.read_write_model import read_images_text, read_points3D_text

# =============================================================================
#  Functions
# =============================================================================

def poses_from_file(extrinsic_file):
    """
    Read camera extrinsics from a COLMAP images.txt file and return them as a torch tensor.

    Parameters:
    extrinsic_file (str): Path to the COLMAP images.txt file.

    Returns:
    torch.Tensor: Camera poses as a tensor.
    """
    extrinsics = read_images_text(extrinsic_file)
    images = OrderedDict(sorted(extrinsics.items()))
    qvecs = torch.from_numpy(np.stack([image.qvec for image in images.values()]))
    tvecs = torch.from_numpy(np.stack([image.tvec for image in images.values()]))
    Rs = camera_utils.quaternion.q_to_R(qvecs)
    poses = torch.cat([Rs, tvecs[..., None]], dim=-1)
    return poses

def matrix_to_quaternion(rotation_matrix):
    """
    Convert a rotation matrix to a quaternion.

    Parameters:
    rotation_matrix (np.ndarray): Rotation matrix.

    Returns:
    np.ndarray: Quaternion.
    """
    return R.from_matrix(rotation_matrix).as_quat()

def quaternion_to_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.

    Parameters:
    quaternion (np.ndarray): Quaternion.

    Returns:
    np.ndarray: Rotation matrix.
    """
    return R.from_quat(quaternion).as_matrix()

def extract_frames(video_path, output_folder, interval=20, verbose=True):
    """
    Extract frames from a video at a specified sampling interval.

    Parameters:
    video_path (str): Path to the video file.
    output_folder (str): Output directory to save the extracted frames.
    interval (int): Sampling interval for frame extraction (save every n-th frame from the video).
    verbose (bool): Flag to print video information.

    Returns:
    None
    """
    if not os.path.exists(output_folder):
        if verbose:
            print(f"Creating output folder {output_folder}")
        os.makedirs(output_folder)
    else:
        if verbose:
            print(f"Output folder {output_folder} exists")

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if verbose:
        print(f"Video resolution: {frame_width}x{frame_height}, FPS: {fps}")
        
    success, image = vidcap.read()
    count = 0

    if verbose:
        print("Extracting frames...")
    while success:
        if count % interval == 0:
            cv2.imwrite(os.path.join(output_folder, f"IMG_{count:05}.png"), image)
        success, image = vidcap.read()
        count += 1
    if verbose:
        print("Done extracting frames")

def visualize_colmap_poses(colmap_dir, depth_scale=1, subsample=100, visualize_points=True, GT_path=None):
    """
    Visualize COLMAP poses and sparse SfM points, and optionally compare to a ground truth point cloud.

    Parameters:
    colmap_dir (str): Directory containing COLMAP sparse model.
    depth_scale (float): Adjust according to scale of the scene, in order to see the cameras.
    subsample (int): Subsampling factor to reduce the number of visualized points. Use a larger value if your mesh is large.
    visualize_points (bool): Flag to visualize the sparse SfM points.
    GT_path (str): Path to the ground truth point cloud, in case one exists and is aligned for comparison.

    Returns:
    None
    """
    vis_depth = 0.02 * depth_scale
    poses = poses_from_file(os.path.join(colmap_dir, 'sparse', '0', 'images.txt'))
    traces_poses = visualize.plotly_visualize_pose(poses, vis_depth=vis_depth, xyz_length=0.02, center_size=0.01, xyz_width=0.005, mesh_opacity=0.05)
    
    trace_points = None
    if visualize_points:
        points3d = read_points3D_text(os.path.join(colmap_dir, 'sparse', '0', 'points3D.txt'))
        xyz = torch.stack([torch.tensor(point3d.xyz) for point3d in points3d.values()], axis=0)
        x, y, z = *xyz.T,
        trace_points = go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=1, opacity=1), hoverinfo="skip", name="COLMAP")

    trace_points_GT = None
    if GT_path is not None:
        if '.obj' in GT_path:
            xyz_GT = torch.tensor(trimesh.load(GT_path).vertices)
        else:
            xyz_GT, _ = read_ply(GT_path)
        xyz_GT = xyz_GT[::subsample]
        x_GT, y_GT, z_GT = *xyz_GT.T,
        trace_points_GT = go.Scatter3d(x=x_GT, y=y_GT, z=z_GT, mode="markers", marker=dict(size=1, opacity=1), hoverinfo="skip", name="GT")

    point_traces = []
    if trace_points is not None:
        point_traces.append(trace_points)
    if trace_points_GT is not None:
        point_traces.append(trace_points_GT)
    
    traces_all = traces_poses
    if len(point_traces) > 0:
        traces_all = traces_poses + point_traces
    layout = go.Layout(scene=dict(xaxis=dict(showspikes=False, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0.1)"),
                                  yaxis=dict(showspikes=False, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0.1)"),
                                  zaxis=dict(showspikes=False, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0.1)"),
                                  xaxis_title="X", yaxis_title="Y", zaxis_title="Z", dragmode="orbit",
                                  aspectratio=dict(x=1, y=1, z=1), aspectmode="data"), height=800)
    fig = go.Figure(data=traces_all, layout=layout)
    
    fig.show()

def convert_to_txt(colmap_dir):
    """
    Convert colmap sparse model from .bin to .txt

    Parameters:
    colmap_dir (str): Output directory for the COLMAP sparse model.

    Returns:
    None
    """
    os.system(f"colmap model_converter --input_path {os.path.join(colmap_dir, 'sparse', '0')} --output_path {os.path.join(colmap_dir, 'sparse', '0')} --output_type TXT")
    
    
def run_colmap(colmap_dir, use_gpu=True):
    """
    Run COLMAP on a directory of images with unknown poses to create a sparse model.

    Parameters:
    colmap_dir (str): Output directory for the COLMAP sparse model.
    use_gpu (bool): Flag to use GPU for feature extraction and matching.

    Returns:
    None
    """
    os.system(f"rm -rf {os.path.join(colmap_dir, 'images', '.ipynb_checkpoints')}")
    os.system(f"colmap feature_extractor --database_path {os.path.join(colmap_dir, 'database.db')} --image_path {os.path.join(colmap_dir, 'images')} --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --SiftExtraction.use_gpu {'1' if use_gpu else '0'}")
    os.system(f"colmap exhaustive_matcher --database_path {os.path.join(colmap_dir, 'database.db')} --SiftMatching.use_gpu {'1' if use_gpu else '0'}")
    sparse = os.path.join(colmap_dir, 'sparse')
    if not os.path.exists(sparse):
        os.makedirs(sparse)
    os.system(f"colmap mapper --database_path {os.path.join(colmap_dir, 'database.db')} --image_path {os.path.join(colmap_dir, 'images')} --output_path {os.path.join(colmap_dir, 'sparse')} --Mapper.num_threads 16 --Mapper.init_min_tri_angle 4 --Mapper.multiple_models 0 --Mapper.extract_colors 0")
    convert_to_txt(colmap_dir)

def run_colmap_known_poses(colmap_dir, use_gpu=True, images_dir_name='images'):
    """
    Run COLMAP on a directory of images with known poses to create a sparse model.

    Parameters:
    colmap_dir (str): Directory containing a COLMAP sparse model with known poses.
    use_gpu (bool): Flag to use GPU for feature extraction and matching.
    images_dir_name (str): Name of the directory containing images. Images should be in the path: .../colmap_dir/images_dir_name

    Returns:
    None
    """
    os.system(f"rm -rf {os.path.join(colmap_dir, images_dir_name, '.ipynb_checkpoints')}")
    os.system(f"colmap feature_extractor --database_path {os.path.join(colmap_dir, 'database.db')} --image_path {os.path.join(colmap_dir, images_dir_name)} --SiftExtraction.use_gpu {'1' if use_gpu else '0'} --ImageReader.camera_model PINHOLE")
    os.system(f"colmap exhaustive_matcher --database_path {os.path.join(colmap_dir, 'database.db')} --SiftMatching.use_gpu {'1' if use_gpu else '0'}")
    os.system(f"colmap point_triangulator --clear_points 1 --database_path {os.path.join(colmap_dir, 'database.db')} --image_path {os.path.join(colmap_dir, images_dir_name)} --input_path {os.path.join(colmap_dir, 'sparse', '0')} --output_path {os.path.join(colmap_dir, 'sparse', '0')}")
    convert_to_txt(colmap_dir)
    
def create_mobile_brick_colmap_files(orig_dir, colmap_name):
    """
    Preprocess a scan in the MobileBrick dataset to create an empty COLMAP sparse model with known poses..

    Parameters:
    orig_dir (str): Original directory containing the scan in the MobileBrick dataset.
    colmap_name (str): Name of the scan.

    Returns:
    None
    """
    sparse_folder = os.path.join(orig_dir, 'sparse', '0')
    os.makedirs(sparse_folder, exist_ok=True)

    extrinsics_dir = os.path.join(orig_dir, 'pose')
    intrinsics_dir = os.path.join(orig_dir, 'intrinsic')
    images_dir = os.path.join(orig_dir, 'images')
    os.system(f"rm -rf {os.path.join(images_dir,'.ipynb_checkpoints')}")
    extrinsics_files = sorted([f for f in os.listdir(extrinsics_dir) if os.path.isfile(os.path.join(extrinsics_dir, f))], key=lambda x: x) 
    intrinsics_files = sorted([f for f in os.listdir(intrinsics_dir) if os.path.isfile(os.path.join(intrinsics_dir, f))], key=lambda x: x)    
    image_files = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))], key=lambda x: x)    
    
    images_txt_path = os.path.join(sparse_folder, "images.txt")
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i, (extrinsic_file, image_file) in enumerate(zip(extrinsics_files, image_files)):
            extrinsic = np.loadtxt(os.path.join(extrinsics_dir, extrinsic_file))
            extrinsic = np.linalg.inv(extrinsic)
            qx, qy, qz, qw, = matrix_to_quaternion(extrinsic[:3, :3])
            tx, ty, tz = extrinsic[:3, 3]
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i+1} {image_file}\n")
            f.write("\n")
    
    cameras_txt_path = os.path.join(sparse_folder, "cameras.txt")
    with open(cameras_txt_path, 'w') as cameras_file:
        cameras_file.write("# Camera list with one line of data per camera:\n")
        cameras_file.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (intrinsic_file, image_file) in enumerate(zip(extrinsics_files, image_files)):
            intrinsic = np.loadtxt(os.path.join(intrinsics_dir, intrinsic_file))
            cameras_file.write(f"{i+1} PINHOLE 1920 1440 {intrinsic[0, 0]} {intrinsic[1, 1]} {intrinsic[0, 2]} {intrinsic[1, 2]}\n")
    
    points3d_txt_path = os.path.join(sparse_folder, "points3D.txt")
    open(points3d_txt_path, 'w').close()
