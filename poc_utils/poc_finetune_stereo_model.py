# =============================================================================
#  Imports
# =============================================================================

from argparse import Namespace
import os
from os import path as osp
import torch
import json
import sys
from glob import glob
import numpy as np
from tqdm import tqdm
import shutil
from itertools import chain


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(osp.abspath(osp.join(__file__, "..", "..", "third_party", "DLNR", "core")))
sys.path.append(osp.abspath(osp.join(__file__, "..", "..", "third_party", "DLNR")))

from gs2mesh_utils.eval_utils import create_strings
from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.eval_utils import prepare_eval, write_to_csv
from third_party.DLNR.core.utils.frame_utils import readPFM, writePFM

from third_party.DLNR.core.dlnr import DLNR
from third_party.DLNR.core.stereo_datasets import StereoDataset
from third_party.DLNR.core.utils.utils import InputPadder as DLNR_InputPadder
from third_party.DLNR.train_stereo import train
from poc_show_train_results import plot_loss
# from poc_run_dtu import test_scans_nums, all_train_scans

device = "cuda" if torch.cuda.is_available() else "cpu"
base_dir = osp.abspath(os.getcwd())

test_scans_nums = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
def all_train_scans():
    all_scans = chain(range(1, 78), range(82, 129))
    return [scan for scan in all_scans if scan not in test_scans_nums]

train_scans_by_class = {
    "DLNR_Finetuned_Full" : all_train_scans(),
    "DLNR_Finetuned_Figures" : [2, 3, 4, 7, 33, 49, 50, 56, 57, 58, 70, 71, 72, 82, 84, 103, 107, 108, 109, 111, 112, 113, 115, 116, 117, 119, 120, 121, 123, 124, 125],
    "DLNR_Finetuned_Food" : [5, 12, 30, 31, 32, 42, 45, 59, 60, 61, 64, 74, 75, 76, 93, 94, 95, 96, 97, 99, 100],
    "DLNR_Finetuned_Buildings" : [6, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 43, 44, 46, 73],
    "DLNR_Finetuned_Hardware_Materials" : [10, 13, 34, 35, 36, 38, 39, 126, 127, 128],
    "DLNR_Finetuned_Body_Parts" : [52, 53, 66, 67, 68, 85, 86, 87, 88, 89, 90, 91, 92 ],
}

def load_dlnr_finetune_args(args):
    # =============================================================================
    #  Load dlnr stereo model and weights
    # =============================================================================
    restore_ckpt = osp.join(base_dir, "third_party", "DLNR", "pretrained", "DLNR_Middlebury.pth")
    start_num_steps = 0

    checkpoint_dir = osp.join(base_dir, "checkpoints", args.trained_model_name)
    if osp.exists(checkpoint_dir):
        checkpoint_files = os.listdir(checkpoint_dir)
        if len(checkpoint_files) > 0:
            checkpoint_file = checkpoint_files[-1]
            restore_ckpt =  osp.join(checkpoint_dir, checkpoint_file)
            start_num_steps = int(checkpoint_file.split('_')[0])
    
    lr = 0.0002
    if osp.exists(f"runs/{args.trained_model_name}/lr.txt"):
        with open(f"runs/{args.trained_model_name}/lr.txt", "r") as file:
            lr = float(file.read())
            assert(type(lr) == float)

    DLNR_args = Namespace(
        name=args.trained_model_name,
        restore_ckpt=restore_ckpt,
        mixed_precision=True,
        batch_size=8,
        train_datasets=["gs2mesh_ds"],
        lr=lr,
        num_steps=40000,
        start_num_steps = start_num_steps,
        image_size=[384, 736],
        train_iters=22,
        wdecay=0.00001,
        valid_iters=32,
        corr_implementation="reg_cuda",
        shared_backbone=False,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        slow_fast_gru=False,
        n_gru_layers=3,
        hidden_dims=[128] * 3,
        img_gamma=None,
        saturation_range=[0, 1.4],
        do_flip=False,
        spatial_scale=[-0.2, 0.4],
        noyjitter=False,
        dataset="gs2mesh_ds",
        scans = args.scans
    )

    return DLNR_args


def finetune_stereo_model(args):
    # =============================================================================
    #  Load model and DS
    # =============================================================================
    if (args.trained_model_name in train_scans_by_class):
        args.scans = train_scans_by_class[args.trained_model_name]
    if ("DLNR_Finetuned_Full" in args.trained_model_name):
        args.scans = all_train_scans()

    args.dataset_name = "DTU_test" if args.scans[0] in test_scans_nums else "DTU_train"
    args.stereo_model = "DLNR_Finetuned"

    DLNR_args = load_dlnr_finetune_args(args)
    print(DLNR_args.scans)
    checkpoints_path, event_file_path = train(DLNR_args)

    # copy last checkpoint model to pretrained folder
    pretraind_path = osp.join(base_dir, "third_party", "DLNR", "pretrained", (f"{args.trained_model_name}.pth"))
    shutil.copy2(checkpoints_path, pretraind_path)

    # =============================================================================
    #  Plot Loss Graph
    # =============================================================================

    # rename event file
    new_event_file_dir_name = os.path.join('runs', args.trained_model_name)
    new_event_file_path = os.path.join(new_event_file_dir_name, os.path.basename(event_file_path))
    os.makedirs(new_event_file_dir_name, exist_ok=True)
    shutil.move(event_file_path, new_event_file_dir_name)

    plot_loss(new_event_file_path, args.trained_model_name)


# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser("DTU")
    args = parser.parse_args()
    finetune_stereo_model(args)
