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
from poc_run_dtu import test_scans_nums, all_train_scans

device = "cuda" if torch.cuda.is_available() else "cpu"
base_dir = osp.abspath(os.getcwd())

scans_by_class = {
    "DLNR_Finetuned_Full" : all_train_scans(),
    "figures" : [2, 3, 4, 7, 33]

}

def load_dlnr_model(args):
    # =============================================================================
    #  Load dlnr stereo model and weights
    # =============================================================================
    DLNR_args = Namespace(
        # name=f"DLNR_Finetuned_scan{'_'.join([str(scan) for scan in args.scans])}",
        name=args.trained_model_name,
        restore_ckpt=osp.join(base_dir, "third_party", "DLNR", "pretrained", "DLNR_Middlebury.pth"),
        mixed_precision=True,
        batch_size=4,
        train_datasets=["gs2mesh_ds"],
        lr=0.0001,
        # lr=2e-5,
        num_steps=4000,
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
    if (args.trained_model_name in scans_by_class):
        args.scans = scans_by_class[args.trained_model_name]

    args.dataset_name = "DTU_test" if args.scans[0] in test_scans_nums else "DTU_train"
    args.stereo_model = "DLNR_Finetuned"

    DLNR_args = load_dlnr_model(args)
    checkpoints_path, event_file_path = train(DLNR_args)

    # copy last checkpoint model to pretrained folder
    # pretraind_path = osp.join(base_dir, "third_party", "DLNR", "pretrained", (f"DLNR_Finetuned_scan{'_'.join([str(scan) for scan in args.scans])}.pth"))
    pretraind_path = osp.join(base_dir, "third_party", "DLNR", "pretrained", (f"{args.trained_model_name}.pth"))
    shutil.copy2(checkpoints_path, pretraind_path)

    # =============================================================================
    #  Plot Loss Graph
    # =============================================================================

    # rename event file
    # new_event_file_name = f"DLNR_Finetuned_scan{'_'.join([str(scan) for scan in args.scans])}"
    new_event_file_name = args.trained_model_name
    new_event_file_path = os.path.join(os.path.dirname(event_file_path), f"{new_event_file_name}.0")
    os.rename(event_file_path, new_event_file_path)

    plot_loss(new_event_file_name)


# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser("DTU")
    args = parser.parse_args()
    finetune_stereo_model(args)
