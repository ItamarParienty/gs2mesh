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


from gs2mesh_utils.eval_utils import create_strings
from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.eval_utils import prepare_eval, write_to_csv
from third_party.DLNR.core.utils.frame_utils import readPFM, writePFM

sys.path.append(osp.abspath(osp.join(__file__, "../", "third_party", "DLNR")))
sys.path.append(osp.abspath(osp.join(__file__, "../", "third_party", "DLNR", "core")))
from third_party.DLNR.core.dlnr import DLNR
from third_party.DLNR.core.stereo_datasets import StereoDataset
from third_party.DLNR.core.utils.utils import InputPadder as DLNR_InputPadder
from third_party.DLNR.train_stereo import train

device = "cuda" if torch.cuda.is_available() else "cpu"
base_dir = osp.abspath(os.getcwd())


# =============================================================================
#  Run
# =============================================================================


def load_dlnr_model(args):
    # =============================================================================
    #  Load dlnr stereo model and weights
    # =============================================================================
    DLNR_args = Namespace(
        name="DLNR_Finetuned",
        restore_ckpt=osp.join(base_dir, "third_party", "DLNR", "pretrained", "DLNR_Middlebury.pth"),
        mixed_precision=True,
        batch_size=8,
        train_datasets=["gs2mesh_ds"],
        lr=2e-5,
        num_steps=10000,
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
    )

    # DLNR_model = torch.nn.DataParallel(DLNR(DLNR_args), device_ids=[0])
    # DLNR_model.load_state_dict(torch.load(DLNR_args.restore_ckpt))
    # DLNR_model = DLNR_model.module
    # DLNR_model.to(device)
    # DLNR_model.eval()

    # return DLNR_model, DLNR_args
    return DLNR_args
    # self.disparity_signs = {'DLNR_Middlebury': -1, 'DLNR_SceneFlow': -1}
    # self.model = DLNR_model
    # self.input_padder = DLNR_InputPadder
    # self.model_args = DLNR_args


def finetune_stereo_model(args):
    # =============================================================================
    #  Load model and DS
    # =============================================================================
    args.dataset_name = "DTU_test"
    args.stereo_model = "DLNR_Finetuned"

    DLNR_args = load_dlnr_model(args)
    checkpoints_path = train(DLNR_args)


# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser("DTU")
    args = parser.parse_args()
    finetune_stereo_model(args)
