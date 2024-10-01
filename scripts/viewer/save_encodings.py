from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from threading import Lock
from typing import Literal
from pdb import set_trace as pause
import os

import tyro

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.base_config import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
#from nerfstudio.viewer.viewer import Viewer as ViewerState
#from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState
from nerfstudio.viewer.server.viewer_utils import ViewerState as ViewerLegacyState
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.configs import base_config as cfg
from nerfstudio.pipelines.base_pipeline import Pipeline
from pdb import set_trace as pause
import glob

from pathlib import Path
import argparse
import numpy as np

import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--load_config', type=str, default='config.yml', help='input config (default: config.yml)')
parser.add_argument('--output_path', type=str, default='encodings.npy', help='output npy file (default: encodings.npy)')
parser.add_argument('--sample_points', type=str, help='path to input sample points (usually data-id_msh.ply)')
parser.add_argument('--by_level', action='store_true', help='save out encodings by hashtable resolution level')
parser.add_argument('--encoding_id', type=int, help='if multiscene, specify which dataset')

args = parser.parse_args()
config_path = Path(args.load_config)
sample_points_file = args.sample_points
output_path = args.output_path


config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
# TODO set include_sdf_samples to True
assert isinstance(config, cfg.Config)

config.trainer.load_dir = config.get_checkpoint_dir()
if not os.path.exists(config.trainer.load_dir):
    config.trainer.load_dir = config_path.parents[0] / config.trainer.relative_model_dir
config.pipeline.datamanager.eval_image_indices = None

# setup pipeline (which includes the DataManager)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('setup pipeline')
pipeline = config.pipeline.setup(device=device, test_mode='inference')

assert isinstance(pipeline, Pipeline)
pipeline.eval()

# load checkpoint
assert config.trainer.load_dir is not None
if config.trainer.load_step is None:
    print("Loading latest checkpoint from load_dir")
    if not os.path.exists(config.trainer.load_dir):
        rule("Error", style="red")
        print(f"No checkpoint directory found at {config.load_dir}, ", justify="center")
        print(
            "Please make sure the checkpoint exists, they should be generated periodically during training",
            justify="center",
        )
        sys.exit(1)
    load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.trainer.load_dir))[-1]
else:
    load_step = config.trainer.load_step
load_path = config.trainer.load_dir / f"step-{load_step:09d}.ckpt"
assert load_path.exists(), f"Checkpoint {load_path} does not exist"
loaded_state = torch.load(load_path, map_location="cpu")

pipeline.load_pipeline(loaded_state["pipeline"])

# forward model

inputs = pipeline.datamanager.dataparser.load_sdf_samples_from_path(sample_points_file)
encoding_id = args.encoding_id
if encoding_id:
    inputs = pipeline.datamanager.dataparser_list[encoding_id].load_sdf_samples_from_path(sample_points_file)




#mesh = trimesh.load(sample_points_file, process=False)
with torch.no_grad():
    #sdf_onsurface = torch.from_numpy(mesh.vertices).float().to(device)
    print(inputs.shape)


    inputs = inputs[0,:,:3].to(device)
    positions = inputs * 1.
    if encoding_id:
        encoding = pipeline.model.field.encodings_list[encoding_id]
    else:
        encoding = pipeline.model.field.encoding
    feature = encoding(positions)
    if not pipeline.model.field.config.vanilla_ngp:
        feature = feature * pipeline.model.field.hash_encoding_mask.to(feature.device)

    num_samples = 20000
    feature = feature.detach().cpu().numpy().astype(np.float16)
    #downsampled_indices = np.random.choice(len(feature), num_samples, replace=False)
    if args.by_level:
        indices = list(range(0,len(feature[0])+1,4))
        idx_pairs = list(zip(indices, indices[1:]))
        for level, idx_pair in enumerate(idx_pairs):
            #level_feature = feature[:, idx_pair[0]:idx_pair[1]][downsampled_indices]
            level_feature = feature[:, idx_pair[0]:idx_pair[1]]
            level_feature = level_feature / np.linalg.norm(level_feature, axis=1, keepdims=True)
            if encoding_id:
                np.save(f'{output_path[:-4]}_{encoding_id}_{level:02d}.npy', level_feature)
            else:
                np.save(f'{output_path[:-4]}_{level:02d}.npy', level_feature)

    print(feature[0:4, :])
    print(feature[-4:, :])
    feature_norms = np.linalg.norm(feature, axis=1)
    feature = feature / feature_norms[:,None]
    #feature = feature[downsampled_indices]
    if encoding_id:
        np.save(f'{output_path[:-4]}_{encoding_id}.npy', feature)
    else:
        np.save(f'{output_path[:-4]}.npy', feature)
