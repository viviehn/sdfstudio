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

for path in glob.glob('/n/fs/3d-indoor/sdfstudio_outputs/3d-indoor/subsets/rgb-only/*/neus-facto-angelo/*/config.yml'):
    try:
        print('running: ', path)
        config_path = Path(path)

        config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
        assert isinstance(config, cfg.Config)
# if eval_num_rays_per_chunk:
#     config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk
# load checkpoints from wherever they were saved
# TODO: expose the ability to choose an arbitrary checkpoint
        config.trainer.load_dir = config.get_checkpoint_dir()
        if not os.path.exists(config.trainer.load_dir):
            config.trainer.load_dir = config_path.parents[0] / config.trainer.relative_model_dir
        config.pipeline.datamanager.eval_image_indices = None
# setup pipeline (which includes the DataManager)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = config.pipeline.setup(device=device, test_mode='test')
        assert isinstance(pipeline, Pipeline)
        pipeline.eval()
# load checkpointed information
# checkpoint_path = eval_load_checkpoint(config.trainer, pipeline)

        assert config.trainer.load_dir is not None
        if config.trainer.load_step is None:
            # CONSOLE.print("Loading latest checkpoint from load_dir")
            # NOTE: this is specific to the checkpoint name format
            if not os.path.exists(config.trainer.load_dir):
                # otherwise:
                # CONSOLE.rule("Error", style="red")
                # CONSOLE.print(f"No checkpoint directory found at {config.load_dir}, ", justify="center")
                # CONSOLE.print(
                #     "Please make sure the checkpoint exists, they should be generated periodically during training",
                #     justify="center",
                # )
                sys.exit(1)
            load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.trainer.load_dir))[-1]
        else:
            load_step = config.trainer.load_step
        load_path = config.trainer.load_dir / f"step-{load_step:09d}.ckpt"
        assert load_path.exists(), f"Checkpoint {load_path} does not exist"
        loaded_state = torch.load(load_path, map_location="cpu")

        embeddings = loaded_state['pipeline']['_model.field.encoding.embeddings']
        offsets = loaded_state['pipeline']['_model.field.encoding.offsets']
        indices = [pair for pair in zip(offsets[:], offsets[1:])]
        feat_levels = [embeddings[idx[0]:idx[1]] for idx in indices]
        feat_dict = {}
        for i, feats in enumerate(feat_levels):
            feat_dict[i] = feats

        save_name = f'{config.experiment_name}_{config.pipeline.datamanager.train_num_images_to_sample_from}_{config_path.parent.name}.feat'
        torch.save(feat_dict,save_name)
    except:
        print('failed: ', path)
        continue
