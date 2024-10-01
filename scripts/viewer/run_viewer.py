# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
Starts viewer in eval mode.
"""

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


@dataclass
class ViewerConfigWithoutNumRays(ViewerConfig):
    """Configuration for viewer instantiation"""

    num_rays_per_chunk: tyro.conf.Suppress[int] = -1

    def as_viewer_config(self):
        """Converts the instance to ViewerConfig"""
        return ViewerConfig(**{x.name: getattr(self, x.name) for x in fields(self)})


@dataclass
class RunViewer:
    """Load a checkpoint and start the viewer."""

    load_config: Path
    """Path to config YAML file."""
    viewer: ViewerConfigWithoutNumRays = field(default_factory=ViewerConfigWithoutNumRays)
    """Viewer configuration"""
    vis: Literal["viewer", "viewer_legacy"] = "viewer"
    """Type of viewer"""

    def main(self) -> None:
        """Main function."""
        config, pipeline, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )
        num_rays_per_chunk = config.viewer.num_rays_per_chunk
        assert self.viewer.num_rays_per_chunk == -1
        config.vis = self.vis
        #config.viewer = self.viewer.as_viewer_config()
        config.viewer.num_rays_per_chunk = num_rays_per_chunk

        if config.trainer.load_step is None:
            load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.trainer.load_dir))[-1]
        else:
            load_step = config.trainer.load_step

        _start_viewer(config, pipeline, load_step)

    def save_checkpoint(self, *args, **kwargs):
        """
        Mock method because we pass this instance to viewer_state.update_scene
        """


def _start_viewer(config: TrainerConfig, pipeline: Pipeline, step: int):
    """Starts the viewer

    Args:
        config: Configuration of pipeline to load
        pipeline: Pipeline instance of which to load weights
        step: Step at which the pipeline was saved
    """
    base_dir = config.trainer.load_dir.parents[0]
    viewer_log_path = base_dir / config.viewer.relative_log_filename
    banner_messages = None
    viewer_state = None
    viewer_callback_lock = Lock()
    #if config.vis == "viewer_legacy":
    viewer_state = ViewerLegacyState(
        config.viewer,
        log_filename=viewer_log_path,
    )
    banner_messages = [f"Legacy viewer at: {viewer_state.viewer_url}"]
    '''
    if config.vis == "viewer":
        viewer_state = ViewerState(
            config.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline.datamanager.get_datapath(),
            pipeline=pipeline,
            share=config.viewer.make_share_url,
            train_lock=viewer_callback_lock,
        )
        banner_messages = viewer_state.viewer_info
    '''

    # We don't need logging, but writer.GLOBAL_BUFFER needs to be populated
    config.logging.local_writer.enable = False
    writer.setup_local_writer(config.logging, max_iter=config.trainer.max_num_iterations, banner_messages=banner_messages)

    assert viewer_state and pipeline.datamanager.train_dataset
    viewer_state.init_scene(
        dataset=pipeline.datamanager.train_dataset,
        start_train=False
    )
    '''
    if isinstance(viewer_state, ViewerLegacyState):
        viewer_state.viser_server.set_training_state("completed")
    '''
    viewer_state.update_scene(step=step, trainer=None, graph=pipeline.model, num_rays_per_batch=config.pipeline.datamanager.train_num_rays_per_batch)
    while True:
        time.sleep(0.01)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[RunViewer]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(tyro.conf.FlagConversionOff[RunViewer])  # noqa
