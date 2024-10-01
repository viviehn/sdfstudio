#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import json
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import mediapy as media
import numpy as np
import torch
import tyro
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from typing_extensions import Literal, assert_never

from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn

def main(args):
    """
    Given data that follows the nerfstudio format such as the output from colmap or polycam,
    convert to a format that sdfstudio will ingest
    """
    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)

    _, pipeline, _ = eval_setup(
        self.load_config,
        eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
        test_mode="test" if self.traj == "spiral" else "inference",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load model and export features"
                                                 )

    parser.add_argument("--load-config", dest="load_config", required=True, help="path to config YAML file")
    parser.add_argument("--output-path", dest="output_path", required=True, help="path to output filename")

    args = parser.parse_args()

    main(args)
