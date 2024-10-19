"""
Field for multiscene split model (separated geom and appearance encodings)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import (
    NeRFEncoding,
    PeriodicVolumeEncoding,
    TensorVMEncoding,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldConfig
from nerfstudio.fields.sdf_field import SDFField, SDFFieldConfig
from nerfstudio.fields.split_field import SplitField, SplitFieldConfig
from nerfstudio.encoding import get_encoder

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass

from pdb import set_trace as pause

@dataclass
class MultisceneSplitFieldConfig(SplitFieldConfig):
    _target: Type = field(default_factory=lambda: MultisceneSplitField)

class MultisceneSplitField(SplitField):

    config: MultisceneSplitFieldConfig

    def __init__(
        self,
        config: SplitFieldConfig,
        aabb,
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        **kwargs
    ) -> None:
        super().__init__(config, aabb, num_images, use_average_appearance_embedding, spatial_distortion, build_encoders=False)
        self.scene_ids = kwargs['scene_ids']

        self.geometry_encodings = {}
        self.appearance_encodings = {}
        for scene_id in self.scene_ids:
            self.geometry_encodings[scene_id], in_dim = get_encoder(
                "hashgrid",
                input_dim=3,
                multires=6,
                degree=4,
                num_levels=self.num_levels, level_dim=self.features_per_level,
                base_resolution=self.base_res, log2_hashmap_size=self.log2_hashmap_size,
                desired_resolution=self.max_res,
                align_corners=False,
                )
            self.appearance_encodings[scene_id], in_dim = get_encoder(
                "hashgrid",
                input_dim=3,
                multires=6,
                degree=4,
                num_levels=self.num_levels, level_dim=self.features_per_level,
                base_resolution=self.base_res, log2_hashmap_size=self.log2_hashmap_size,
                desired_resolution=self.max_res,
                align_corners=False,
                )

        self.geometry_encodings = nn.ModuleDict(self.geometry_encodings)
        self.appearance_encodings = nn.ModuleDict(self.appearance_encodings)

        #self.geometry_encoding = self.geometry_encodings[self.scene_ids[0]]
        #self.appearance_encoding = self.appearance_encodings[self.scene_ids[0]]
        self.geometry_encoding, in_dim = get_encoder(
            "hashgrid",
            input_dim=3,
            multires=6,
            degree=4,
            num_levels=self.num_levels, level_dim=self.features_per_level,
            base_resolution=self.base_res, log2_hashmap_size=self.log2_hashmap_size,
            desired_resolution=self.max_res,
            align_corners=False,
            )
        self.appearance_encoding, in_dim = get_encoder(
            "hashgrid",
            input_dim=3,
            multires=6,
            degree=4,
            num_levels=self.num_levels, level_dim=self.features_per_level,
            base_resolution=self.base_res, log2_hashmap_size=self.log2_hashmap_size,
            desired_resolution=self.max_res,
            align_corners=False,
            )
        self.encoding = None

