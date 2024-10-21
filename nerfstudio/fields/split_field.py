"""
Field for split model (separated geom and appearance encodings)
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
from nerfstudio.encoding import get_encoder

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass

from pdb import set_trace as pause

@dataclass
class SplitFieldConfig(SDFFieldConfig):
    _target: Type = field(default_factory=lambda: SplitField)
    pop_appearance_net: bool = False
    """whether to re-use pretrained appearance network"""
    fix_appearance_net: bool = False
    """whether to fix pretrained appearance network; e.g. not pop + not fix = finetune"""
    pop_appearance_encoding: bool = True
    """whether to train geometry encodings from scratch"""
    fix_appearance_encoding: bool = False
    """whether to fix pretrained appearance encodings"""

class SplitField(SDFField):
    """_summary_

    Args:
        Field (_type_): _description_
    """

    config: SplitFieldConfig

    def __init__(
        self,
        config: SplitFieldConfig,
        aabb,
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        build_encoders: bool = True,
        **kwargs
    ) -> None:
        super().__init__(config, aabb, num_images, use_average_appearance_embedding, spatial_distortion, build_color_network=False)

        if build_encoders:
            geom_encoding, in_dim = get_encoder(  #encoding,
                "hashgrid",
                input_dim=3,
                multires=6,
                degree=4,
                num_levels=self.num_levels, level_dim=self.features_per_level,
                base_resolution=self.base_res, log2_hashmap_size=self.log2_hashmap_size,
                desired_resolution=self.max_res,
                align_corners=False,
                )
            appearance_encoding, in_dim = get_encoder(  #encoding,
                "hashgrid",
                input_dim=3,
                multires=6,
                degree=4,
                num_levels=self.num_levels, level_dim=self.features_per_level,
                base_resolution=self.base_res, log2_hashmap_size=self.log2_hashmap_size,
                desired_resolution=self.max_res,
                align_corners=False,
                )

            self.encoder_dict = nn.ModuleDict({
                    'geometry': geom_encoding,
                    'appearance': appearance_encoding,
                    })
            self.geometry_encoding = self.encoder_dict['geometry']
            self.appearance_encoding = self.encoder_dict['appearance']
            self.encoding = None


        self.color_in_dim = self.color_in_dim + self.num_levels*self.features_per_level
        self.build_color_network(self.color_in_dim)

    def get_colors(self, points, directions, gradients, geo_features, camera_indices):
        # diffuse color and specular tint
        if self.config.use_diffuse_color:
            raw_rgb_diffuse = self.diffuse_color_pred(geo_features.view(-1, self.config.geo_feat_dim))
        if self.config.use_specular_tint:
            tint = self.sigmoid(self.specular_tint_pred(geo_features.view(-1, self.config.geo_feat_dim)))

        normals = F.normalize(gradients, p=2, dim=-1)

        if self.config.use_reflections:
            # https://github.com/google-research/multinerf/blob/5d4c82831a9b94a87efada2eee6a993d530c4226/internal/ref_utils.py#L22
            refdirs = 2.0 * torch.sum(normals * -directions, axis=-1, keepdims=True) * normals + directions
            d = self.direction_encoding(refdirs)
        else:
            d = self.direction_encoding(directions)

        if self.config.vanilla_ngp:
            positions = points * 1.0
            # positions = inputs / 2
        else:
            positions = (points + 2.0) / 4.0
        appearance_feature = self.appearance_encoding(positions)
        if not self.config.vanilla_ngp:
            appearance_feature = appearance_feature * self.hash_encoding_mask.to(appearance_feature.device)

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
            # set it to zero if don't use it
            if not self.config.use_appearance_embedding:
                embedded_appearance = torch.zeros_like(embedded_appearance)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                )
        if self.config.use_diffuse_color:
            h = [
                d,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
                appearance_feature.view(-1, self.num_levels*self.features_per_level)
            ]
        else:
            h = [
                points,
                d,
                gradients,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
                appearance_feature.view(-1, self.num_levels*self.features_per_level)
            ]

        if self.config.use_n_dot_v:
            n_dot_v = torch.sum(normals * directions, dim=-1, keepdims=True)
            h.append(n_dot_v)

        h = torch.cat(h, dim=-1)

        for l in range(0, self.num_layers_color - 1):
            lin = getattr(self, "clin" + str(l))

            h = lin(h)

            if l < self.num_layers_color - 2:
                h = self.relu(h)

        rgb = self.sigmoid(h)

        if self.config.use_diffuse_color:
            # Initialize linear diffuse color around 0.25, so that the combined
            # linear color is initialized around 0.5.
            diffuse_linear = self.sigmoid(raw_rgb_diffuse - math.log(3.0))
            if self.config.use_specular_tint:
                specular_linear = tint * rgb
            else:
                specular_linear = 0.5 * rgb

            # TODO linear to srgb?
            # Combine specular and diffuse components and tone map to sRGB.
            rgb = torch.clamp(specular_linear + diffuse_linear, 0.0, 1.0)

        # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
        rgb = rgb * (1 + 2 * self.config.rgb_padding) - self.config.rgb_padding

        return rgb
