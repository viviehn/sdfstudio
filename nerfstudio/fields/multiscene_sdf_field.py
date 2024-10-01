import math
from dataclasses import dataclass, field
from typing import Optional, Type, Union, List

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
class MultisceneSDFFieldConfig(SDFFieldConfig):
    _target: Type = field(default_factory=lambda: MultisceneSDFField)
    num_scenes: int = 3

class MultisceneSDFField(SDFField):

    def __init__(
        self,
        config: SDFFieldConfig,
        aabb,
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__(config, aabb, num_images,
                         use_average_appearance_embedding,
                         spatial_distortion)

        self.num_scenes = config.num_scenes

        assert self.config.encoding_type == "hash", "incompatible with multiscene training"
        assert self.config.vanilla_ngp, "incompatible with multiscene training"

        self.encodings_list = []
        for i in range(self.num_scenes):
            enc, in_dim = get_encoder(  #encoding,
                "hashgrid",
                input_dim=3,
                multires=6,
                degree=4,
                num_levels=self.num_levels, level_dim=self.features_per_level,
                base_resolution=self.base_res, log2_hashmap_size=self.log2_hashmap_size,
                desired_resolution=self.max_res,
                align_corners=False,
                )
            #enc.embeddings.requires_grad = False
            self.encodings_list.append(enc)

        self.encodings_list = nn.ModuleList(self.encodings_list)
        self.encoding = None


    def forward_geonetwork(self, inputs, encoding):
        """forward the geonetwork"""
        if self.use_grid_feature:
            if self.config.vanilla_ngp:
                positions = inputs * 1.0
            else:
                positions = (inputs + 2.0) / 4.0
            feature = encoding(positions)
            if not self.config.vanilla_ngp:
                feature = feature * self.hash_encoding_mask.to(feature.device)
        else:
            feature = torch.zeros_like(inputs[:, :1].repeat(1, encoding.n_output_dims))

        if not self.config.vanilla_ngp:
            pe = self.position_encoding(inputs)
            if not self.config.use_position_encoding:
                pe = torch.zeros_like(pe)

            inputs = torch.cat((inputs, pe, feature), dim=-1)
        else:
            inputs = feature.float()

        x = inputs

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "glin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            if self.config.vanilla_ngp and l == self.num_layers - 2:
                lin_geofeat = getattr(self, "glin" + str(l+1))
                x_geofeat = lin_geofeat(x)
            x = lin(x)

            if l < self.num_layers - 2:
                if self.config.vanilla_ngp:
                    x = self.relu(x)
                else:
                    x = self.softplus(x)
        if self.config.vanilla_ngp:
            x = torch.cat((x, x_geofeat), -1)
        # pause()
        return x

    def get_outputs(self, ray_samples: List, return_alphas=False, return_occupancy=False, sdf_only=False, need_rgb=False, scene_id=0):

        # Use the corresponding encoding
        encoding = self.encodings_list[scene_id]
        #encoding.embeddings.requires_grad = True

        if isinstance(ray_samples[0], torch.Tensor):
            # We are SDF training
            inputs = ray_samples[:, :3]
            # compute gradient in constracted space
            inputs.requires_grad_(True)
            with torch.enable_grad():
                h = self.forward_geonetwork(inputs, encoding)
                sdf, geo_feature = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
            outputs = {
                    FieldHeadNames.SDF: sdf,
                }
            if ray_samples.shape[1] == 4:
                if sdf_only:
                    return outputs

            if self.config.use_numerical_gradients:
                gradients, sampled_sdf = self.gradient(
                    inputs,
                    skip_spatial_distortion=True,
                    return_sdf=True,
                )
                sampled_sdf = sampled_sdf.permute(1, 0).contiguous()
            else:
                d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                gradients = torch.autograd.grad(
                    outputs=sdf,
                    inputs=inputs,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                sampled_sdf = None

            if ray_samples.shape[1] == 4 and need_rgb == False:
                outputs.update({
                        FieldHeadNames.GRADIENT: gradients,
                        "sampled_sdf": sampled_sdf,
                    })
                return outputs

            # For now, these code blocks suggest that the same MLP is used for RGB pred across scenes
            if ray_samples.shape[1] > 4:
                mask = (ray_samples[:, 3] == 0) & (ray_samples[:, 3:].sum(1) != 0)
                directions = ray_samples[:, 7:10][mask]
                noise = (torch.rand((mask.sum(), 3)).to(directions.device) - 0.5) * 0.1
                directions = F.normalize(directions + noise, p=2, dim=-1)
                camera_indices = torch.zeros_like(directions[:, 0]).int()
                rgb = self.get_colors(inputs[mask], directions, gradients[mask], geo_feature[mask], camera_indices)
            else:
                directions = gradients
                directions = F.normalize(directions, p=2, dim=-1)
                camera_indices = torch.zeros_like(directions[:, 0]).int()
                rgb = self.get_colors(inputs, directions, gradients, geo_feature, camera_indices)

            density = self.laplace_density(sdf)
            outputs.update({
                    # FieldHeadNames.SDF: sdf,
                    # FieldHeadNames.NORMAL: normals,
                    # FieldHeadNames.GRADIENT: gradients,
                    # "points_norm": points_norm,
                    # "sampled_sdf": sampled_sdf,
                    FieldHeadNames.RGB: rgb,
                    FieldHeadNames.DENSITY: density,
                })
            return outputs
        else:
            # We are not SDF training
            if ray_samples.camera_indices is None:
                raise AttributeError("Camera indices are not provided.")

            outputs = {}

            camera_indices = ray_samples.camera_indices.squeeze()

            inputs = ray_samples.frustums.get_start_positions()
            inputs = inputs.view(-1, 3)

            directions = ray_samples.frustums.directions
            directions_flat = directions.reshape(-1, 3)

            if self.spatial_distortion is not None:
                inputs = self.spatial_distortion(inputs)
            points_norm = inputs.norm(dim=-1)
            # compute gradient in constracted space
            inputs.requires_grad_(True)
            with torch.enable_grad():
                h = self.forward_geonetwork(inputs, encoding)
                sdf, geo_feature = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)

            if self.config.use_numerical_gradients:
                gradients, sampled_sdf = self.gradient(
                    inputs,
                    skip_spatial_distortion=True,
                    return_sdf=True,
                )
                sampled_sdf = sampled_sdf.view(-1, *ray_samples.frustums.directions.shape[:-1]).permute(1, 2, 0).contiguous()
            else:
                d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                gradients = torch.autograd.grad(
                    outputs=sdf,
                    inputs=inputs,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                sampled_sdf = None

            rgb = self.get_colors(inputs, directions_flat, gradients, geo_feature, camera_indices)

            density = self.laplace_density(sdf)

            rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
            sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
            density = density.view(*ray_samples.frustums.directions.shape[:-1], -1)
            gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
            normals = F.normalize(gradients, p=2, dim=-1)
            points_norm = points_norm.view(*ray_samples.frustums.directions.shape[:-1], -1)

            outputs.update(
                {
                    FieldHeadNames.RGB: rgb,
                    FieldHeadNames.DENSITY: density,
                    FieldHeadNames.SDF: sdf,
                    FieldHeadNames.NORMAL: normals,
                    FieldHeadNames.GRADIENT: gradients,
                    "points_norm": points_norm,
                    "sampled_sdf": sampled_sdf,
                }
            )

            if return_alphas:
                # TODO use mid point sdf for NeuS
                alphas = self.get_alpha(ray_samples, sdf, gradients)
                outputs.update({FieldHeadNames.ALPHA: alphas})

            if return_occupancy:
                occupancy = self.get_occupancy(sdf)
                outputs.update({FieldHeadNames.OCCUPANCY: occupancy})

        return outputs

    def forward(self, ray_samples: RaySamples, return_alphas=False, return_occupancy=False, sdf_only=False, scene_id=0):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        field_outputs = self.get_outputs(ray_samples, return_alphas=return_alphas, return_occupancy=return_occupancy, sdf_only=sdf_only, scene_id=scene_id)
        return field_outputs
