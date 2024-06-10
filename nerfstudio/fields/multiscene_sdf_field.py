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
            self.encodings_list.append(enc)


    def forward_geonetwork(self, inputs):
        """forward the geonetwork"""
        # inputs = torch.ones_like(inputs)[:90000]
        # inputs[:, 0] *= torch.arange(90000).cuda() / 90000 - 0.5
        # inputs[:, 1] *= -torch.arange(90000).cuda() / 90000
        # inputs[:, 2] *= 1-torch.arange(90000).cuda() / 90000
        # inputs *= 2
        # MULTISCENE TODO: hash features are sampled here. need to do this separately for each scene
        if self.use_grid_feature:
            #TODO normalize inputs depending on the whether we model the background or not
            # pause()
            if self.config.vanilla_ngp:
                positions = inputs * 1.0
                # positions = inputs / 2
            else:
                positions = (inputs + 2.0) / 4.0
            # positions = (inputs + 1.0) / 2.0
            feature = self.encoding(positions)
            # mask feature
            if not self.config.vanilla_ngp:
                feature = feature * self.hash_encoding_mask.to(feature.device)
        else:
            feature = torch.zeros_like(inputs[:, :1].repeat(1, self.encoding.n_output_dims))

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

    def get_outputs(self, ray_samples_list: List, return_alphas=False, return_occupancy=False, sdf_only=False, need_rgb=False):
        outputs_list = []
        for ray_samples in ray_samples_list:
            inputs = ray_samples[:, :3]
            # compute gradient in constracted space
            inputs.requires_grad_(True)
            with torch.enable_grad():
                h = self.forward_geonetwork(inputs)
                sdf, geo_feature = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
            outputs = {
                    FieldHeadNames.SDF: sdf,
                }

            # if self.config.curvature_loss_multi== 0.0:
            # if True: #self.config.eikonal_loss_mult == 0.0 and 
            if ray_samples.shape[1] == 4:
                if sdf_only:
                    outputs_list.append(outputs)
                    continue
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
                        # FieldHeadNames.SDF: sdf,
                        # FieldHeadNames.NORMAL: normals,
                        FieldHeadNames.GRADIENT: gradients,
                        # "points_norm": points_norm,
                        "sampled_sdf": sampled_sdf,
                    })
                outputs_list.append(outputs)
                continue
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
            outputs_list.append(outputs)
            continue
        # list of outputs
        return outputs_list
