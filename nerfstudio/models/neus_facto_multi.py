# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Implementation of VolSDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Type, Tuple, Dict
import numpy as np

import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.models.neus_facto import NeuSFactoModel, NeuSFactoModelConfig
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import interlevel_loss, interlevel_loss_zip
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.utils import colormaps
from pdb import set_trace as pause
from nerfstudio.utils import profiler

@dataclass
class NeuSFactoMultiModelConfig(NeuSFactoModelConfig):
    """UniSurf Model Config"""

    _target: Type = field(default_factory=lambda: NeuSFactoMultiModel)


class NeuSFactoMultiModel(NeuSFactoModel):
    """NeuS facto model

    Args:
        config: NeuS configuration to instantiate model
    """

    config: NeuSFactoMultiModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()


    '''
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        # Possibly use this function to "swap" scenes for multiscene training???
    '''


    #@profiler.time_function
    def sample_and_forward_field(self, ray_bundle: RayBundle, scene_id: int):
        # pause()
        if isinstance(ray_bundle, torch.Tensor):
            sdf_only = self.config.eikonal_loss_mult == 0 and self.config.curvature_loss_multi == 0
            field_outputs = self.field(ray_bundle, return_alphas=False, sdf_only=sdf_only, scene_id=scene_id)
            return {"field_outputs": field_outputs}

        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        field_outputs = self.field(ray_samples, return_alphas=True, scene_id=scene_id)

        if self.config.background_model != "none":
            field_outputs = self.forward_background_field_and_merge(ray_samples, field_outputs)

        weights = ray_samples.get_weights_from_alphas(field_outputs[FieldHeadNames.ALPHA])

        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "weights_list": weights_list,
            "ray_samples_list": ray_samples_list,
        }
        return samples_and_field_outputs

    #@profiler.time_function
    '''
    Return loss dict for a single scene, which will be averaged elsewhere
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training and not "sparse_sdf_samples" in batch.keys():
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss_zip(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

        # curvature loss
        if self.training and self.config.curvature_loss_multi > 0.0 and self.config.sdf_field.use_numerical_gradients:
            delta = self.field.numerical_gradients_delta
            centered_sdf = outputs["field_outputs"][FieldHeadNames.SDF]
            sourounding_sdf = outputs["field_outputs"]["sampled_sdf"]
            # pause()

            # sourounding_sdf = sourounding_sdf.reshape(centered_sdf.shape[:2] + (3, 2))
            sourounding_sdf = sourounding_sdf.reshape(centered_sdf.shape[:centered_sdf.dim()-1] + (3, 2))

            # (a - b)/d - (b -c)/d = (a + c - 2b)/d
            # ((a - b)/d - (b -c)/d)/d = (a + c - 2b)/(d*d)
            curvature = (sourounding_sdf.sum(dim=-1) - 2 * centered_sdf) / (delta * delta)
            loss_dict["curvature_loss"] = (
                torch.abs(curvature).mean() * self.config.curvature_loss_multi * self.curvature_loss_multi_factor
            )

        return loss_dict
    '''

    def forward(self, ray_bundle: RayBundle, scene_id: int) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None and isinstance(ray_bundle, RayBundle):
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, scene_id)

    def get_outputs(self, ray_bundle: RayBundle, scene_id:int) -> Dict:
        outputs = {}
        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle, scene_id=scene_id)
        # pause()

        # Shotscuts
        field_outputs = samples_and_field_outputs["field_outputs"]
        if self.training:
            if FieldHeadNames.GRADIENT in field_outputs:
                grad_points = field_outputs[FieldHeadNames.GRADIENT]
                # points_norm = field_outputs["points_norm"]
                # outputs.update({"eik_grad": grad_points, "points_norm": points_norm})
                outputs.update({"eik_grad": grad_points})
            if "points_norm" in field_outputs:
                points_norm = field_outputs["points_norm"]
                outputs.update({"points_norm": points_norm})

            # TODO volsdf use different point set for eikonal loss
            # grad_points = self.field.gradient(eik_points)
            # outputs.update({"eik_grad": grad_points})

            outputs.update(samples_and_field_outputs)
            if isinstance(ray_bundle, torch.Tensor):
                if FieldHeadNames.RGB in field_outputs.keys():
                    outputs.update({"rgb": field_outputs[FieldHeadNames.RGB]})
                return outputs

        if isinstance(ray_bundle, torch.Tensor):
            outputs.update(samples_and_field_outputs)
            if FieldHeadNames.RGB in field_outputs.keys():
                outputs.update({"rgb": field_outputs[FieldHeadNames.RGB]})
            return outputs



        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.directions_norm

        # remove the rays that don't intersect with the surface
        # hit = (field_outputs[FieldHeadNames.SDF] > 0.0).any(dim=1) & (field_outputs[FieldHeadNames.SDF] < 0).any(dim=1)
        # depth[~hit] = 10000.0

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        # TODO add a flat to control how the background model are combined with foreground sdf field
        # background model
        if self.config.background_model != "none" and "bg_transmittance" in samples_and_field_outputs:
            bg_transmittance = samples_and_field_outputs["bg_transmittance"]

            # sample inversely from far to 1000 and points and forward the bg model
            ray_bundle.nears = ray_bundle.fars
            ray_bundle.fars = torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg

            ray_samples_bg = self.sampler_bg(ray_bundle)
            # use the same background model for both density field and occupancy field
            field_outputs_bg = self.field_background(ray_samples_bg)
            weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

            rgb_bg = self.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)

            # merge background color to forgound color
            rgb = rgb + bg_transmittance * rgb_bg

        outputs.update({
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            "ray_points": self.scene_contraction(
                ray_samples.frustums.get_start_positions()
            ),  # used for creating visiblity mask
            "directions_norm": ray_bundle.directions_norm,  # used to scale z_vals for free space and sdf loss
        })

        # TODO how can we move it to neus_facto without out of memory
        if "weights_list" in samples_and_field_outputs:
            weights_list = samples_and_field_outputs["weights_list"]
            ray_samples_list = samples_and_field_outputs["ray_samples_list"]

            for i in range(len(weights_list) - 1):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )
        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, scene_id: int) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle, scene_id=scene_id)
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs
