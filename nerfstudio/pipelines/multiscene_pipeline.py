from __future__ import annotations

import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional, Type, Union, cast

import torch
import torch.distributed as dist
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch import nn
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import Literal

from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.multiscene_datamanager import (
    MultisceneDataManagerConfig,
    MultisceneDataManager
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.images import BasicImages
from pdb import set_trace as pause

from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline, VanillaPipelineConfig

@dataclass
class MultiscenePipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: MultiscenePipeline)
    """target class to instantiate"""
    datamanager: MultisceneDataManagerConfig = MultisceneDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig() # make model with split_field
    """specifies the model config"""


class MultiscenePipeline(VanillaPipeline):
    def __init__(
        self,
        config: MultiscenePipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        Pipeline.__init__(self)
        self.config = config
        self.test_mode = test_mode
        if config.model.sdf_sample_training:
            assert config.datamanager.dataparser.include_sdf_samples, "Trying to train with sdf samples but did not include sdf samples in data"

        self.datamanager: MultisceneDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank,
        )
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            world_size=world_size,
            local_rank=local_rank,
            scene_ids=self.datamanager.scene_ids
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """

        # datamanager will swap to different scene based on step
        ray_bundle, batch = self.datamanager.next_train(step)

        # need to swap model elements based on step
        scene_id = self.datamanager.scene_ids[step % self.datamanager.num_scenes]
        self.model.field.geometry_encoding = self.model.field.geometry_encodings[scene_id]
        self.model.field.appearance_encoding = self.model.field.appearance_encodings[scene_id]
        if self.model.config.sdf_sample_training:
            model_outputs = self._model(batch['sparse_sdf_samples'].to(self.device))
        else:
            model_outputs = self._model(ray_bundle)

        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
        if camera_opt_param_group in self.datamanager.get_param_groups():
            # Report the camera optimization metrics
            metrics_dict["camera_opt_translation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
            )
            metrics_dict["camera_opt_rotation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
            )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        aggr_metrics_dict = {}
        aggr_loss_dict = {}
        for scene_id in self.datamanager.scene_ids:
            ray_bundle, batch = self.datamanager.next_eval(step, scene_id)
            self.model.field.geometry_encoding = self.model.field.geometry_encodings[scene_id]
            self.model.field.appearance_encoding = self.model.field.appearance_encodings[scene_id]
            if self.model.config.sdf_sample_training:
                model_outputs = self._model(batch['sparse_sdf_samples'].to(self.device))
            else:
                model_outputs = self._model(ray_bundle)
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
            metrics_dict_scene = {f'{k}_{scene_id}': v for k, v in metrics_dict.items()}
            loss_dict_scene = {f'{k}_{scene_id}': v for k, v in loss_dict.items()}
            aggr_metrics_dict.update(metrics_dict_scene)
            aggr_loss_dict.update(loss_dict_scene)
        self.train()
        return model_outputs, aggr_loss_dict, aggr_metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        torch.cuda.empty_cache()
        aggr_metrics_dict = {}
        aggr_images_dict = {}
        for scene_id in self.datamanager.scene_ids:
            self.model.field.geometry_encoding = self.model.field.geometry_encodings[scene_id]
            self.model.field.appearance_encoding = self.model.field.appearance_encodings[scene_id]
            image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step, scene_id)
            #if self.model.config.sdf_sample_training:
            #    outputs = self.model.get_outputs_for_camera_ray_bundle(batch['sparse_sdf_samples'].to(self.device))
            #else:
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
            assert "image_idx" not in metrics_dict
            metrics_dict["image_idx"] = image_idx
            assert "num_rays" not in metrics_dict
            metrics_dict["num_rays"] = len(camera_ray_bundle)
            metrics_dict_scene = {f'{k}_{scene_id}': v for k, v in metrics_dict.items()}
            images_dict_scene = {f'{k}_{scene_id}': v for k, v in images_dict.items()}
            aggr_metrics_dict.update(metrics_dict_scene)
            aggr_images_dict.update(images_dict_scene)
        aggr_metrics_dict['num_rays'] = len(camera_ray_bundle)
        aggr_metrics_dict["image_idx"] = image_idx
        self.train()
        return aggr_metrics_dict, aggr_images_dict

