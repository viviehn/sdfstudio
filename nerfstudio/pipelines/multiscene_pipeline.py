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
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    FlexibleDataManager,
    FlexibleDataManagerConfig,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datamanagers.multiscene_datamanager import MultisceneDataManager, MultisceneDataManagerConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.neus_facto_multi import NeuSFactoMultiModel, NeuSFactoMultiModelConfig
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.images import BasicImages
from pdb import set_trace as pause
import random

@dataclass
class MultiscenePipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: MultiscenePipeline)
    """target class to instantiate"""
    datamanager: MultisceneDataManagerConfig = MultisceneDataManagerConfig()
    """specifies the datamanager config"""
    model: NeuSFactoMultiModelConfig = NeuSFactoMultiModelConfig()
    """specifies the model config"""


class MultiscenePipeline(VanillaPipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test datset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: MultiscenePipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config=config, device=device)
        self.config = config
        self.test_mode = test_mode
        self.datamanager: MultisceneDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        assert len(self.datamanager.train_dataset_list) > 0, "Missing input dataset"


        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset_list[0].scene_box,
            num_train_data= sum(len(train_dataset) for train_dataset in
                                self.datamanager.train_dataset_list),
            metadata=self.datamanager.train_dataset_list[0].metadata,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.model.to(device)
        [e.to(device) for e in self.model.field.encodings_list]

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        # pause()

        # Multiscene Datamanager returns a list of ray_bundles and batches
        ray_bundle_list, batch_list = self.datamanager.next_train(step)

        # Forward model takes one scene at a time

        metrics_dict_list = []
        loss_dict_list = []
        model_outputs = []

        '''
        for enc in self.model.field.encodings_list:
            enc.embeddings.requires_grad = False
        '''

        sampled_scene_ids = random.sample(range(self.datamanager.num_scenes), min(self.datamanager.num_scenes,5))
        for scene_id in sampled_scene_ids:
            ray_bundle = ray_bundle_list[scene_id]
            batch = batch_list[scene_id]
            model_output = self.model(ray_bundle, scene_id)
            metrics_dict = self.model.get_metrics_dict(model_output, batch)
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                # Report the camera optimization metrics
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )
            metrics_dict_list.append(metrics_dict)
            loss_dict = self.model.get_loss_dict(model_output, batch, metrics_dict)
            loss_dict_list.append(loss_dict)
            model_outputs.append(model_output)


        assert len(sampled_scene_ids) == len(loss_dict_list)

        # Average losses and metrics over scenes
        loss_dict_merged = {}
        for loss_dict_key in loss_dict_list[0].keys():
            loss_dict_merged[loss_dict_key] = sum(d[loss_dict_key] for d in loss_dict_list) / len(loss_dict_list)

        metrics_dict_merged = {}
        for metrics_dict_key in metrics_dict_list[0].keys():
            metrics_dict_merged[metrics_dict_key] = sum(d[metrics_dict_key] for d in metrics_dict_list) / len(metrics_dict_list)

        return model_outputs, loss_dict_merged, metrics_dict_merged

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int, sampled_scene_ids: list) -> Tuple[List[Dict], Dict, Dict]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle_list, batch_list = self.datamanager.next_eval(step)

        metrics_dict_list = []
        loss_dict_list = []
        model_outputs = []
        for scene_id in sampled_scene_ids:
            ray_bundle = ray_bundle_list[scene_id]
            batch = batch_list[scene_id]
            model_output = self.model(ray_bundle, scene_id)
            model_outputs.append(model_output)
            metrics_dict = self.model.get_metrics_dict(model_output, batch)
            metrics_dict_list.append(metrics_dict)
            loss_dict = self.model.get_loss_dict(model_output, batch, metrics_dict)
            loss_dict_list.append(loss_dict)

        loss_dict_merged = {}
        for loss_dict_key in loss_dict_list[0].keys():
            loss_dict_merged[loss_dict_key] = sum(d[loss_dict_key] for d in loss_dict_list) / len(loss_dict_list)

        metrics_dict_merged = {}
        for metrics_dict_key in metrics_dict_list[0].keys():
            metrics_dict_merged[metrics_dict_key] = sum(d[metrics_dict_key] for d in metrics_dict_list) / len(metrics_dict_list)
        self.train()

        # Returns dicts with per-scene keys [e.g. for directly logging to tensorboard]
        return model_outputs, loss_dict_merged, metrics_dict_merged

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int, sampled_scene_ids: list):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        torch.cuda.empty_cache()
        image_idx_list, camera_ray_bundle_list, batch_list = self.datamanager.next_eval_image(step)
        model_outputs = []
        metrics_dict_list = []
        images_dict_list = []
        for scene_id in sampled_scene_ids:
            print(scene_id)
            camera_ray_bundle = camera_ray_bundle_list[scene_id]
            batch = batch_list[scene_id]
            image_idx = image_idx_list[scene_id]
            outputs =  self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, scene_id=scene_id)
            metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
            assert "image_idx" not in metrics_dict
            metrics_dict["image_idx"] = image_idx
            assert "num_rays" not in metrics_dict
            metrics_dict["num_rays"] = len(camera_ray_bundle)

            model_outputs.append(outputs)
            metrics_dict_list.append(metrics_dict)
            images_dict_list.append(images_dict)
        self.train()
        mega_metrics_dict = {}
        mega_images_dict = {}
        for idx, i in enumerate(sampled_scene_ids):
            for k, v in images_dict_list[idx].items():
                new_k = f'{k}_{i}'
                mega_images_dict[new_k] = v
        for k, v in metrics_dict_list[0].items():
            mega_metrics_dict[k] = sum(d[k] for d in metrics_dict_list) / len(metrics_dict_list)
        return mega_metrics_dict, mega_images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        images_dict_list = []
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                isbasicimages = False
                if isinstance(
                    batch["image"], BasicImages
                ):  # If this is a generalized dataset, we need to get image tensor
                    isbasicimages = True
                    batch["image"] = batch["image"].images[0]
                    camera_ray_bundle = camera_ray_bundle.reshape((*batch["image"].shape[:-1],))
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                images_dict_list.append(images_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )
        self.train()
        return metrics_dict, images_dict_list

    '''

    @profiler.time_function
    def get_visibility_mask(
        self,
        coarse_grid_resolution: int = 512,
        valid_points_thres: float = 0.005,
        sub_sample_factor: int = 8,
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()

        coarse_mask = torch.ones(
            (1, 1, coarse_grid_resolution, coarse_grid_resolution, coarse_grid_resolution), requires_grad=True
        ).to(self.device)
        coarse_mask.retain_grad()

        num_images = len(self.datamanager.fixed_indices_train_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_train_dataloader:
                isbasicimages = False
                if isinstance(
                    batch["image"], BasicImages
                ):  # If this is a generalized dataset, we need to get image tensor
                    isbasicimages = True
                    batch["image"] = batch["image"].images[0]
                    camera_ray_bundle = camera_ray_bundle.reshape((*batch["image"].shape[:-1],))
                # downsample by factor of 4 to speed up
                camera_ray_bundle = camera_ray_bundle[::sub_sample_factor, ::sub_sample_factor]
                height, width = camera_ray_bundle.shape
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                ray_points = outputs["ray_points"].reshape(height, width, -1, 3)
                weights = outputs["weights"]

                valid_points = ray_points.reshape(-1, 3)[weights.reshape(-1) > valid_points_thres]
                valid_points = valid_points * 0.5  # normalize from [-2, 2] to [-1, 1]
                # update mask based on ray samples
                with torch.enable_grad():
                    out = torch.nn.functional.grid_sample(coarse_mask, valid_points[None, None, None])
                    out.sum().backward()
                progress.advance(task)

        coarse_mask = (coarse_mask.grad > 0.0001).float()

        self.train()
        return coarse_mask
    '''

    def load_pipeline(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state.items()}
        if self.test_mode == 'val' and state["_model.field.embedding_appearance.embedding.weight"].shape[0] == 305:
            # need to pop off ALL encodings
            state.pop("_model.field.embedding_appearance.embedding.weight")
            state.pop("_model.field.encoding.embeddings")
            state.pop("_model.field.encoding.offsets")
            #TODO: handle _model.field.encodings
        if self.test_mode == "inference":
            state.pop("datamanager.train_camera_optimizer.pose_adjustment", None)
            state.pop("datamanager.train_ray_generator.image_coords", None)
            state.pop("datamanager.train_ray_generator.pose_optimizer.pose_adjustment", None)
            state.pop("datamanager.eval_ray_generator.image_coords", None)
            state.pop("datamanager.eval_ray_generator.pose_optimizer.pose_adjustment", None)

        missing, unexpected = self.load_state_dict(state, strict=False)  # type: ignore
        print(f"Missing: {missing}")
        print(f"Unexpected: {unexpected}")

