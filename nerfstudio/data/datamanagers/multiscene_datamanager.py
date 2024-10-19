from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import tyro
from rich.progress import Console
from torch import nn
from torch.nn import Parameter
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import Literal

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager, DataManager
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.friends_dataparser import FriendsDataParserConfig
from nerfstudio.data.dataparsers.heritage_dataparser import HeritageDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
)
from nerfstudio.data.dataparsers.mipnerf360_dataparser import Mipnerf360DataParserConfig
from nerfstudio.data.dataparsers.monosdf_dataparser import MonoSDFDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
# from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import (
    PhototourismDataParserConfig,
)
from nerfstudio.data.dataparsers.record3d_dataparser import Record3DDataParserConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
# from nerfstudio.data.dataparsers.scannetpp_dataparser import ScanNetppDataParserConfig
from nerfstudio.data.datasets.base_dataset import GeneralizedDataset, InputDataset
from nerfstudio.data.pixel_samplers import EquirectangularPixelSampler, PixelSampler
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.images import BasicImages
from nerfstudio.utils.misc import IterableWrapper
from pdb import set_trace as pause

CONSOLE = Console(width=120)

AnnotatedDataParserUnion = tyro.conf.OmitSubcommandPrefixes[  # Omit prefixes of flags in subcommands.
    tyro.extras.subcommand_type_from_defaults(
        {
            "sdfstudio-data": SDFStudioDataParserConfig(),
        },
        prefix_names=False,  # Omit prefixes in subcommands themselves.
    )
]

@dataclass
class MultisceneDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: MultisceneDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = SDFStudioDataParserConfig()
    """Specifies the dataparser used to unpack the data."""


    # TODO: Change lists to dictionaries, where data_id is key

class MultisceneDataManager(VanillaDataManager):

    config: MultisceneDataManagerConfig
    train_dataset: InputDataset
    eval_dataset: InputDataset
    train_datasets: Dict[str, InputDataset]
    eval_datasets: Dict[str, InputDataset]

    def __init__(
        self,
        config: MultisceneDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_plit = "test" if test_mode in ["test", "inference"] else "val"
        self.num_scenes = len(self.config.dataparser.multiscene_data)
        self.scenes = {scene.parts[-3]: scene
                for scene in self.config.dataparser.multiscene_data}
        self.scene_ids = list(self.scenes.keys())

        self.dataparsers = {scene_id: self.config.dataparser.setup()
                for scene_id in self.scenes.keys()}

        self.train_datasets = {scene_id: self.create_dataset(scene_id,
                "train")
                for scene_id in self.scenes.keys()}
        self.eval_datasets = {scene_id: self.create_dataset(scene_id,
                self.test_mode)
                for scene_id in self.scenes.keys()}

        self.train_dataset = next(iter(self.train_datasets.values()))
        self.eval_dataset = next(iter(self.eval_datasets.values()))
        DataManager.__init__(self)

    def create_dataset(self, scene_id, split):
        dataparser = self.dataparsers[scene_id]
        return GeneralizedDataset(
                dataparser_outputs=dataparser._generate_dataparser_outputs(
                    split="train",
                    data_path=self.scenes[scene_id]),
                scale_factor=self.config.camera_res_scale_factor,
                )

    def _get_pixel_sampler(  # pylint: disable=no-self-use
        self, dataset: InputDataset, *args: Any, **kwargs: Any
    ) -> PixelSampler:
        """Infer pixel sampler to use."""
        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)
        # Otherwise, use the default pixel sampler
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return PixelSampler(*args, **kwargs)


    def setup_train(self):
        """Sets up the data loaders for training"""
        #return
        # here for running on all samples
        CONSOLE.print("Setting up training dataset(s)...")
        self.train_image_dataloaders = {}
        self.iter_train_image_dataloaders = {}
        self.train_pixel_samplers = {}
        self.train_camera_optimizers = nn.ModuleDict()
        self.train_ray_generators = nn.ModuleDict()
        self.fixed_indices_train_dataloaders = {}
        for scene_id in self.scenes.keys():
            train_dataset = self.train_datasets[scene_id]
            train_image_dataloader = CacheDataloader(
                train_dataset,
                num_images_to_sample_from=1,
                num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
                device=self.device,
                num_workers=self.world_size * 4,
                pin_memory=True,
                collate_fn=self.config.collate_fn,
            )
            iter_train_image_dataloader = iter(train_image_dataloader)
            train_pixel_sampler = self._get_pixel_sampler(train_dataset, self.config.train_num_rays_per_batch)
            train_camera_optimizer = self.config.camera_optimizer.setup(
                num_cameras=train_dataset.cameras.size, device=self.device
            )
            train_ray_generator = RayGenerator(
                train_dataset.cameras.to(self.device),
                train_camera_optimizer,
            )
            # for loading full images
            fixed_indices_train_dataloader = FixedIndicesEvalDataloader(
                input_dataset=train_dataset,
                device=self.device,
                num_workers=self.world_size * 2,
                shuffle=False,
            )

            self.train_image_dataloaders[scene_id] = train_image_dataloader
            self.iter_train_image_dataloaders[scene_id] = iter_train_image_dataloader
            self.train_pixel_samplers[scene_id] = train_pixel_sampler
            self.train_camera_optimizers[scene_id] = train_camera_optimizer
            self.train_ray_generators[scene_id] = train_ray_generator
            self.fixed_indices_train_dataloaders[scene_id] = fixed_indices_train_dataloader

        first_scene = self.scene_ids[0]
        self.train_image_dataloader = self.train_image_dataloaders[first_scene]
        self.iter_train_image_dataloader = self.iter_train_image_dataloaders[first_scene]
        self.train_pixel_sampler = self.train_pixel_samplers[first_scene]
        self.train_camera_optimizer = self.train_camera_optimizers[first_scene]
        self.train_ray_generator = self.train_ray_generators[first_scene]
        self.fixed_indices_train_dataloader = self.fixed_indices_train_dataloaders[first_scene]

    def setup_eval(self):
        """Sets up the data loaders for evaling"""
        #return
        # here for running on all samples
        CONSOLE.print("Setting up evaling dataset(s)...")
        self.eval_image_dataloaders = {}
        self.iter_eval_image_dataloaders = {}
        self.eval_pixel_samplers = {}
        self.eval_camera_optimizers = nn.ModuleDict()
        self.eval_ray_generators = nn.ModuleDict()
        self.fixed_indices_eval_dataloaders = {}
        self.eval_dataloaders = {}
        for scene_id in self.scenes.keys():
            eval_dataset = self.eval_datasets[scene_id]
            eval_image_dataloader = CacheDataloader(
                eval_dataset,
                num_images_to_sample_from=1,
                num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
                device=self.device,
                num_workers=self.world_size * 4,
                pin_memory=True,
                collate_fn=self.config.collate_fn,
            )
            iter_eval_image_dataloader = iter(eval_image_dataloader)
            eval_pixel_sampler = self._get_pixel_sampler(eval_dataset, self.config.eval_num_rays_per_batch)
            eval_camera_optimizer = self.config.camera_optimizer.setup(
                num_cameras=eval_dataset.cameras.size, device=self.device
            )
            eval_ray_generator = RayGenerator(
                eval_dataset.cameras.to(self.device),
                eval_camera_optimizer,
            )
            # for loading full images
            fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
                input_dataset=eval_dataset,
                device=self.device,
                num_workers=self.world_size * 2,
                shuffle=False,
            )

            eval_dataloader = RandIndicesEvalDataloader(
                input_dataset=self.eval_dataset,
                image_indices=self.config.eval_image_indices,
                device=self.device,
                num_workers=self.world_size * 2,
                shuffle=False,
            )

            self.eval_image_dataloaders[scene_id] = eval_image_dataloader
            self.iter_eval_image_dataloaders[scene_id] = iter_eval_image_dataloader
            self.eval_pixel_samplers[scene_id] = eval_pixel_sampler
            self.eval_camera_optimizers[scene_id] = eval_camera_optimizer
            self.eval_ray_generators[scene_id] = eval_ray_generator
            self.fixed_indices_eval_dataloaders[scene_id] = fixed_indices_eval_dataloader
            self.eval_dataloaders[scene_id] = eval_dataloader

        first_scene = self.scene_ids[0]
        self.eval_image_dataloader = self.eval_image_dataloaders[first_scene]
        self.iter_eval_image_dataloader = self.iter_eval_image_dataloaders[first_scene]
        self.eval_pixel_sampler = self.eval_pixel_samplers[first_scene]
        self.eval_camera_optimizer = self.eval_camera_optimizers[first_scene]
        self.eval_ray_generator = self.eval_ray_generators[first_scene]
        self.fixed_indices_eval_dataloader = self.fixed_indices_eval_dataloaders[first_scene]
        self.eval_dataloader = self.eval_dataloaders[first_scene]

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        scene_id = self.scene_ids[step % self.num_scenes]
        self.train_image_dataloader = self.train_image_dataloaders[scene_id]
        self.iter_train_image_dataloader = self.iter_train_image_dataloaders[scene_id]
        self.train_pixel_sampler = self.train_pixel_samplers[scene_id]
        self.train_camera_optimizer = self.train_camera_optimizers[scene_id]
        self.train_ray_generator = self.train_ray_generators[scene_id]
        self.fixed_indices_train_dataloader = self.fixed_indices_train_dataloaders[scene_id]
        return super().next_train(step)

    def next_eval(self, step: int, scene_id) -> Tuple[RayBundle, Dict]:
        self.eval_image_dataloader = self.eval_image_dataloaders[scene_id]
        self.iter_eval_image_dataloader = self.iter_eval_image_dataloaders[scene_id]
        self.eval_pixel_sampler = self.eval_pixel_samplers[scene_id]
        self.eval_camera_optimizer = self.eval_camera_optimizers[scene_id]
        self.eval_ray_generator = self.eval_ray_generators[scene_id]
        self.fixed_indices_eval_dataloader = self.fixed_indices_eval_dataloaders[scene_id]
        self.eval_dataloader = self.eval_dataloaders[scene_id]
        return super().next_eval(step)

    def next_eval_image(self, step: int, scene_id) -> Tuple[RayBundle, Dict]:
        self.eval_image_dataloader = self.eval_image_dataloaders[scene_id]
        self.iter_eval_image_dataloader = self.iter_eval_image_dataloaders[scene_id]
        self.eval_pixel_sampler = self.eval_pixel_samplers[scene_id]
        self.eval_camera_optimizer = self.eval_camera_optimizers[scene_id]
        self.eval_ray_generator = self.eval_ray_generators[scene_id]
        self.fixed_indices_eval_dataloader = self.fixed_indices_eval_dataloaders[scene_id]
        self.eval_dataloader = self.eval_dataloaders[scene_id]
        return super().next_eval_image(step)

    # ignore
    # def get_param_groups(self) -> Dict[str, List[Parameter]]:
