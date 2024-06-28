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

class MultisceneDataManager(DataManager):


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
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"

        self.num_scenes = len(self.config.dataparser.multiscene_data)
        self.dataparser_list = [self.config.dataparser.setup() for scene in self.config.dataparser.multiscene_data]
        print(self.dataparser_list)

        self.train_dataset_list = []
        for scene, dataparser in zip(self.config.dataparser.multiscene_data, self.dataparser_list):
            print(scene)
            self.train_dataset_list.append(self.create_train_dataset(scene, dataparser))

        print(self.train_dataset_list)

        self.eval_dataset_list = []
        for scene, dataparser in zip(self.config.dataparser.multiscene_data, self.dataparser_list):
            self.eval_dataset_list.append(self.create_eval_dataset(scene, dataparser))


        self.train_dataset = self.train_dataset_list[0]
        self.eval_dataset = self.eval_dataset_list[0]

        super().__init__()

    def create_train_dataset(self, scene=None, dataparser=None) -> InputDataset:
        """Sets up the data loaders for training"""
        return GeneralizedDataset(
            dataparser_outputs=dataparser.get_dataparser_outputs(split="train", scene=scene),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self, scene=None, dataparser=None) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        return GeneralizedDataset(
            dataparser_outputs=dataparser.get_dataparser_outputs(split=self.test_split, scene=scene),
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

    # MULTISCENE TODO: basically every instance variable here should be a list, where each
    # entry corresponds to a single scene
    def setup_train(self):
        """Sets up the data loaders for training"""
        assert len(self.train_dataset_list) > 0
        CONSOLE.print("Setting up training dataset(s)...")
        self.train_image_dataloaders = []
        self.iter_train_image_dataloaders = []
        self.train_pixel_samplers = []
        self.train_camera_optimizers = []
        self.train_ray_generators = []
        self.fixed_indices_train_dataloaders = []
        for train_dataset in self.train_dataset_list:
            train_image_dataloader = CacheDataloader(
                train_dataset,
                num_images_to_sample_from=self.config.train_num_images_to_sample_from,
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

            self.train_image_dataloaders.append(train_image_dataloader)
            self.iter_train_image_dataloaders.append(iter_train_image_dataloader)
            self.train_pixel_samplers.append(train_pixel_sampler)
            self.train_camera_optimizers.append(train_camera_optimizer)
            self.train_ray_generators.append(train_ray_generator)
            self.fixed_indices_train_dataloaders.append(fixed_indices_train_dataloader)

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset(s)...")
        self.eval_image_dataloaders = []
        self.iter_eval_image_dataloaders = []
        self.eval_pixel_samplers = []
        self.eval_camera_optimizers = []
        self.eval_ray_generators = []
        self.fixed_indices_eval_dataloaders = []
        self.eval_dataloaders = []
        for eval_dataset in self.eval_dataset_list:
            eval_image_dataloader = CacheDataloader(
                eval_dataset,
                num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
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
                input_dataset=eval_dataset,
                image_indices=self.config.eval_image_indices,
                device=self.device,
                num_workers=self.world_size * 2,
                shuffle=False,
            )

            self.eval_image_dataloaders.append(eval_image_dataloader)
            self.iter_eval_image_dataloaders.append(iter_eval_image_dataloader)
            self.eval_pixel_samplers.append(eval_pixel_sampler)
            self.eval_camera_optimizers.append(eval_camera_optimizer)
            self.eval_ray_generators.append(eval_ray_generator)
            self.fixed_indices_eval_dataloaders.append(fixed_indices_eval_dataloader)
            self.eval_dataloaders.append(eval_dataloader)

    # MULTISCENE TODO: next_train should gather data for each scene,
    # concat them together, then output the concatenated ray_bundle 

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        cat_ray_bundles = []
        batch_list = []
        for i in range(len(self.train_dataset_list)):
            iter_train_image_dataloader = self.iter_train_image_dataloaders[i]
            train_pixel_sampler = self.train_pixel_samplers[i]
            train_ray_generator = self.train_ray_generators[i]
            image_batch = next(iter_train_image_dataloader)
            batch = train_pixel_sampler.sample(image_batch)
            batch_list.append(batch)
            if self.config.dataparser.include_sdf_samples:
                ray_bundle = batch["sparse_sdf_samples"].to(self.device)
                cat_ray_bundles.append(ray_bundle)
            else:
                ray_indices = batch["indices"]
                ray_bundle = train_ray_generator(ray_indices)
                cat_ray_bundles.append(ray_bundle)
        return cat_ray_bundles, batch_list

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        cat_ray_bundles = []
        batch_list = []
        for i in range(len(self.eval_dataset_list)):
            iter_eval_image_dataloader = self.iter_eval_image_dataloaders[i]
            eval_pixel_sampler = self.eval_pixel_samplers[i]
            eval_ray_generator = self.eval_ray_generators[i]
            image_batch = next(iter_eval_image_dataloader)
            batch = eval_pixel_sampler.sample(image_batch)
            batch_list.append(batch)
            if self.config.dataparser.include_sdf_samples:
                ray_bundle = batch["sparse_sdf_samples"].to(self.device)
                cat_ray_bundles.append(ray_bundle)
            else:
                ray_indices = batch["indices"]
                ray_bundle = eval_ray_generator(ray_indices)
                cat_ray_bundles.append(ray_bundle)
        return cat_ray_bundles, batch_list

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        image_idx_list = []
        camera_ray_bundle_list = []
        batch_list = []
        for eval_dataloader in self.eval_dataloaders:
            for camera_ray_bundle, batch in eval_dataloader:
                assert camera_ray_bundle.camera_indices is not None
                if isinstance(batch["image"], BasicImages):  # If this is a generalized dataset, we need to get image tensor
                    batch["image"] = batch["image"].images[0]
                    camera_ray_bundle = camera_ray_bundle.reshape((*batch["image"].shape[:-1], 1))
                image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
                if batch['image'].dtype == torch.uint8:
                    batch['image'] = batch['image'].float() / 255.0
                image_idx_list.append(image_idx)
                camera_ray_bundle_list.append(camera_ray_bundle)
                batch_list.append(batch)
                break
        return image_idx_list, camera_ray_bundle_list, batch_list
        raise ValueError("No more eval images")

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        camera_opt_params = []
        for train_cam_opt in self.train_camera_optimizers:
            camera_opt_params.extend(list(train_cam_opt.parameters()))


        if self.config.camera_optimizer.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

        return param_groups

