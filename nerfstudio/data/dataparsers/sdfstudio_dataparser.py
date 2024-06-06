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

"""Data parser for friends dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type
from typing_extensions import Literal

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from torchtyping import TensorType
import trimesh

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.images import BasicImages
from nerfstudio.utils.io import load_from_json
from pdb import set_trace as pause
from time import time

CONSOLE = Console()

import struct
def read_sdf_slow(filename):
    # Define the format string for unpacking the floats
    # '<' for little-endian, '4f' for four floats
    format_str = '<4f' 

    # Create an empty list to store the read values
    data = []

    # Open the file in binary mode
    with open(filename, 'rb') as file:
        while True:
            # Read 4 floats (16 bytes) at a time
            bytes = file.read(16)

            # Break the loop if we've reached the end of the file
            if not bytes:
                break

            # Unpack the bytes and append to the data list
            data.append(struct.unpack(format_str, bytes))

    return np.array(data)

def read_sdf(filename, normal=False):
    """
    Read floats from a binary file into an Nx4 numpy array efficiently.

    Args:
    - filename: String, the path to the binary file to read.

    Returns:
    - Nx4 numpy array of the read floats.
    """
    # Open the file in binary read mode
    with open(filename, 'rb') as file:
        # Read the entire file content into a bytes object
        data = file.read()

    # Convert the bytes data into a 1D numpy array of float32
    # Assuming the file contains only float32 values
    array = np.frombuffer(data, dtype=np.float32)

    # Reshape the array to Nx4 since we know each point consists of 4 floats
    # The '-1' in reshape automatically calculates the size of the first dimension
    array = array.reshape(-1, 7 if normal else 4)

    return array


def get_src_from_pairs(
    ref_idx, all_imgs, pairs_srcs, neighbors_num=None, neighbors_shuffle=False
) -> Dict[str, TensorType]:
    # src_idx[0] is ref img
    src_idx = pairs_srcs[ref_idx]
    # randomly sample neighbors
    if neighbors_num and neighbors_num > -1 and neighbors_num < len(src_idx) - 1:
        if neighbors_shuffle:
            perm_idx = torch.randperm(len(src_idx) - 1) + 1
            src_idx = torch.cat([src_idx[[0]], src_idx[perm_idx[:neighbors_num]]])
        else:
            src_idx = src_idx[: neighbors_num + 1]
    src_idx = src_idx.to(all_imgs.device)
    return {"src_imgs": all_imgs[src_idx], "src_idxs": src_idx}


def get_image(image_filename, alpha_color=None) -> TensorType["image_height", "image_width", "num_channels"]:
    """Returns a 3 channel image.

    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_filename)
    np_image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
    assert len(np_image.shape) == 3
    assert np_image.dtype == np.uint8
    assert np_image.shape[2] in [3, 4], f"Image shape of {np_image.shape} is in correct."
    image = torch.from_numpy(np_image.astype("float32") / 255.0)
    if alpha_color is not None and image.shape[-1] == 4:
        assert image.shape[-1] == 4
        image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
    else:
        image = image[:, :, :3]
    return image


def get_depths_and_normals(image_idx: int, depths, normals):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    # depth
    depth = depths[image_idx]
    # normal
    normal = normals[image_idx]

    return {"depth": depth, "normal": normal}


def get_sensor_depths(image_idx: int, sensor_depths):
    """function to process additional sensor depths

    Args:
        image_idx: specific image index to work with
        sensor_depths: semantics data
    """

    # sensor depth
    sensor_depth = sensor_depths[image_idx]

    return {"sensor_depth": sensor_depth}


def get_foreground_masks(image_idx: int, fg_masks):
    """function to process additional foreground_masks

    Args:
        image_idx: specific image index to work with
        fg_masks: foreground_masks
    """

    # sensor depth
    fg_mask = fg_masks[image_idx]

    return {"fg_mask": fg_mask}


def get_sparse_sfm_points(image_idx: int, sfm_points):
    """function to process additional sparse sfm points

    Args:
        image_idx: specific image index to work with
        sfm_points: sparse sfm points
    """

    # sfm points
    sparse_sfm_points = sfm_points[image_idx]
    sparse_sfm_points = BasicImages([sparse_sfm_points])
    return {"sparse_sfm_points": sparse_sfm_points}


def get_sdf_samples(image_idx: int, sdf_samples):
    """function to process additional sdf samples

    Args:
        image_idx: specific image index to work with
        sdf samples: sdf points
    """

    # choices = np.random.RandomState(image_idx).choice(sdf_samples.shape[0], size=10000, replace=False)
    # if image_idx == 0:
        # pause()
    # v1
    sparse_sdf_samples = sdf_samples[image_idx]
    # sparse_sdf_samples[..., 3] = -sparse_sdf_samples[..., 3]
    # v2
    # choices = np.random.choice(sdf_samples.shape[0], size=10, replace=False)
    # sparse_sdf_samples = sdf_samples[choices].reshape(-1, 4)
    sparse_sdf_samples = BasicImages([sparse_sdf_samples])
    return {"sparse_sdf_samples": sparse_sdf_samples}


@dataclass
class SDFStudioDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: SDFStudio)
    """target class to instantiate"""
    data: Path = Path("data/DTU/scan65")
    """Directory specifying location of data."""
    include_mono_prior: bool = False
    """whether or not to load monocular depth and normal """
    include_sensor_depth: bool = False
    """whether or not to load sensor depth"""
    include_foreground_mask: bool = False
    """whether or not to load foreground mask"""
    include_sfm_points: bool = False
    """whether or not to load sfm points"""
    include_sdf_samples: bool = False
    """whether or not to load sdf samples"""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    # TODO supports downsample
    # downscale_factor: Optional[int] = None
    # """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    orientation_method: Literal["up", "none"] = "up"
    """The method to use for orientation."""
    center_poses: bool = False
    """Whether to center the poses."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    load_pairs: bool = False
    """whether to load pairs for multi-view consistency"""
    neighbors_num: Optional[int] = None
    neighbors_shuffle: Optional[bool] = False
    pairs_sorted_ascending: Optional[bool] = True
    """if src image pairs are sorted in ascending order by similarity i.e. 
    the last element is the most similar to the first (ref)"""
    skip_every_for_val_split: int = 1
    """sub sampling validation images"""
    train_val_no_overlap: bool = False
    """remove selected / sampled validation images from training set"""
    auto_orient: bool = False
    """automatically orient the scene such that the up direction is the same as the viewer's up direction"""
    load_dtu_highres: bool = False
    """load high resolution images from DTU dataset, should only be used for the preprocessed DTU dataset"""
    use_point_color: bool = False
    """use point color when training with sdf samples"""


def filter_list(list_to_filter, indices):
    """Returns a copy list with only selected indices"""
    if list_to_filter:
        return [list_to_filter[i] for i in indices]
    else:
        return []


@dataclass
class SDFStudio(DataParser):
    """SDFStudio Dataset"""

    config: SDFStudioDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements
        # load meta data
        config_is_json = self.config.data.suffix == '.json'
        data_dir = self.config.data.parent if config_is_json else self.config.data
        if config_is_json:
            meta = load_from_json(self.config.data)
        else:
            meta = load_from_json(self.config.data / "meta_data.json")

        indices = list(range(len(meta["frames"])))

        # subsample to avoid out-of-memory for validation set
        if split != "train" and self.config.skip_every_for_val_split >= 1:
            indices = indices[:: self.config.skip_every_for_val_split]
        else:
            # if you use this option, training set should not contain any image in validation set
            if self.config.train_val_no_overlap:
                indices = [i for i in indices if i % self.config.skip_every_for_val_split != 0]
        # print(split, indices)

        image_filenames = []
        depth_images = []
        normal_images = []
        sensor_depth_images = []
        foreground_mask_images = []
        sfm_points = []
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        for i, frame in enumerate(meta["frames"]):
            if config_is_json:
                image_filename = data_dir / frame["rgb_path"]
            else:
                image_filename = self.config.data / frame["rgb_path"]

            intrinsics = torch.tensor(frame["intrinsics"])
            camtoworld = torch.tensor(frame["camtoworld"])

            # here is hard coded for DTU high-res images
            if self.config.load_dtu_highres:
                image_filename = self.config.data / "image" / frame["rgb_path"].replace("_rgb", "")
                intrinsics[:2, :] *= 1200 / 384.0
                intrinsics[0, 2] += 200
                height, width = 1200, 1600
                meta["height"], meta["width"] = height, width

            if self.config.include_mono_prior:
                assert meta["has_mono_prior"]
                # load mono depth
                if config_is_json:
                    depth = np.load(data_dir / frame["mono_depth_path"])
                else:
                    depth = np.load(self.config.data / frame["mono_depth_path"])
                depth_images.append(torch.from_numpy(depth).float())

                # load mono normal
                if config_is_json:
                    normal = np.load(data_dir / frame["mono_normal_path"])
                else:
                    normal = np.load(self.config.data / frame["mono_normal_path"])

                # transform normal to world coordinate system
                normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here
                normal = torch.from_numpy(normal).float()

                rot = camtoworld[:3, :3]

                normal_map = normal.reshape(3, -1)
                normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

                normal_map = rot @ normal_map
                normal_images.append(normal_map)

                '''
                normal_map = rot.to('cuda') @ normal_map.to('cuda')
                normal_map = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)
                normal_map = normal_map.to('cpu')
                '''

            if self.config.include_sensor_depth:
                assert meta["has_sensor_depth"]
                # load sensor depth
                if config_is_json:
                    sensor_depth = np.load(data_dir / frame["sensor_depth_path"])
                else:
                    sensor_depth = np.load(self.config.data / frame["sensor_depth_path"])
                sensor_depth_images.append(torch.from_numpy(sensor_depth).float())

            if self.config.include_foreground_mask:
                assert meta["has_foreground_mask"]
                # load foreground mask
                if self.config.load_dtu_highres:
                    # filenames format is 000.png
                    foreground_mask = np.array(
                        Image.open(
                            self.config.data / "mask" / frame["foreground_mask"].replace("_foreground_mask", "")[-7:]
                        ),
                        dtype="uint8",
                    )
                else:
                    # filenames format is 000000_foreground_mask.png
                    if config_is_json:
                        foreground_mask = np.array(Image.open(data_dir / frame["foreground_mask"]), dtype="uint8")
                    else:
                        foreground_mask = np.array(Image.open(self.config.data / frame["foreground_mask"]), dtype="uint8")
                foreground_mask = foreground_mask[..., :1]
                foreground_mask_images.append(torch.from_numpy(foreground_mask).float() / 255.0)

            if self.config.include_sfm_points:
                assert meta["has_sparse_sfm_points"]
                # load sparse sfm points
                if config_is_json:
                    sfm_points_view = np.loadtxt(data_dir / frame["sfm_sparse_points_view"])
                else:
                    sfm_points_view = np.loadtxt(self.config.data / frame["sfm_sparse_points_view"])
                sfm_points.append(torch.from_numpy(sfm_points_view).float())

            # append data
            image_filenames.append(image_filename)
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)

        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        camera_to_worlds = torch.stack(camera_to_worlds)

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1

        if self.config.auto_orient:
            if "orientation_override" in meta:
                orientation_method = meta["orientation_override"]
                CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
            else:
                orientation_method = self.config.orientation_method

            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method=orientation_method,
                center_poses=self.config.center_poses,
            )

            # we should also transform normal accordingly
            normal_images_aligned = []
            for normal_image in normal_images:
                h, w, _ = normal_image.shape
                normal_image = transform[:3, :3] @ normal_image.reshape(-1, 3).permute(1, 0)
                normal_image = normal_image.permute(1, 0).reshape(h, w, 3)
                normal_images_aligned.append(normal_image)
            normal_images = normal_images_aligned

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(camera_to_worlds[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        camera_to_worlds[:, :3, 3] *= scale_factor

        # scene box from meta data
        meta_scene_box = meta["scene_box"]
        aabb = torch.tensor(meta_scene_box["aabb"], dtype=torch.float32)
        scene_box = SceneBox(
            aabb=aabb,
            near=meta_scene_box["near"],
            far=meta_scene_box["far"],
            radius=meta_scene_box["radius"],
            collider_type=meta_scene_box["collider_type"],
        )

        height, width = meta["height"], meta["width"]
        cameras = Cameras(
            fx=fx[indices],
            fy=fy[indices],
            cx=cx[indices],
            cy=cy[indices],
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[indices, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        # TODO supports downsample
        # cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)

        if self.config.include_mono_prior:
            additional_inputs_dict = {
                "cues": {
                    "func": get_depths_and_normals,
                    "kwargs": {
                        "depths": filter_list(depth_images, indices),
                        "normals": filter_list(normal_images, indices),
                    },
                }
            }
        else:
            additional_inputs_dict = {}

        if self.config.include_sensor_depth:
            additional_inputs_dict["sensor_depth"] = {
                "func": get_sensor_depths,
                "kwargs": {"sensor_depths": filter_list(sensor_depth_images, indices)},
            }

        if self.config.include_foreground_mask:
            additional_inputs_dict["foreground_masks"] = {
                "func": get_foreground_masks,
                "kwargs": {"fg_masks": filter_list(foreground_mask_images, indices)},
            }

        if self.config.include_sfm_points:
            additional_inputs_dict["sfm_points"] = {
                "func": get_sparse_sfm_points,
                "kwargs": {"sfm_points": filter_list(sfm_points, indices)},
            }
        self.n_images = 0
        if self.config.include_sdf_samples:
            self.meta = meta
            self.w2gt = np.array(meta["worldtogt"])
            # sdf_fname = self.config.data / "rand_surf-4m.ply"
            # scene = "785e7504b9"
            if config_is_json:
                self.sdf_path = f"{str(data_dir)}/../../scans/rand_surf-" #-40m-v9"
            else:
                self.sdf_path = f"{str(self.config.data)}/../../scans/rand_surf-" #-40m-v9"
            n_images = len(self.meta["frames"])
            n_images = 305
            self.n_images = n_images
            if split == "train":
                # self.choices = {}
                # choices = np.arange(int(8e7))
                indices = indices[:self.n_images]
            # else:
                # choices = np.arange(280000)
            # np.random.RandomState(0).shuffle(choices)

            # self.choices[split] = choices
            # if split == "train":
            # else:
                # self.sdf_path = f"{str(self.config.data)}/../../scans/rand_surf-140k-v9"
            # self.num_samples = num_samples
            # self.size = size
            sdf_samples = self.load_sdf_samples(0, split)

            additional_inputs_dict = {
                    "sdf_samples": {
                    "func": get_sdf_samples,
                    "kwargs": {"sdf_samples": sdf_samples},
                }
            }
        else:
            self.bbox_min = (-1.0, -1.0, -1.0)
            self.bbox_max = (1.0, 1.0, 1.0)
        # load pair information
        if config_is_json:
            pairs_path = data_dir / "pairs.txt"
        else:
            pairs_path = self.config.data / "pairs.txt"
        if pairs_path.exists() and split == "train" and self.config.load_pairs:
            with open(pairs_path, "r") as f:
                pairs = f.readlines()
            split_ext = lambda x: x.split(".")[0]
            pairs_srcs = []
            for sources_line in pairs:
                sources_array = [int(split_ext(img_name)) for img_name in sources_line.split(" ")]
                if self.config.pairs_sorted_ascending:
                    # invert (flip) the source elements s.t. the most similar source is in index 1 (index 0 is reference)
                    sources_array = [sources_array[0]] + sources_array[:1:-1]
                pairs_srcs.append(sources_array)
            pairs_srcs = torch.tensor(pairs_srcs)
            # TODO: check correctness of sorting
            all_imgs = torch.stack([get_image(image_filename) for image_filename in sorted(image_filenames)], axis=0)[
                indices
            ].cuda()

            additional_inputs_dict["pairs"] = {
                "func": get_src_from_pairs,
                "kwargs": {
                    "all_imgs": all_imgs,
                    "pairs_srcs": pairs_srcs,
                    "neighbors_num": self.config.neighbors_num,
                    "neighbors_shuffle": self.config.neighbors_shuffle,
                },
            }

        dataparser_outputs = DataparserOutputs(
            image_filenames=filter_list(image_filenames, indices),
            cameras=cameras,
            scene_box=scene_box,
            additional_inputs=additional_inputs_dict,
            depths=filter_list(depth_images, indices),
            normals=filter_list(normal_images, indices),
            bbox_min=self.bbox_min,
            bbox_max=self.bbox_max,
            # sdf_samples_len= self.n_images
        )
        return dataparser_outputs


    def load_sdf_samples(self, part, split):
        print(f'Loading sdf samples from part {part}')
        pnum = "40m" if split=="train" else "140k"
        postfix = "-v9"
        if self.config.use_point_color:
            postfix += "-rgb"
        # TODO eval load smaller size
        path = self.sdf_path + pnum + postfix  #  + "-v9"
        path = f"{path}-{part}.ply"
        sdf_fname = path.replace("rand_surf", "near_surf")[:-3].replace("-cur1", "").replace(postfix, "-exp5") + "sdf"
        print(f"loading {path}, {sdf_fname}")
        # sdf_onsurface = mesh.vertices.astype(np.float32)
        k = int(4e7)
        # sdf_onsurface = np.zeros((k,3))
        # sdf_offsurface = np.zeros((k, 4))
        mesh = trimesh.load(path, process=False)
        sdf_onsurface = torch.from_numpy(mesh.vertices).float()
        sdf_offsurface = read_sdf(sdf_fname)
        n_onsurface = sdf_onsurface.shape[0]
        n_offsurface = sdf_offsurface.shape[0]
        # sdf_samples = np.zeros((n_onsurface + n_offsurface, 4), dtype=np.float32)
        dim = 4 + self.config.use_point_color * 6
        sdf_samples = torch.zeros((n_onsurface + n_offsurface, dim)).float()
        sdf_samples[:n_onsurface, :3] = sdf_onsurface
        sdf_samples[n_onsurface:, :4] = torch.from_numpy(sdf_offsurface * 1.0).float()


        # w2gt = self.w2gt 
        # sdf_samples[:, :3] = (sdf_samples[:, :3] - w2gt[:3, 3][None]) / np.diag(w2gt)[:3][None]
        # sdf_samples[:, 3] = sdf_samples[:, 3] / w2gt[0, 0]
        w2gt = torch.from_numpy(self.w2gt).float()
        sdf_samples[:, :3] = (sdf_samples[:, :3] - w2gt[:3, 3][None]) #/ np.diag(w2gt)[:3][None]
        sdf_samples[:, :4] = sdf_samples[:, :4] / w2gt[0, 0]
        n_sdf_samples = sdf_samples.shape[0]
        # choices = self.choices[split]
        if split == "train":
            n_images = self.n_images
            npoints_per_image = 2**18 #n_sdf_samples // n_images
            if part == 0:
                # bbox_min = sdf_samples[:n_onsurface, :3].min(0)
                # bbox_max = sdf_samples[:n_onsurface, :3].max(0)
                bbox_min = sdf_samples[:n_onsurface, :3].min(0)[0]
                bbox_max = sdf_samples[:n_onsurface, :3].max(0)[0]
                self.bbox_min = tuple([i.item() for i in bbox_min])
                self.bbox_max = tuple([i.item() for i in bbox_max])
        else:
            n_images = 2000
            npoints_per_image = 60
        # choices = self.choices
        if self.config.use_point_color:
            # rgb
            colors_onsurface = torch.from_numpy(mesh.colors[:, :3]).float() / 255.0
            # colors = torch.zeros_like(sdf_samples[:, :3])
            # colors[:n_onsurface] = colors_onsurface
            sdf_samples[:n_onsurface, 4:7] = colors_onsurface
            # normal
            # data=mesh.metadata['_ply_raw']['vertex']['data']
            # normals = torch.tensor([[idata[3], idata[4], idata[5]] for idata in data])
            path_norm = path[:-3] + "npy"
            if Path(path_norm).exists():
                normals = np.load(path_norm)
            else:
                data=mesh.metadata['_ply_raw']['vertex']['data']
                normals = np.array([[idata[3], idata[4], idata[5]] for idata in data], dtype=np.float32)
                np.save(path_norm, normals)
            sdf_samples[:n_onsurface, 7:10] = torch.tensor(normals)
            # sdf_samples = torch.concatenate((sdf_samples, colors), 1)
        # sdf_samples = torch.from_numpy(sdf_samples).float()
        # sdf_samples = sdf_samples[choices][: npoints_per_image * n_images].reshape(n_images, npoints_per_image, -1)
        sdf_samples = sdf_samples[: npoints_per_image * n_images].reshape(npoints_per_image, n_images, -1).permute(1,0,2)
        return sdf_samples
