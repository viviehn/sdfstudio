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
Code to train model.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import time
from typing import Dict, List, Tuple
from glob import glob

import torch
from rich.console import Console
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from nerfstudio.configs import base_config as cfg
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers, setup_optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import (
    check_eval_enabled,
    check_main_thread,
    check_viewer_enabled,
)
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.server import viewer_utils
from torch_ema import ExponentialMovingAverage
from pdb import set_trace as pause

CONSOLE = Console(width=120)


class Trainer:
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
    """

    pipeline: VanillaPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: cfg.Config, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = "cpu" if world_size == 0 else f"cuda:{local_rank}"
        self.mixed_precision = self.config.trainer.mixed_precision
        if self.device == "cpu":
            self.mixed_precision = False
            CONSOLE.print("Mixed precision is disabled for CPU training.")
        self._start_step = 0
        # optimizers
        self.grad_scaler = GradScaler(enabled=self.mixed_precision)

        self.base_dir = config.get_base_dir()
        # directory to save checkpoints
        self.checkpoint_dir = config.get_checkpoint_dir()
        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")
        # set up viewer if enabled
        viewer_log_path = self.base_dir / config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_enabled() and local_rank == 0:
            self.viewer_state, banner_messages = viewer_utils.setup_viewer(config.viewer, log_filename=viewer_log_path)
        self._check_viewer_warnings()
        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / config.logging.relative_log_dir
        writer.setup_event_writer(config, log_dir=writer_log_path)
        writer.setup_local_writer(
            config.logging, max_iter=config.trainer.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(config), step=0)
        profiler.setup_profiler(config.logging)

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val"):
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datset into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device, test_mode=test_mode, world_size=self.world_size, local_rank=self.local_rank
        )
        self.optimizers = setup_optimizers(self.config, self.pipeline.get_param_groups())

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,  # type: ignore
                grad_scaler=self.grad_scaler,  # type: ignore
                pipeline=self.pipeline,  # type: ignore
                config=self.config.trainer,  # type: ignore
            )
        )
        if False:
        # if True:
            ema_decay = 0.95
            self.ema = ExponentialMovingAverage(self.pipeline.model.field.parameters(), decay=ema_decay) 
        else:
            self.ema = None

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.trainer.max_num_iterations
            step = 0
            # TODO loss weight
            n_images = self.pipeline.datamanager.dataparser.n_images 
            for step in range(self._start_step, self._start_step + num_iterations):
                with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                    # if n_images > 0 and step > 0 and step % n_images ==0:
                        # part = step // n_images
                        # self.pipeline.datamanager.train_image_dataloader.dataset._dataparser_outputs.additional_inputs['sdf_samples']["kwargs"]["sdf_samples"]=self.pipeline.datamanager.dataparser.load_sdf_samples(part, "train")
                    self.pipeline.train()

                    # training callbacks before the training iteration
                    for callback in self.callbacks:
                        callback.run_callback_at_location(
                            step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                        )

                    # time the forward pass
                    loss, loss_dict, metrics_dict = self.train_iteration(step)

                    # training callbacks after the training iteration
                    for callback in self.callbacks:
                        callback.run_callback_at_location(step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.config.pipeline.datamanager.train_num_rays_per_batch / train_t.duration,
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)

                self.eval_iteration(step)

                if step_check(step, self.config.trainer.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()
            # save checkpoint at the end of training
            self.save_checkpoint(step)

            CONSOLE.rule()
            CONSOLE.print("[bold green]:tada: :tada: :tada: Training Finished :tada: :tada: :tada:", justify="center")
            if not self.config.viewer.quit_on_train_completion:
                CONSOLE.print("Use ctrl+c to quit", justify="center")
                self._always_render(step)

    @check_main_thread
    def _always_render(self, step):
        if self.config.is_viewer_enabled():
            while True:
                self.viewer_state.vis["renderingState/isTraining"].write(False)
                self._update_viewer_state(step)

    @check_main_thread
    def _check_viewer_warnings(self) -> None:
        """Helper to print out any warnings regarding the way the viewer/loggers are enabled"""
        if self.config.is_viewer_enabled():
            string = (
                "[NOTE] Not running eval iterations since only viewer is enabled."
                " Use [yellow]--vis wandb[/yellow] or [yellow]--vis tensorboard[/yellow] to run with eval instead."
            )
            CONSOLE.print(f"{string}")

    @check_viewer_enabled
    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        assert self.viewer_state and self.pipeline.datamanager.train_dataset
        self.viewer_state.init_scene(
            dataset=self.pipeline.datamanager.train_dataset,
            start_train=self.config.viewer.start_train,
        )
        if not self.config.viewer.start_train:
            self._always_render(self._start_step)

    @check_viewer_enabled
    def _update_viewer_state(self, step: int):
        """Updates the viewer state by rendering out scene with current pipeline
        Returns the time taken to render scene.

        Args:
            step: current train step
        """
        assert self.viewer_state is not None
        with TimeWriter(writer, EventName.ITER_VIS_TIME, step=step) as _:
            num_rays_per_batch = self.config.pipeline.datamanager.train_num_rays_per_batch
            try:
                self.viewer_state.update_scene(self, step, self.pipeline.model, num_rays_per_batch)
            except RuntimeError:
                time.sleep(0.03)  # sleep to allow buffer to reset
                assert self.viewer_state.vis is not None
                self.viewer_state.vis["renderingState/log_errors"].write(
                    "Error: GPU out of memory. Reduce resolution to prevent viewer from crashing."
                )

    @check_viewer_enabled
    def _update_viewer_rays_per_sec(self, train_t: TimeWriter, vis_t: TimeWriter, step: int):
        """Performs update on rays/sec calclation for training

        Args:
            train_t: timer object carrying time to execute total training iteration
            vis_t: timer object carrying time to execute visualization step
            step: current step
        """
        train_num_rays_per_batch = self.config.pipeline.datamanager.train_num_rays_per_batch
        writer.put_time(
            name=EventName.TRAIN_RAYS_PER_SEC,
            duration=train_num_rays_per_batch / (train_t.duration - vis_t.duration),
            step=step,
            avg_over_steps=True,
        )

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.trainer.load_dir
        if "ngp_models" in str(load_dir):
            load_path = glob(str(load_dir) + "/*")[0]
            loaded_state = torch.load(load_path, map_location="cpu")
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            # pause()
            self.pipeline._model.field.glin0.weight.data = loaded_state["model"]['backbone.0.weight'].cuda()
            self.pipeline._model.field.glin1.weight.data = loaded_state["model"]['backbone.1.weight'].cuda()
            if self.config.pipeline.model.sdf_field.fix_geonet:
                self.pipeline._model.field.glin0.weight.requires_grad = False
                self.pipeline._model.field.glin1.weight.requires_grad = False
            if 'backbone.2.weight' in loaded_state["model"].keys():
                self.pipeline._model.field.glin2.weight.data = loaded_state["model"]['backbone.2.weight'].cuda()[:1]
                if self.config.pipeline.model.sdf_field.fix_geonet:
                    self.pipeline._model.field.glin2.weight.requires_grad = False
                self.pipeline._model.field.glin3.weight.data = loaded_state["model"]['backbone.2.weight'].cuda()[1:]
                if self.config.pipeline.model.sdf_field.fix_geonet:
                    self.pipeline._model.field.glin3.weight.requires_grad = False
            # else:
                # self.pipeline._model.field.glin1.weight.data[:1] = loaded_state["model"]['backbone.1.weight'].cuda()
                
            CONSOLE.print(f"done loading checkpoint from {load_path}")

        elif load_dir is not None:
            load_step = self.config.trainer.load_step
            if load_step is None:
                print("Loading latest checkpoint from load_dir")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"])
            if self.config.pipeline.model.sdf_field.fix_geonet:
                self.pipeline._model.field.glin0.weight.requires_grad = False
                self.pipeline._model.field.glin1.weight.requires_grad = False
                self.pipeline._model.field.glin2.weight.requires_grad = False
                if not self.config.pipeline.model.sdf_field.vanilla_ngp:
                    self.pipeline._model.field.glin0.bias.requires_grad = False
                    self.pipeline._model.field.glin1.bias.requires_grad = False
                    self.pipeline._model.field.glin2.bias.requires_grad = False
                else:
                    self.pipeline._model.field.glin3.weight.requires_grad = False
                self.pipeline._model.field.clin0.weight.requires_grad = False
                self.pipeline._model.field.clin0.bias.requires_grad = False
                self.pipeline._model.field.clin1.weight.requires_grad = False
                self.pipeline._model.field.clin1.bias.requires_grad = False
                self.pipeline._model.field.clin2.weight.requires_grad = False
                self.pipeline._model.field.clin2.bias.requires_grad = False
            if True:
                print("counting training step from 0, didn't load optimizers and schedulers")
                return
            self._start_step = loaded_state["step"] + 1
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.trainer.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"done loading checkpoint from {load_path}")
        else:
            CONSOLE.print("No checkpoints to load, training from scratch")

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.trainer.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    @profiler.time_function
    def train_iteration(self, step: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """
        self.optimizers.zero_grad_all()
        cpu_or_cuda_str = self.device.split(":")[0]
        for _ in range(self.config.trainer.accumulate_grad_steps):
            with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
                loss = functools.reduce(torch.add, loss_dict.values())
            self.grad_scaler.scale(loss).backward()  # type: ignore
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)
        self.grad_scaler.update()
        # pause()
        if self.ema is not None:
            self.ema.update()
        self.optimizers.scheduler_step_all(step)


        # only return the last accumulate_grad_step's loss and metric for logging
        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step):
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.trainer.steps_per_eval_batch, run_at_zero=self.config.trainer.sanity_check):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.trainer.steps_per_eval_image, run_at_zero=self.config.trainer.sanity_check):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.trainer.steps_per_eval_all_images):
            metrics_dict, _ = self.pipeline.get_average_eval_image_metrics(step=step)
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)
