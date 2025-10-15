# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy
from typing import Any

import torch

from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.pipelines.basic.wan.wan_i2v_pipeline import (
    WanImageToVideoPipeline)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TrainingBatch
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.utils import is_vsa_available, shallow_asdict

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class WanI2VTrainingPipeline(TrainingPipeline):
    """
    A training pipeline for Wan.
    """
    _required_config_modules = ["scheduler", "transformer", "vae"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def set_schemas(self):
        self.train_dataset_schema = pyarrow_schema_i2v

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)

        args_copy.inference_mode = True
        args_copy.dit_cpu_offload = True
        # args_copy.pipeline_config.vae_config.load_encoder = False
        # validation_pipeline = WanImageToVideoValidationPipeline.from_pretrained(
        self.validation_pipeline = WanImageToVideoPipeline.from_pretrained(
            training_args.model_path,
            args=None,
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            dit_cpu_offload=True)

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        latents = batch['vae_latent']
        latents = latents[:, :, :self.training_args.num_latent_t]
        encoder_hidden_states = batch['text_embedding']
        encoder_attention_mask = batch['text_attention_mask']
        clip_features = batch['clip_feature']
        image_latents = batch['first_frame_latent']
        image_latents = image_latents[:, :, :self.training_args.num_latent_t]
        pil_image = batch['pil_image']
        infos = batch['info_list']

        training_batch.latents = latents.to(get_local_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.encoder_hidden_states = encoder_hidden_states.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.preprocessed_image = pil_image.to(
            get_local_torch_device())
        training_batch.image_embeds = clip_features.to(get_local_torch_device())
        training_batch.image_latents = image_latents.to(
            get_local_torch_device())
        training_batch.infos = infos

        return training_batch

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        """Override to properly handle I2V concatenation - call parent first, then concatenate image conditioning."""

        # First, call parent method to prepare noise, timesteps, etc. for video latents
        training_batch = super()._prepare_dit_inputs(training_batch)

        assert isinstance(training_batch.image_latents, torch.Tensor)
        image_latents = training_batch.image_latents.to(
            get_local_torch_device(), dtype=torch.bfloat16)

        temporal_compression_ratio = 4
        num_frames = (self.training_args.num_latent_t -
                      1) * temporal_compression_ratio + 1
        batch_size, num_channels, _, latent_height, latent_width = image_latents.shape
        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height,
                                   latent_width)
        mask_lat_size[:, :, 1:] = 0

        first_frame_mask = mask_lat_size[:, :, :1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=temporal_compression_ratio)
        mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:]],
                                  dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1,
                                           temporal_compression_ratio,
                                           latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(
            image_latents.device).to(dtype=torch.bfloat16)

        training_batch.noisy_model_input = torch.cat(
            [training_batch.noisy_model_input, mask_lat_size, image_latents],
            dim=1)

        return training_batch

    def _build_input_kwargs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:

        # Image Embeds for conditioning
        image_embeds = training_batch.image_embeds
        assert torch.isnan(image_embeds).sum() == 0
        image_embeds = image_embeds.to(get_local_torch_device(),
                                       dtype=torch.bfloat16)
        encoder_hidden_states_image = image_embeds

        # NOTE: noisy_model_input already contains concatenated image_latents from _prepare_dit_inputs
        training_batch.input_kwargs = {
            "hidden_states":
            training_batch.noisy_model_input,
            "encoder_hidden_states":
            training_batch.encoder_hidden_states,
            "timestep":
            training_batch.timesteps.to(get_local_torch_device(),
                                        dtype=torch.bfloat16),
            "encoder_attention_mask":
            training_batch.encoder_attention_mask,
            "encoder_hidden_states_image":
            encoder_hidden_states_image,
            "return_dict":
            False,
        }
        return training_batch

    def _prepare_validation_batch(self, sampling_param: SamplingParam,
                                  training_args: TrainingArgs,
                                  validation_batch: dict[str, Any],
                                  num_inference_steps: int) -> ForwardBatch:
        sampling_param.prompt = validation_batch['prompt']
        sampling_param.height = training_args.num_height
        sampling_param.width = training_args.num_width
        sampling_param.image_path = validation_batch['video_path']
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "video"
        assert self.seed is not None
        sampling_param.seed = self.seed

        latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                        sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
        temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (training_args.num_latent_t -
                      1) * temporal_compression_factor + 1
        sampling_param.num_frames = num_frames
        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
        )

        return batch


def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = WanI2VTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
