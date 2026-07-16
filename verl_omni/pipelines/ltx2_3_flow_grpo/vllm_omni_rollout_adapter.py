# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
LTX-2.3 rollout-side adapter for FlowGRPO.

Subclasses the vllm-omni LTXVideoPipeline and adds per-step SDE log-probability
collection required for RL training (FlowGRPO).

.. note::

   This adapter requires ``vllm_omni`` to be installed with LTX-2.3 support.
   The base pipeline class ``LTXVideoPipeline`` must be provided by
   ``vllm_omni.diffusion.models.ltx2_3``.  If vllm-omni does not yet ship
   an LTX pipeline, this module will fail to import (which is caught by the
   ``__init__.py`` guard).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3 import LTXVideoPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import (
    DEFAULT_FPS,
    LTX2_VAE_SPATIAL_SCALE_FACTOR,
    LTX2_VAE_TEMPORAL_SCALE_FACTOR,
    apply_true_cfg,
    compute_ltx2_sigmas,
    coalesce_not_none,
)

logger = logging.getLogger(__name__)
__all__ = ["LTXVideoPipelineWithLogProb"]


@VllmOmniPipelineBase.register("LTXVideoTransformerModel", algorithm="flow_grpo")
class LTXVideoPipelineWithLogProb(LTXVideoPipeline):
    """Rollout pipeline for LTX-2.3 that captures per-step log-probabilities.

    Extends :class:`~vllm_omni.diffusion.models.ltx2_3.pipeline_ltx2_3.LTXVideoPipeline`
    with a custom SDE-based scheduler and additional output fields required for
    FlowGRPO RL training.

    Supports full-modal generation (T2V, I2V, I2AV) including audio, matching
    the training adapter's capabilities.

    Registered under ``("LTXVideoTransformerModel", "flow_grpo")``.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.device = get_local_device()
        self._interrupt = False
        model = od_config.model
        local_files_only = os.path.exists(model)

        # LTX-2.3 does not ship a scheduler subfolder; build from scratch.
        self.scheduler = FlowMatchSDEDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=False,
        )

    @property
    def interrupt(self):
        return self._interrupt

    @interrupt.setter
    def interrupt(self, value):
        self._interrupt = value

    def diffuse(
        self,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        latents,
        video_shapes,
        txt_seq_lens,
        negative_txt_seq_lens,
        timesteps,
        do_true_cfg,
        guidance,
        true_cfg_scale,
        noise_level,
        sde_window,
        sde_type,
        generator,
        logprobs,
        audio_hidden_states=None,
        audio_timesteps=None,
        audio_prompt_embeds=None,
        fps=None,
        ref_seq_len=None,
    ):
        """Run the full SDE diffusion loop and collect per-step rollout data.

        Iterates over all timesteps, optionally applying True-CFG guidance, and
        collects latents and log-probabilities within the SDE window.

        Args:
            prompt_embeds: Positive prompt embeddings ``(B, L, D)``.
            prompt_embeds_mask: Attention mask for *prompt_embeds*.
            negative_prompt_embeds: Negative prompt embeddings for CFG.
            negative_prompt_embeds_mask: Attention mask for *negative_prompt_embeds*.
            latents: Initial noisy video latents ``(B, C, F, H, W)``.
            video_shapes: Per-sample latent shape tuples.
            txt_seq_lens: Sequence lengths for positive prompt embeddings.
            negative_txt_seq_lens: Sequence lengths for negative prompt embeddings.
            timesteps: Scheduler timestep sequence.
            do_true_cfg: Whether to apply True-CFG guidance.
            guidance: Guidance scale tensor, or ``None``.
            true_cfg_scale: Classifier-free guidance scale.
            noise_level: SDE noise injection magnitude within the window.
            sde_window: ``(start, end)`` step indices for SDE noise injection.
            sde_type: SDE variant (``"sde"`` or ``"cps"``).
            generator: Optional random generator for reproducibility.
            logprobs: Whether to compute and return per-step log-probabilities.
            audio_hidden_states: Audio latents for audio generation (optional).
            audio_timesteps: Timesteps for audio generation (optional).
            audio_prompt_embeds: Audio encoder hidden states (optional).
            fps: Per-sample frames-per-second (optional).
            ref_seq_len: Per-sample reference sequence lengths for I2V (optional).

        Returns:
            tuple: ``(latents, all_latents, all_log_probs, all_timesteps)``.
        """
        all_latents = []
        all_log_probs = []
        all_timesteps = []
        self.scheduler.set_begin_index(0)

        batch_size = latents.shape[0]
        if fps is None:
            fps = [float(DEFAULT_FPS)] * batch_size
        elif not isinstance(fps, list):
            fps = [float(fps)] * batch_size

        for i, timestep_value in enumerate(timesteps):
            if self.interrupt:
                continue

            if i < sde_window[0]:
                cur_noise_level = 0.0
            elif i == sde_window[0]:
                cur_noise_level = noise_level
                all_latents.append(latents.float())
            elif i > sde_window[0] and i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            self._current_timestep = timestep_value
            timestep = timestep_value.expand(batch_size).to(
                device=latents.device, dtype=latents.dtype
            )

            # Cast to model dtype for transformer forward.
            x = latents.to(self.transformer.dtype if hasattr(self.transformer, 'dtype')
                          else next(self.transformer.parameters()).dtype)

            # Build list-format inputs for LTX-2.3 transformer.
            model_inputs = {
                "hidden_states": [x],
                "timestep": [timestep],
                "encoder_hidden_states": [prompt_embeds],
                "context_mask": [prompt_embeds_mask] if prompt_embeds_mask is not None else None,
                "fps": fps,
                "return_dict": True,
            }

            if ref_seq_len is not None:
                model_inputs["ref_seq_len"] = (
                    [int(r) for r in ref_seq_len] if torch.is_tensor(ref_seq_len)
                    else [int(ref_seq_len)]
                )

            if audio_hidden_states is not None:
                model_inputs["audio_hidden_states"] = [audio_hidden_states]
            if audio_timesteps is not None:
                model_inputs["audio_timestep"] = [audio_timesteps]
            if audio_prompt_embeds is not None:
                model_inputs["audio_encoder_hidden_states"] = [audio_prompt_embeds]

            # Positive forward.
            output = self.transformer(**model_inputs)
            if hasattr(output, "predictions") and output.predictions:
                noise_pred = output.predictions[0]
            else:
                noise_pred = output[0] if isinstance(output, (list, tuple)) else output

            # CFG forward (if enabled).
            if do_true_cfg:
                neg_model_inputs = dict(model_inputs)
                neg_model_inputs["encoder_hidden_states"] = [negative_prompt_embeds]
                neg_model_inputs["context_mask"] = (
                    [negative_prompt_embeds_mask]
                    if negative_prompt_embeds_mask is not None
                    else None
                )
                neg_output = self.transformer(**neg_model_inputs)
                if hasattr(neg_output, "predictions") and neg_output.predictions:
                    neg_noise_pred = neg_output.predictions[0]
                else:
                    neg_noise_pred = neg_output[0] if isinstance(neg_output, (list, tuple)) else neg_output
                noise_pred = apply_true_cfg(noise_pred, neg_noise_pred, true_cfg_scale)

            # Compute the previous noisy sample x_t -> x_t-1.
            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.to(torch.float32),
                timestep_value,
                latents.to(torch.float32),
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            # Save fp32 trajectory.
            if i >= sde_window[0] and i < sde_window[1]:
                all_latents.append(latents.to(torch.float32))
                all_log_probs.append(log_prob)
                all_timesteps.append(timestep_value)

        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = (
            torch.stack(all_log_probs, dim=1)
            if all_log_probs and all_log_probs[0] is not None
            else None
        )
        all_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(batch_size, -1)
        return latents, all_latents, all_log_probs, all_timesteps

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt_token_ids: torch.Tensor | list[int] | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | list[int] | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_frames: int = 41,
        num_inference_steps: int = 50,
        fps: int = 24,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_videos_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end_tensor_inputs: tuple[str, ...] = ("latents",),
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: Literal["sde", "cps"] = "sde",
        logprobs: bool = True,
        # LTX-2.3 multimodal fields
        audio_hidden_states: torch.Tensor | None = None,
        audio_timesteps: torch.Tensor | None = None,
        audio_prompt_embeds: torch.Tensor | None = None,
        ref_seq_len: int | list[int] | None = None,
        # Condition image for I2V/I2AV
        condition_images: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> DiffusionOutput:
        """End-to-end video generation with rollout data collection.

        This method mirrors :meth:`QwenImagePipelineWithLogProb.forward` but
        adapted for LTX-2.3's video/audio multimodal generation.  It:

        1. Encodes prompts (or uses pre-computed embeddings).
        2. Prepares initial latents (noise or from condition images).
        3. Configures the SDE scheduler with LTX-2.3's sigma schedule.
        4. Runs :meth:`diffuse` to generate video and collect log-probs.
        5. Decodes latents to video frames via VAE.

        Args:
            req: The diffusion request from verl-omni's rollout worker.
            prompt_token_ids: Token IDs for the positive prompt.
            prompt_mask: Attention mask for *prompt_token_ids*.
            negative_prompt_ids: Token IDs for the negative prompt (CFG).
            negative_prompt_mask: Attention mask for *negative_prompt_ids*.
            true_cfg_scale: True-CFG scale (>1 enables CFG).
            height: Output video height in pixels.
            width: Output video width in pixels.
            num_frames: Number of video frames to generate.
            num_inference_steps: Number of denoising steps.
            fps: Frames per second for the output video.
            sigmas: Optional custom sigma schedule.
            guidance_scale: Guidance scale for the transformer.
            num_videos_per_prompt: Number of videos to generate per prompt.
            generator: Optional random generator.
            latents: Optional pre-computed initial latents.
            prompt_embeds: Pre-computed positive prompt embeddings.
            prompt_embeds_mask: Mask for *prompt_embeds*.
            negative_prompt_embeds: Pre-computed negative prompt embeddings.
            negative_prompt_embeds_mask: Mask for *negative_prompt_embeds*.
            output_type: Output format (``"pil"``, ``"np"``, ``"latent"``).
            attention_kwargs: Extra attention kwargs.
            callback_on_step_end_tensor_inputs: Tensor inputs for step callback.
            noise_level: SDE noise injection magnitude.
            sde_window_size: Size of the SDE window (defaults to full range).
            sde_window_range: ``(start_fraction, end_fraction)`` of steps.
            sde_type: SDE variant (``"sde"`` or ``"cps"``).
            logprobs: Whether to compute and return log-probabilities.
            audio_hidden_states: Pre-computed audio latents.
            audio_timesteps: Timesteps for audio generation.
            audio_prompt_embeds: Audio encoder hidden states.
            ref_seq_len: Reference sequence length(s) for I2V/I2AV.
            condition_images: Condition image(s) for I2V/I2AV generation.

        Returns:
            :class:`DiffusionOutput` with generated video, latents, log-probs,
            and timesteps.
        """
        height = height or self.default_height
        width = width or self.default_width

        # 1. Encode prompts (or use pre-computed embeddings).
        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                prompt_token_ids=prompt_token_ids,
                prompt_mask=prompt_mask,
                num_videos_per_prompt=num_videos_per_prompt,
            )

        if negative_prompt_embeds is None and true_cfg_scale > 1.0:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt_token_ids=negative_prompt_ids,
                prompt_mask=negative_prompt_mask,
                num_videos_per_prompt=num_videos_per_prompt,
            )

        do_true_cfg = true_cfg_scale > 1.0

        # 2. Prepare latents.
        batch_size = prompt_embeds.shape[0] * num_videos_per_prompt
        latent_height = height // LTX2_VAE_SPATIAL_SCALE_FACTOR
        latent_width = width // LTX2_VAE_SPATIAL_SCALE_FACTOR
        latent_frames = (num_frames - 1) // LTX2_VAE_TEMPORAL_SCALE_FACTOR + 1
        latent_channels = 128  # LTX-2.3 VAE latent channels

        if latents is None:
            shape = (batch_size, latent_channels, latent_frames, latent_height, latent_width)
            latents = torch.randn(shape, device=self.device, dtype=torch.float32)

        # 3. Configure scheduler with LTX-2.3 sigma schedule.
        computed_sigmas = compute_ltx2_sigmas(num_inference_steps).to(self.device)
        self.scheduler.set_timesteps(
            num_inference_steps, device=self.device, sigmas=computed_sigmas
        )
        timesteps = self.scheduler.timesteps.to(self.device)

        # 4. Compute SDE window.
        if sde_window_size is None:
            sde_window_size = int(
                sde_window_range[1] * num_inference_steps
                - sde_window_range[0] * num_inference_steps
            )
        sde_window = (
            int(sde_window_range[0] * num_inference_steps),
            int(sde_window_range[0] * num_inference_steps) + sde_window_size,
        )

        # 5. Prepare per-sample metadata.
        fps_list = [float(fps)] * batch_size
        ref_seq_len_list = None
        if ref_seq_len is not None:
            ref_seq_len_list = (
                [int(r) for r in ref_seq_len] if isinstance(ref_seq_len, list)
                else [int(ref_seq_len)] * batch_size
            )

        # 6. Run diffusion loop.
        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            latents=latents,
            video_shapes=None,  # LTX-2.3 infers shapes from latents.
            txt_seq_lens=None,
            negative_txt_seq_lens=None,
            timesteps=timesteps,
            do_true_cfg=do_true_cfg,
            guidance=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            noise_level=noise_level,
            sde_window=sde_window,
            sde_type=sde_type,
            generator=generator,
            logprobs=logprobs,
            audio_hidden_states=audio_hidden_states,
            audio_timesteps=audio_timesteps,
            audio_prompt_embeds=audio_prompt_embeds,
            fps=fps_list,
            ref_seq_len=ref_seq_len_list,
        )

        # 7. Decode latents to video.
        if output_type == "latent":
            video = latents
        else:
            video = self.vae.decode(latents.to(self.vae.dtype) / self.vae.config.scaling_factor)
            if output_type == "np":
                video = video.cpu().numpy()
            elif output_type == "pil":
                # Convert to PIL frames (delegated to vllm-omni utils).
                video = self._tensor_to_pil_frames(video)

        return DiffusionOutput(
            video=video,
            latents=latents,
            all_latents=all_latents,
            all_log_probs=all_log_probs,
            all_timesteps=all_timesteps,
        )

    def _tensor_to_pil_frames(self, video_tensor: torch.Tensor) -> list:
        """Convert a video tensor to a list of PIL images.

        Falls back to numpy intermediate if PIL conversion is not available.
        Override or extend in subclasses for custom post-processing.
        """
        import numpy as np

        # video_tensor: (B, C, F, H, W) → (B, F, H, W, C) for frame extraction
        video_np = video_tensor.cpu().float().numpy()
        if video_np.ndim == 4:
            video_np = video_np[None]
        # Normalize to [0, 255]
        video_np = (video_np / 2 + 0.5).clip(0, 1)
        video_np = (video_np * 255).round().astype("uint8")
        # Transpose to (B, F, H, W, C)
        video_np = video_np.transpose(0, 2, 3, 4, 1)
        frames = []
        for batch in video_np:
            for frame in batch:
                try:
                    from PIL import Image
                    frames.append(Image.fromarray(frame))
                except ImportError:
                    frames.append(frame)
        return frames
