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

"""VeOmni-specific helpers for LTX-2.3 FlowGRPO.

This module is **not** registered with ``DiffusionModelBase`` — it provides
helper functions and monkey-patches consumed by :mod:`diffusers_training_adapter`
when the module is detected as VeOmni's ``LTX2VideoTransformer3DModel``.

VeOmni's transformer has a materially different ``forward()`` signature from
diffusers' ``LTXVideoTransformerModel``:

* inputs are **per-sample lists** (length == batch size, each element with
  batch dim == 1) rather than a single batched tensor;
* ``hidden_states`` elements are 5-D ``(1, C, F, H, W)`` so H/W/F are inferred
  from the tensor shape — ``height``/``width``/``num_frames`` are not accepted;
* the audio timestep is passed via ``audio_timestep`` (there is no ``sigma``
  argument);
* the output is a dataclass with ``.predictions`` / ``.audio_predictions``
  lists rather than a ``(video, audio)`` tuple.
"""

from __future__ import annotations

import torch
from diffusers import ModelMixin
from tensordict import TensorDict

from .common import apply_x0_cfg, remap_veomni_to_diffusers_key

__all__ = [
    "is_veomni_module",
    "prepare_veomni_inputs",
    "predict_veomni",
    "forward_and_sample_previous_step",
    "convert_export_key",
    "configure_train_mode",
]


def is_veomni_module(module: torch.nn.Module) -> bool:
    """Detect whether *module* is VeOmni's LTX2 transformer.

    VeOmni's ``LTXModel`` exposes ``_process_transformer_blocks``, while
    diffusers' ``LTXVideoTransformerModel`` does not.
    """
    return hasattr(module, "_process_transformer_blocks")


def convert_export_key(name: str) -> str:
    """Remap VeOmni parameter names to diffusers naming for the rollout loader."""
    return remap_veomni_to_diffusers_key(name)


def configure_train_mode(module: torch.nn.Module) -> None:
    """Keep gradient checkpointing enabled for VeOmni full-weight training.

    VeOmni's LTX2.3 model gates checkpointing on
    ``self.gradient_checkpointing and self.training``.  Previously this
    method forced ``gradient_checkpointing=False`` in train mode to avoid a
    train/eval forward-output mismatch: the model hard-codes
    ``use_reentrant=True`` in ``_process_transformer_blocks``, and the
    FSDP2 unshard/reshard timing inside reentrant checkpoint differs from
    the direct (non-checkpointed) path used in eval mode, inflating the
    PPO ratio at step 1.

    The mismatch is now resolved at the engine level instead of by
    disabling checkpointing:

    * :class:`~verl_omni.workers.engine.veomni.diffusion_impl.EngineEvalModeCtx`
      sets ``inner.training=True`` after ``module.eval()`` so eval-mode
      forward takes the same checkpointed path as train mode (dropout is
      still disabled because inner *blocks* keep ``training=False``).
    * :meth:`~verl_omni.workers.engine.veomni.diffusion_impl.VeOmniDiffusionEngine.forward_backward_batch`
      runs eval passes under ``torch.enable_grad()`` instead of
      ``torch.no_grad()`` so FSDP2 sees the same gradient-enabled state
      in both modes, eliminating the unshard/reshard timing divergence.
    * :meth:`~verl_omni.workers.engine.veomni.diffusion_impl.VeOmniDiffusionEngine.forward_step`
      detaches outputs after each eval step to release the autograd
      graph that ``enable_grad()`` would otherwise retain.

    With train and eval sharing an identical forward path, gradient
    checkpointing can stay enabled — avoiding OOM in full-weight training
    while preserving PPO ratio correctness.

    This is a no-op for diffusers (FSDP) models.
    """
    return


def prepare_veomni_inputs(
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    negative_prompt_embeds_mask: torch.Tensor | None,
    micro_batch,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    frame_rate: float,
    guidance_scale: float,
) -> tuple[dict, dict | None]:
    """Build per-sample-list inputs for VeOmni's LTX2VideoTransformer3DModel."""
    B = video_latents.shape[0]
    C = video_latents.shape[2]
    video_latents_5d = video_latents.reshape(B, latent_frames, latent_height, latent_width, C).permute(0, 4, 1, 2, 3)

    audio_C = 8
    audio_mel_bins = 16
    audio_latents_4d = audio_latents.reshape(B, -1, audio_C, audio_mel_bins).permute(0, 2, 1, 3)

    timestep_list = [timestep[i : i + 1] for i in range(B)]
    common = {
        "hidden_states": [video_latents_5d[i : i + 1] for i in range(B)],
        "audio_hidden_states": [audio_latents_4d[i : i + 1] for i in range(B)],
        "timestep": timestep_list,
        "audio_timestep": timestep_list,
        "fps": [frame_rate] * B,
    }
    model_inputs = {
        **common,
        "encoder_hidden_states": [prompt_embeds[i : i + 1] for i in range(B)],
        "audio_encoder_hidden_states": [micro_batch["audio_prompt_embeds"][i : i + 1] for i in range(B)],
    }

    if guidance_scale <= 1.0:
        return model_inputs, None
    if negative_prompt_embeds is None or negative_prompt_embeds_mask is None:
        raise ValueError("LTX-2.3 CFG requires negative prompt embeddings and attention masks.")
    if "negative_audio_prompt_embeds" not in micro_batch:
        raise KeyError("LTX-2.3 CFG requires `negative_audio_prompt_embeds` from rollout.")
    negative_model_inputs = {
        **common,
        "encoder_hidden_states": [negative_prompt_embeds[i : i + 1] for i in range(B)],
        "audio_encoder_hidden_states": [
            micro_batch["negative_audio_prompt_embeds"][i : i + 1] for i in range(B)
        ],
    }
    return model_inputs, negative_model_inputs


def predict_veomni(
    module: ModelMixin, model_inputs: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run VeOmni's LTX transformer and return float32 video/audio velocities in 3D format."""
    output = module(**model_inputs)
    if isinstance(output, tuple):
        video_pred, audio_pred = output
    else:
        preds = output.predictions or []
        video_pred = torch.cat(preds, dim=0) if preds else preds
        audio_preds = output.audio_predictions or []
        audio_pred = torch.cat(audio_preds, dim=0) if audio_preds else None

    video_pred = video_pred.float()
    audio_pred = audio_pred.float() if audio_pred is not None else None

    if video_pred.ndim == 5:
        B, C, F, H, W = video_pred.shape
        video_pred = video_pred.permute(0, 2, 3, 4, 1).reshape(B, F * H * W, C)

    if audio_pred is not None and audio_pred.ndim == 4:
        B, C, F, M = audio_pred.shape
        audio_pred = audio_pred.permute(0, 2, 1, 3).reshape(B, F, C * M)

    return video_pred, audio_pred


def forward_and_sample_previous_step(
    module: ModelMixin,
    scheduler,
    model_config,
    model_inputs: dict[str, torch.Tensor],
    negative_model_inputs: dict[str, torch.Tensor] | None,
    scheduler_inputs: TensorDict | dict[str, torch.Tensor] | None,
    step: int,
):
    """Recompute one selected CPS/SDE transition and its joint log-probability for VeOmni."""
    if scheduler_inputs is None:
        raise ValueError("LTX-2.3 FlowGRPO requires rollout scheduler inputs.")

    video_latents = torch.cat(model_inputs["hidden_states"], dim=0).float()
    audio_latents = torch.cat(model_inputs["audio_hidden_states"], dim=0).float()

    video_pred, audio_pred = predict_veomni(module, model_inputs)

    if video_latents.ndim == 5:
        B, C, F, H, W = video_latents.shape
        video_latents = video_latents.permute(0, 2, 3, 4, 1).reshape(B, F * H * W, C)

    if audio_latents.ndim == 4:
        B, C, F, M = audio_latents.shape
        audio_latents = audio_latents.permute(0, 2, 1, 3).reshape(B, F, C * M)

    guidance_scale = model_config.pipeline.guidance_scale or 1.0
    if guidance_scale > 1.0:
        if negative_model_inputs is None:
            raise ValueError("LTX-2.3 CFG requires negative model inputs.")
        negative_video_pred, negative_audio_pred = predict_veomni(module, negative_model_inputs)
        sigma = (torch.cat(model_inputs["timestep"], dim=0).float() / 1000.0).view(-1, 1, 1)
        video_pred = apply_x0_cfg(video_latents, video_pred, negative_video_pred, sigma, guidance_scale)
        audio_pred = apply_x0_cfg(audio_latents, audio_pred, negative_audio_pred, sigma, guidance_scale)

    current = torch.cat([video_latents, audio_latents], dim=1)
    model_output = torch.cat([video_pred, audio_pred], dim=1)
    next_sample = scheduler_inputs["all_next_latents"][:, step].float()
    timestep = scheduler_inputs["all_timesteps"][:, step]
    _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
        sample=current,
        model_output=model_output,
        timestep=timestep,
        noise_level=model_config.algo.noise_level,
        prev_sample=next_sample,
        sde_type=model_config.algo.sde_type,
        return_logprobs=True,
        return_sqrt_dt=True,
    )
    return log_prob, prev_sample_mean, std_dev_t, sqrt_dt
