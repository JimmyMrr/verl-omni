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
LTX-2.3 training-side adapter for diffusers-based diffusion RL (FlowGRPO).

LTX-2.3 is a multimodal video/audio diffusion transformer that accepts
``list[Tensor]`` inputs (one tensor per sample in the batch) rather than
batched tensors.  This adapter bridges the gap between verl-omni's batched
pipeline and LTX-2.3's list-based forward signature.

LTX-2.3 uses non-concat I2V/I2AV conditioning: reference frame information
is conveyed via ``ref_seq_len`` and the transformer internally splits the
latent.  Because this does not fit the concat-crop pattern of
``DiffusionI2IModelBase``, we inherit from ``DiffusionModelBase`` and handle
all condition fields (fps, ref_seq_len, audio) directly in
``prepare_model_inputs``.
"""

from typing import Optional

import torch
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import (
    DEFAULT_FPS,
    apply_true_cfg,
    compute_ltx2_sigmas,
)

__all__ = ["LTX23FlowGRPO"]


def _build_ltx2_scheduler() -> FlowMatchSDEDiscreteScheduler:
    """Instantiate a FlowMatchSDEDiscreteScheduler with LTX-2.3 defaults.

    LTX-2.3 does not ship a ``scheduler`` subfolder; its sigma schedule is
    computed from hard-coded shift/stretch parameters (see
    :func:`compute_ltx2_sigmas`).  We create the scheduler with minimal
    defaults and override sigmas at ``set_timesteps`` time.
    """
    return FlowMatchSDEDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )


@DiffusionModelBase.register("LTXVideoTransformerModel", algorithm="flow_grpo")
class LTX23FlowGRPO(DiffusionModelBase):
    """Training adapter for LTX-2.3 with FlowGRPO.

    Supports full-modal generation (T2V, I2V, I2AV) including audio.
    The adapter converts verl-omni's batched tensor inputs into the
    ``list[Tensor]`` format expected by
    :class:`~veomni.models.diffusers.ltx2_3.LTXVideoTransformerModel.forward`.

    Registered under ``("LTXVideoTransformerModel", "flow_grpo")``.
    """

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        """Build the SDE scheduler with LTX-2.3's sigma schedule."""
        scheduler = _build_ltx2_scheduler()
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler, model_config: DiffusionModelConfig, device: str):
        """Configure timesteps using LTX-2.3's shifted-and-stretched sigmas."""
        num_inference_steps = model_config.pipeline.num_inference_steps
        sigmas = compute_ltx2_sigmas(num_inference_steps).to(device)
        # FlowMatchEulerDiscreteScheduler.set_timesteps accepts a sigmas override.
        scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas)

    # ------------------------------------------------------------------
    # Model input preparation
    # ------------------------------------------------------------------

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        """Build LTX-2.3-specific inputs for the transformer forward pass.

        Converts verl-omni's batched tensors into the ``list[Tensor]`` format
        expected by ``LTXVideoTransformerModel.forward``.  Each list contains
        a single batched tensor (the transformer iterates per-sample
        internally).

        Supports T2V, I2V (via ``ref_seq_len``), and I2AV (audio fields)
        conditioning.  All condition fields are extracted from ``micro_batch``
        and passed directly as model-input keys — no separate
        ``prepare_condition`` / ``inject_condition`` cycle is needed because
        LTX-2.3 uses non-concat conditioning.

        Args:
            module: The LTX-2.3 transformer module.
            model_config: Configuration providing guidance scale and settings.
            latents: Full latent trajectory of shape ``(B, T, C, F, H, W)``.
            timesteps: Full timestep trajectory of shape ``(B, T)``.
            prompt_embeds: Positive prompt embeddings ``(B, L, D)``.
            prompt_embeds_mask: Attention mask for *prompt_embeds*.
            negative_prompt_embeds: Negative prompt embeddings for CFG.
            negative_prompt_embeds_mask: Attention mask for *negative_prompt_embeds*.
            micro_batch: Micro-batch metadata (fps, ref_seq_len, audio fields, etc.).
            step: Current denoising step index.

        Returns:
            ``(model_inputs, negative_model_inputs)`` dicts ready for the
            transformer forward call.
        """
        true_cfg_scale = model_config.pipeline.get("true_cfg_scale", 1.0)
        do_true_cfg = true_cfg_scale > 1.0

        # Slice to current denoising step.
        # latents shape: (B, T, C, F, H, W) -> (B, C, F, H, W)
        hidden_states = latents[:, step]
        # timesteps shape: (B, T) -> (B,)
        timestep = timesteps[:, step]

        # Convert batched tensors to lists (LTX-2.3 forward signature).
        # Each list has one element representing the full batch.
        model_inputs: dict = {
            "hidden_states": [hidden_states],
            "timestep": [timestep],
            "encoder_hidden_states": [prompt_embeds],
            "context_mask": [prompt_embeds_mask] if prompt_embeds_mask is not None else None,
            "return_dict": True,
        }

        # fps: per-sample frames-per-second (required for positional encoding).
        fps = micro_batch.get("fps")
        if fps is not None:
            model_inputs["fps"] = (
                [float(f) for f in fps] if torch.is_tensor(fps) else [float(fps)]
            )
        else:
            model_inputs["fps"] = [float(DEFAULT_FPS)]

        # ref_seq_len: for I2V/I2AV conditioning (non-concat).
        # The transformer uses this to split hidden_states into ref / target.
        ref_seq_len = micro_batch.get("ref_seq_len")
        if ref_seq_len is not None:
            model_inputs["ref_seq_len"] = (
                [int(r) for r in ref_seq_len]
                if torch.is_tensor(ref_seq_len)
                else [int(ref_seq_len)]
            )

        # Audio fields (when with_audio=True).
        audio_latents = micro_batch.get("audio_latents")
        if audio_latents is not None:
            # audio_latents may be a trajectory (B, T, ...) or a single step.
            if audio_latents.dim() > 3 and audio_latents.shape[1] == latents.shape[1]:
                audio_latents = audio_latents[:, step]
            model_inputs["audio_hidden_states"] = [audio_latents]

        audio_timesteps = micro_batch.get("audio_timesteps")
        if audio_timesteps is not None:
            if audio_timesteps.dim() > 1 and audio_timesteps.shape[1] == timesteps.shape[1]:
                audio_timesteps = audio_timesteps[:, step]
            model_inputs["audio_timestep"] = [audio_timesteps]

        audio_prompt_embeds = micro_batch.get("audio_prompt_embeds")
        if audio_prompt_embeds is not None:
            model_inputs["audio_encoder_hidden_states"] = [audio_prompt_embeds]

        # Build negative model inputs for CFG.
        if do_true_cfg:
            negative_model_inputs: dict = {
                "hidden_states": [hidden_states],
                "timestep": [timestep],
                "encoder_hidden_states": [negative_prompt_embeds],
                "context_mask": (
                    [negative_prompt_embeds_mask]
                    if negative_prompt_embeds_mask is not None
                    else None
                ),
                "return_dict": True,
                "fps": model_inputs["fps"],
            }
            # Share condition fields with negative inputs.
            for cond_key in (
                "ref_seq_len",
                "audio_hidden_states",
                "audio_timestep",
                "audio_encoder_hidden_states",
            ):
                if cond_key in model_inputs:
                    negative_model_inputs[cond_key] = model_inputs[cond_key]
        else:
            negative_model_inputs = {}

        return model_inputs, negative_model_inputs

    # ------------------------------------------------------------------
    # Forward & sampling
    # ------------------------------------------------------------------

    @classmethod
    def forward(
        cls,
        module,
        model_config: DiffusionModelConfig,
        model_inputs: dict,
        negative_model_inputs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run a single LTX-2.3 transformer forward pass.

        LTX-2.3's forward accepts ``list[Tensor]`` inputs and returns an
        :class:`LTXVideoModelOutput` whose ``predictions`` field is a
        ``list[Tensor]`` (one per sample).  We extract the first (and
        typically only) prediction, which is the noise prediction for the
        full batch.

        Args:
            module: The LTX-2.3 transformer.
            model_config: Model configuration.
            model_inputs: Dict with list-valued inputs.
            negative_model_inputs: Optional negative inputs for CFG.

        Returns:
            Noise prediction tensor of shape ``(B, C, F, H, W)``.
        """
        true_cfg_scale = model_config.pipeline.get("true_cfg_scale", 1.0)
        do_true_cfg = true_cfg_scale > 1.0

        # Positive forward — LTX-2.3 returns LTXVideoModelOutput.
        output = module(**model_inputs)
        # predictions is a list[Tensor]; take the first (batch-level) prediction.
        if hasattr(output, "predictions") and output.predictions:
            noise_pred = output.predictions[0]
        else:
            # Fallback: some call paths may return a tuple.
            noise_pred = output[0] if isinstance(output, (list, tuple)) else output

        # CFG forward (if enabled).
        if do_true_cfg and negative_model_inputs:
            neg_output = module(**negative_model_inputs)
            if hasattr(neg_output, "predictions") and neg_output.predictions:
                neg_noise_pred = neg_output.predictions[0]
            else:
                neg_noise_pred = (
                    neg_output[0] if isinstance(neg_output, (list, tuple)) else neg_output
                )
            noise_pred = apply_true_cfg(noise_pred, neg_noise_pred, true_cfg_scale)

        return noise_pred

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict,
        negative_model_inputs: Optional[dict],
        scheduler_inputs: Optional[TensorDict | dict],
        step: int,
    ):
        """Run the LTX-2.3 transformer and sample the previous denoising step.

        Used by FlowGRPO which requires log-probabilities for reversed-sampling.

        Args:
            module: The LTX-2.3 transformer module.
            scheduler: SDE scheduler used to sample the previous step.
            model_config: Configuration providing guidance scale, noise level,
                and SDE type.
            model_inputs: Positive-prompt inputs (list-valued) for the forward.
            negative_model_inputs: Negative-prompt inputs for CFG.
            scheduler_inputs: Must contain ``"all_latents"`` and ``"all_timesteps"``.
            step: Current denoising step index.

        Returns:
            tuple: ``(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)``.
        """
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        # Forward pass (handles CFG internally via cls.forward).
        noise_pred = cls.forward(module, model_config, model_inputs, negative_model_inputs)

        # Sample previous step via SDE scheduler.
        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )

        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt
