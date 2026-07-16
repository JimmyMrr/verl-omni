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
"""CPU tests for LTX-2.3 FlowGRPO adapter registration and sigma schedule."""

import math

import pytest
import torch

from verl_omni.pipelines.ltx2_3_flow_grpo.common import (
    DEFAULT_FPS,
    LTX2_VAE_SPATIAL_SCALE_FACTOR,
    LTX2_VAE_TEMPORAL_SCALE_FACTOR,
    apply_true_cfg,
    build_video_shapes,
    compute_ltx2_sigmas,
)
from verl_omni.pipelines.ltx2_3_flow_grpo.diffusers_training_adapter import LTX23FlowGRPO
from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_config(
    architecture: str = "LTXVideoTransformerModel",
    algorithm: str = "flow_grpo",
    external_lib=None,
) -> DiffusionModelConfig:
    """Build a minimal DiffusionModelConfig without hitting __post_init__."""
    cfg = object.__new__(DiffusionModelConfig)
    object.__setattr__(cfg, "architecture", architecture)
    object.__setattr__(cfg, "external_lib", external_lib)
    object.__setattr__(cfg, "algorithm", algorithm)
    return cfg


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestLTX23Registration:
    def test_training_adapter_registered(self):
        cfg = _make_model_config()
        cls = DiffusionModelBase.get_class(cfg)
        assert cls is LTX23FlowGRPO

    def test_rollout_pipeline_registered(self):
        # The rollout adapter may be unavailable if vllm_omni is not installed,
        # but the registry entry should exist.
        cls = VllmOmniPipelineBase.get_class("LTXVideoTransformerModel", "flow_grpo")
        assert cls is not None

    def test_adapter_inherits_diffusion_model_base(self):
        from verl_omni.pipelines.model_base import DiffusionModelBase

        assert issubclass(LTX23FlowGRPO, DiffusionModelBase)


# ---------------------------------------------------------------------------
# Sigma schedule tests
# ---------------------------------------------------------------------------


class TestLTX2SigmaSchedule:
    def test_sigmas_shape(self):
        for n in [10, 30, 50]:
            sigmas = compute_ltx2_sigmas(n)
            assert sigmas.shape == (n + 1,)
            assert abs(sigmas[0].item() - 1.0) < 1e-6
            assert abs(sigmas[-1].item()) < 1e-6

    def test_sigmas_monotonic_decreasing(self):
        sigmas = compute_ltx2_sigmas(30)
        non_zero = sigmas[:-1]
        assert torch.all(non_zero[1:] <= non_zero[:-1] + 1e-6)

    def test_sigmas_terminal_stretch(self):
        """The last non-zero sigma should be stretched to ~0.1."""
        sigmas = compute_ltx2_sigmas(30)
        # The second-to-last sigma (last non-zero) should map to terminal value
        last_non_zero = sigmas[-2]
        # After stretching, 1 - sigma_last should equal 1 - 0.1 = 0.9
        # So sigma_last should be 0.1
        assert abs(last_non_zero.item() - 0.1) < 0.05


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestLTX2Utilities:
    def test_build_video_shapes(self):
        shapes = build_video_shapes(
            height=512, width=512, num_frames=41, batch_size=2
        )
        assert len(shapes) == 2
        latent_shape = shapes[0][0]
        # (C, F, H, W)
        assert latent_shape[2] == 512 // LTX2_VAE_SPATIAL_SCALE_FACTOR
        assert latent_shape[3] == 512 // LTX2_VAE_SPATIAL_SCALE_FACTOR
        assert latent_shape[1] == (41 - 1) // LTX2_VAE_TEMPORAL_SCALE_FACTOR + 1

    def test_apply_true_cfg(self):
        noise_pred = torch.randn(2, 4, 5, 8, 8)
        neg_noise_pred = torch.randn(2, 4, 5, 8, 8)
        result = apply_true_cfg(noise_pred, neg_noise_pred, cfg_scale=4.0)
        assert result.shape == noise_pred.shape

    def test_default_fps(self):
        assert DEFAULT_FPS == 24


# ---------------------------------------------------------------------------
# prepare_model_inputs tests (CPU, with mock tensors)
# ---------------------------------------------------------------------------


class TestLTX23PrepareModelInputs:
    def _make_mock_inputs(self, batch_size=2, num_steps=5):
        """Create mock tensors matching verl-omni's micro_batch layout."""
        latents = torch.randn(batch_size, num_steps, 128, 6, 16, 16)
        timesteps = torch.rand(batch_size, num_steps)
        prompt_embeds = torch.randn(batch_size, 32, 4096)
        prompt_embeds_mask = torch.ones(batch_size, 32, dtype=torch.long)
        neg_prompt_embeds = torch.randn(batch_size, 32, 4096)
        neg_prompt_embeds_mask = torch.ones(batch_size, 32, dtype=torch.long)
        return latents, timesteps, prompt_embeds, prompt_embeds_mask, neg_prompt_embeds, neg_prompt_embeds_mask

    def test_prepare_model_inputs_t2v(self):
        from tensordict import TensorDict

        latents, timesteps, pe, pe_mask, neg_pe, neg_pe_mask = self._make_mock_inputs()
        micro_batch = TensorDict({"fps": torch.tensor([24.0, 24.0])}, batch_size=[2])

        model_inputs, neg_inputs = LTX23FlowGRPO.prepare_model_inputs(
            module=None,
            model_config=_make_model_config(),
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=pe,
            prompt_embeds_mask=pe_mask,
            negative_prompt_embeds=neg_pe,
            negative_prompt_embeds_mask=neg_pe_mask,
            micro_batch=micro_batch,
            step=0,
        )

        # Verify list[Tensor] format
        assert isinstance(model_inputs["hidden_states"], list)
        assert model_inputs["hidden_states"][0].shape == (2, 128, 6, 16, 16)
        assert isinstance(model_inputs["timestep"], list)
        assert isinstance(model_inputs["encoder_hidden_states"], list)
        assert model_inputs["fps"] == [24.0, 24.0]
        assert model_inputs["return_dict"] is True

    def test_prepare_model_inputs_i2v_with_ref_seq_len(self):
        from tensordict import TensorDict

        latents, timesteps, pe, pe_mask, neg_pe, neg_pe_mask = self._make_mock_inputs()
        micro_batch = TensorDict(
            {"fps": torch.tensor([24.0, 24.0]), "ref_seq_len": torch.tensor([16, 16])},
            batch_size=[2],
        )

        model_inputs, _ = LTX23FlowGRPO.prepare_model_inputs(
            module=None,
            model_config=_make_model_config(),
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=pe,
            prompt_embeds_mask=pe_mask,
            negative_prompt_embeds=neg_pe,
            negative_prompt_embeds_mask=neg_pe_mask,
            micro_batch=micro_batch,
            step=1,
        )

        assert "ref_seq_len" in model_inputs
        assert model_inputs["ref_seq_len"] == [16, 16]
