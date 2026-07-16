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

"""Shared utilities for the LTX-2.3 FlowGRPO pipeline."""

import math

import torch

# LTX-2.3 VAE spatial/temporal down-sampling factors.
# The VAE downsamples by 32x spatially and 8x temporally (with causal stride).
LTX2_VAE_SPATIAL_SCALE_FACTOR = 32
LTX2_VAE_TEMPORAL_SCALE_FACTOR = 8

# Default FPS for LTX-2.3 video generation.
DEFAULT_FPS = 24

# Sigma-shift anchors used by the LTX-2.3 noise schedule.
_BASE_SHIFT_ANCHOR = 1024
_MAX_SHIFT_ANCHOR = 4096
_BASE_SHIFT = 0.95
_MAX_SHIFT = 2.05
# Terminal stretch target (the last non-zero sigma is stretched to this value).
_TERMINAL_SIGMA = 0.1


def coalesce_not_none(value, default):
    """Return *default* when *value* is ``None``, otherwise *value*."""
    return default if value is None else value


def compute_ltx2_sigmas(num_inference_steps: int) -> torch.Tensor:
    """Compute the LTX-2.3 shifted-and-stretched sigma schedule.

    Replicates the sigma computation from ``LTX2Scheduler.set_timesteps`` in
    VeOmni.  The schedule:

    1. Linearly spaces sigmas from 1 → 0 (``num+1`` points).
    2. Applies a logistic shift with ``sigma_shift = _MAX_SHIFT`` (2.05).
    3. Stretches non-zero sigmas so the terminal sigma equals ``_TERMINAL_SIGMA``.

    Args:
        num_inference_steps: Number of denoising steps.

    Returns:
        Tensor of shape ``(num_inference_steps + 1,)`` with the final 0 appended.
    """
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)

    # Logistic shift
    sigma_shift = _MAX_SHIFT  # For the max anchor (4096 tokens), shift = 2.05
    sigmas = torch.where(
        sigmas != 0,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1)),
        0,
    )

    # Stretch: map the last non-zero sigma to _TERMINAL_SIGMA
    non_zero_mask = sigmas != 0
    non_zero_sigmas = sigmas[non_zero_mask]
    one_minus_z = 1.0 - non_zero_sigmas
    scale_factor = one_minus_z[-1] / (1.0 - _TERMINAL_SIGMA)
    stretched = 1.0 - (one_minus_z / scale_factor)
    sigmas[non_zero_mask] = stretched

    return sigmas.to(torch.float32)


def build_video_shapes(
    height: int,
    width: int,
    num_frames: int,
    batch_size: int,
) -> list[list[tuple[int, int, int, int]]]:
    """Compute per-sample latent shapes for LTX-2.3 video generation.

    The latent shape is ``(C, F, H, W)`` where:
    - ``C`` is the VAE latent channels.
    - ``F = (num_frames - 1) // temporal_scale + 1``.
    - ``H = height // spatial_scale``.
    - ``W = width // spatial_scale``.

    Args:
        height: Video height in pixels.
        width: Video width in pixels.
        num_frames: Number of video frames.
        batch_size: Number of samples.

    Returns:
        A list of length *batch_size*, each element a singleton list containing
        the ``(C, F, H, W)`` latent shape tuple.
    """
    latent_channels = 128  # LTX-2.3 VAE latent channels
    latent_frames = (num_frames - 1) // LTX2_VAE_TEMPORAL_SCALE_FACTOR + 1
    latent_height = height // LTX2_VAE_SPATIAL_SCALE_FACTOR
    latent_width = width // LTX2_VAE_SPATIAL_SCALE_FACTOR
    return [[(latent_channels, latent_frames, latent_height, latent_width)]] * batch_size


def apply_true_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    true_cfg_scale: float,
) -> torch.Tensor:
    """Apply True-CFG with norm preservation.

    Mirrors the CFG logic used by other verl-omni pipelines (Qwen-Image, Wan2.2).
    """
    comb_pred = negative_noise_pred + true_cfg_scale * (noise_pred - negative_noise_pred)
    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    return comb_pred * (cond_norm / noise_norm)
