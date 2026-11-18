# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Portions of this file are adapted from S2M2
# (https://github.com/junhong-3dv/s2m2, file src/s2m2/core/model/submodules.py
# at commit 8f3bd785), by Junhong Min, Youngpil Jeon, Jimin Kim and Minyong
# Choi, licensed under CC BY-NC 4.0
# (https://creativecommons.org/licenses/by-nc/4.0/):
#
# * ReleaseCostVolume is a modified version of upstream's ``CostVolume`` and
#   _sample_release of its ``bilinear_sampler``. Modified: the sampler keeps
#   the ``align_corners=True`` normalisation but samples with
#   ``align_corners=False`` and samples x only; the coarse volume is passed in
#   (built by build_coarse_cost_volume and upsampled back to full resolution)
#   instead of average-pooled from the full volume; variables were renamed.
# * build_coarse_cost_volume reuses the layer-norm-and-correlate step of
#   upstream's ``DispInit.forward``. Modified: the normalised features are
#   subsampled to half resolution before they are correlated, in fp32.
#
# The module docstring describes what these changes do to the computed function.

"""The correlation lookup the released S2M2 checkpoint was trained with (S2M2 only).

**Read this before trusting any number you get from the released weights.**

What to do: nothing, for the released S2M2-XL + CI checkpoint --
``CrossInjectionS2M2(cost_volume="release")``, the default, uses the
lookup below. To add cross-injection to the *public* S2M2 weights (e.g. to
fine-tune it yourself), build the adapter with ``cost_volume="upstream"``; the
model then predicts exactly as upstream S2M2 does until it is fine-tuned.
``scripts/verify.py`` exposes the same choice as ``--cost-volume`` and
``--baseline-cost-volume``. FoundationStereo is not affected.

The released checkpoint loads into the upstream model parameter for parameter,
but it was trained with a correlation lookup that differs from upstream's
``CostVolume`` in two *parameter-free* ways, so the same weights do not compute
the same function under upstream's lookup:

1. **A position-dependent sampling offset.** The lookup normalises sample
   coordinates with the ``align_corners=True`` formula, ``2x/(W-1) - 1``, and
   then calls ``grid_sample(..., align_corners=False)``. Under
   ``align_corners=False`` a normalised coordinate ``g`` addresses pixel
   ``((g+1)W - 1)/2``, so an intended pixel ``x`` is actually read at
   ``xW/(W-1) - 0.5``: a stretch whose error runs from -0.5 px at the first
   column through 0 at the centre to +0.5 px at the last. Upstream is
   consistent (``align_corners=True`` throughout) and has no such offset.

2. **A different coarse cost volume.** Upstream derives its second, coarser
   correlation by average-pooling the full-resolution volume along the
   disparity axis. The lookup here builds it from *half-resolution features*:
   the normalised features are nearest-neighbour subsampled by 2 and correlated
   again, and the resulting volume is nearest-neighbour upsampled back to full
   spatial resolution. These are not the same tensor.

Neither touches a learned parameter, which is exactly why a strict
``load_state_dict`` succeeds and gives no warning. The failure is silent, and
it is large: on Middlebury 2014 ``Motorcycle`` at half resolution, the released
S2M2-XL + CI checkpoint run under the upstream lookup instead of the release
one goes from EPE (left) 0.434 to 1.03 px and BP-2 (right) 1.77 to 2.60 %,
while left-right consistency collapses -- LR-BP-1 9.52 to 68.3 %. The one
metric the paper is about is the one this breaks most.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["ReleaseCostVolume", "build_coarse_cost_volume"]


def _sample_release(volume: Tensor, coords: Tensor) -> Tensor:
    """Bilinear lookup the released weights were trained with, offset included.

    ``volume`` is ``(N, 1, 1, W2)``; ``coords`` is ``(N, 1, K, 1)`` of pixel
    x-positions. See the module docstring for why the normalisation and the
    ``align_corners`` flag deliberately disagree.
    """
    width = volume.shape[-1]
    xgrid = 2 * coords / (width - 1) - 1
    grid = torch.cat([xgrid, torch.zeros_like(xgrid)], dim=-1).to(volume.dtype)
    return F.grid_sample(
        volume, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def build_coarse_cost_volume(layer_norm: torch.nn.Module, features: Tensor) -> Tensor:
    """Coarse correlation volume, built from half-resolution features.

    Normalise, split into the two views, subsample by 2 (nearest neighbour),
    and correlate -- without scaling the correlation, like upstream. Computed
    in fp32 and returned in the dtype of ``layer_norm``'s parameters, i.e. in
    fp32 unless the model itself was cast to a lower precision.

    Args:
        layer_norm: The upstream ``DispInit.layer_norm`` module, reused so the
            normalisation is bit-identical.
        features: Stacked transformed features ``(2B, C, H, W)``.

    Returns:
        ``(B, H/2, W/2, W/2)`` coarse cost volume.
    """
    weight, bias = layer_norm.weight, layer_norm.bias
    with torch.amp.autocast(
        "cuda" if features.device.type == "cuda" else "cpu", enabled=False
    ):
        normed = F.layer_norm(
            features.float().permute(0, 2, 3, 1), layer_norm.normalized_shape,
            None if weight is None else weight.float(),
            None if bias is None else bias.float(),
            layer_norm.eps,
        )
        feat0, feat1 = normed.chunk(2, dim=0)
        down0 = F.interpolate(feat0.permute(0, 3, 1, 2), scale_factor=0.5)
        down1 = F.interpolate(feat1.permute(0, 3, 1, 2), scale_factor=0.5)
        volume = torch.einsum("...chi,...chj -> ...hij", down0, down1)
    return volume.to(weight.dtype if weight is not None else volume.dtype)


class ReleaseCostVolume:
    """Drop-in replacement for upstream ``CostVolume`` with release semantics.

    Same call signature as upstream's: ``cv_fn(disp) -> (corr, corr_coarse)``,
    each ``(B, 2r+1, H, W)``.

    Args:
        cv: Full-resolution cost volume ``(B, H, W, W)``.
        cv_coarse: Coarse volume from :func:`build_coarse_cost_volume`.
        coords: Pixel x-grid ``(B, H, W, 1)``.
        radius: Correlation search radius (4 for S2M2).
    """

    def __init__(self, cv: Tensor, cv_coarse: Tensor, coords: Tensor,
                 radius: int) -> None:
        self.radius = radius
        offsets = torch.linspace(-radius, radius, 2 * radius + 1,
                                 device=cv.device, dtype=cv.dtype)
        self.dx = offsets.reshape(1, 1, 2 * radius + 1, 1)

        b, h, w, w2 = cv.shape
        self.cv = cv.reshape(b * h * w, 1, 1, w2)
        # Back up to full spatial resolution, keeping the coarse disparity axis.
        self.cv_coarse = (
            F.interpolate(cv_coarse.permute(0, 3, 1, 2), scale_factor=2)
            .permute(0, 2, 3, 1)
            .reshape(b * h * w, 1, 1, w2 // 2)
        )
        self.coords = coords.reshape(b * h * w, 1, 1, 1)

    def __call__(self, disp: Tensor) -> tuple[Tensor, Tensor]:
        b, _, h, w = disp.shape
        flat = disp.reshape(b * h * w, 1, 1, 1)
        shape = (b, h, w, 2 * self.radius + 1)

        x_full = self.coords - flat + self.dx
        corr = _sample_release(self.cv, x_full).reshape(*shape).permute(0, 3, 1, 2)

        x_coarse = self.coords / 2 - flat / 2 + self.dx
        corr_coarse = (
            _sample_release(self.cv_coarse, x_coarse).reshape(*shape).permute(0, 3, 1, 2)
        )
        return corr, corr_coarse
