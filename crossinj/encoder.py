# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The zero-initialized cue encoder: the only learned part of cross-injection in S2M2.

(FoundationStereo takes the cue through zero-padded ``gru04`` input channels
instead; see :mod:`crossinj.adapters.foundation_stereo`.)
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

__all__ = ["CrossInjectionEncoder"]


class CrossInjectionEncoder(nn.Sequential):
    """Encodes the inconsistency cue into the refiner's update-feature space.

    Two convs with the final one zero-initialized, so the module emits exactly
    zero until it is trained. That is what makes cross-injection bolt-on: adding
    it to a pretrained refiner leaves the model's predictions bit-identical, and
    the coupling is learned by fine-tuning.

    It is deliberately a *separate* module added alongside the refiner rather
    than a widening of an existing conv. A widened conv changes the shape of an
    existing checkpoint key, and the usual non-strict loaders either drop it
    (leaving it at random init, which breaks the no-op property) or refuse the
    load outright. As a new module its keys are simply absent from a vanilla
    checkpoint and stay at zero.

    It subclasses ``nn.Sequential`` on purpose: the released S2M2 checkpoint
    carries exactly three tensors for it -- ``0.weight``, ``0.bias``, ``2.weight``.

    Args:
        in_channels: 2 for the masked cue ``[c, m]``, 1 for the cue alone.
        dim: Refiner feature width; the output is added residually to a tensor
            of this width, so it must match exactly.
    """

    def __init__(self, in_channels: int, dim: int) -> None:
        out_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        nn.init.zeros_(out_conv.weight)
        super().__init__(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.GELU(),
            out_conv,
        )

    def forward(self, cue: Tensor) -> Tensor:
        """``(B, in_channels, H, W)`` -> ``(B, dim, H, W)``."""
        for module in self:
            cue = module(cue)
        return cue
