# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Parameter-free geometry behind the cross-injection cue.

Everything here is a warp, a subtraction, or a mask -- there is nothing to
learn. The two public entry points are :func:`covisibility` (Eq. 3 of the
paper) and :func:`inconsistency` (Eq. 2).

Convention throughout: disparities are non-negative, shape ``(B, 1, H, W)``,
and a left pixel ``(x, y)`` corresponds to the right pixel ``(x - D_L, y)``.
``direction`` is the sign applied to the warp field: ``-1`` samples the right
view in the left frame, ``+1`` samples the left view in the right frame.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["warp_disp", "nonocc_single_view", "covisibility", "inconsistency"]


def warp_disp(
    disp_src: Tensor,
    disp_ref: Tensor,
    direction: float,
    mode: str = "bilinear",
    offset: float = 0.0,
) -> Tensor:
    """Backward-warp ``disp_src`` into ``disp_ref``'s frame using ``disp_ref``.

    This is the sampler S_s of Eq. 1: ``S(x, y) -> S(x + s * delta(x, y), y)``,
    evaluated bilinearly with out-of-grid samples set to zero.

    Args:
        disp_src: Disparity to sample from, ``(B, 1, H, W)``.
        disp_ref: Disparity defining the warp field, ``(B, 1, H, W)``.
        direction: ``+1.0`` for left-to-right, ``-1.0`` for right-to-left.
        mode: ``grid_sample`` interpolation mode.
        offset: Constant added to every sample position, in pixels; for a
            ``disp_src`` whose columns are shifted against ``disp_ref``'s.

    Returns:
        ``(B, 1, H, W)``; out-of-bounds pixels are zero.
    """
    b, _, h, w = disp_ref.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=disp_ref.device, dtype=torch.float32),
        torch.arange(w, device=disp_ref.device, dtype=torch.float32),
        indexing="ij",
    )

    sample_x = grid_x.unsqueeze(0) + direction * disp_ref[:, 0].float()
    if offset:
        sample_x = sample_x + offset
    sample_x = 2.0 * sample_x / max(w - 1, 1) - 1.0
    sample_y = 2.0 * grid_y.unsqueeze(0) / max(h - 1, 1) - 1.0
    sample_y = sample_y.expand(b, -1, -1)

    grid = torch.stack([sample_x, sample_y], dim=-1)
    return F.grid_sample(
        disp_src.float(), grid, mode=mode, padding_mode="zeros", align_corners=True
    )


def nonocc_single_view(
    disp: Tensor,
    mask: Tensor | None = None,
    erosion_iters: int = 0,
) -> Tensor:
    """Non-occlusion from one disparity map via forward-warp uniqueness.

    The standard single-view z-buffer test: forward-warp every pixel to the
    other view and, where several land on the same target column, keep only the
    nearest (largest-disparity) one. Needs no opposite-view ground truth, which
    is why it can run inside the refinement loop.

    Args:
        disp: ``(H, W)`` or ``(B, H, W)``, left convention.
        mask: Same shape; pixels where ``mask == 0`` are ignored.
        erosion_iters: 3x3 erosions applied to the result. Shrinking the
            non-occluded region discards unreliable pixels next to depth
            boundaries (flying pixels, interpolation artifacts).

    Returns:
        ``uint8`` mask, same shape as ``disp``; 1 at non-occluded valid pixels.
    """
    unbatched = disp.ndim == 2
    if unbatched:
        disp = disp.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    b, h, w = disp.shape

    if mask is not None:
        invalid = mask == 0
        d = disp.clone()
        d[invalid] = 0
    else:
        invalid = None
        d = disp

    xx = torch.arange(w, device=disp.device).expand(b, h, w)
    target_x = torch.round(xx - d).to(torch.int64)
    valid = (target_x >= 0) & (target_x < w)
    if invalid is not None:
        valid = valid & ~invalid

    bb = torch.arange(b, device=disp.device).view(b, 1, 1).expand(b, h, w)
    yy = torch.arange(h, device=disp.device).view(1, h, 1).expand(b, h, w)

    bb_v, yy_v, xx_v = bb[valid], yy[valid], xx[valid]
    tx_v, d_v = target_x[valid], d[valid]

    # z-buffer: per target cell keep the largest disparity, then mark every
    # source pixel that achieved it (within half a pixel) as non-occluded.
    max_disp = torch.full(
        (b * h * w,), float("-inf"), device=disp.device, dtype=disp.dtype
    )
    flat_tgt = bb_v * (h * w) + yy_v * w + tx_v
    max_disp.scatter_reduce_(0, flat_tgt, d_v, reduce="amax")

    matches = torch.abs(d_v - max_disp[flat_tgt]) < 0.5
    nonocc = torch.zeros(b * h * w, dtype=torch.bool, device=disp.device)
    flat_src = bb_v[matches] * (h * w) + yy_v[matches] * w + xx_v[matches]
    nonocc.index_fill_(0, flat_src, True)

    result = nonocc.view(b, h, w)
    if mask is not None:
        result = result & (mask > 0)

    if erosion_iters > 0:
        r = result.unsqueeze(1).float()
        for _ in range(erosion_iters):
            r = -F.max_pool2d(-r, kernel_size=3, stride=1, padding=1)
        result = r.squeeze(1) > 0.5

    if unbatched:
        result = result.squeeze(0)
    return result.to(torch.uint8)


def covisibility(
    disp_own: Tensor,
    disp_other: Tensor,
    direction: float = -1.0,
    erosion_iters: int = 1,
) -> Tensor:
    """Per-pixel co-visibility mask m (Eq. 3): in-bounds AND not self-occluded.

    Left convention only (``direction < 0``). Both refinement streams are run in
    left convention -- the right one on a flipped-and-swapped input pair -- so
    the occlusion test never needs a flip. A right-convention caller would have
    to flip ``disp_own`` before the test and unflip after.

    Returns:
        ``(B, 1, H, W)`` float mask in {0, 1}.
    """
    if direction >= 0:
        raise ValueError(
            "cross-injection geometry is left-convention only (direction < 0); "
            "feed the right view flipped-and-swapped instead"
        )
    in_bounds = (warp_disp(torch.ones_like(disp_other), disp_own, direction) > 0.5).float()
    nonocc = nonocc_single_view(disp_own[:, 0], erosion_iters=erosion_iters)
    return in_bounds * nonocc.unsqueeze(1).float()


def inconsistency(
    disp_own: Tensor,
    disp_other: Tensor,
    direction: float = -1.0,
    erosion_iters: int = 1,
    with_mask: bool = True,
) -> Tensor:
    """The cross-injection cue: masked signed left-right inconsistency (Eq. 2).

    ``c = (D_own - warp(D_other)) * m``, zeroed wherever the signal is noise.

    ``disp_other`` should already be detached by the caller -- cross-injection
    deliberately carries no gradient between the two passes.

    Args:
        with_mask: Append ``m`` as a second channel so the refiner can tell
            "not co-visible" apart from "the two views agree", which would
            otherwise both read as zero.

    Returns:
        ``(B, 2, H, W)`` if ``with_mask`` else ``(B, 1, H, W)``.
    """
    warped = warp_disp(disp_other, disp_own, direction)
    mask = covisibility(disp_own, disp_other, direction, erosion_iters)
    cue = (disp_own - warped) * mask
    if with_mask:
        return torch.cat([cue, mask], dim=1)
    return cue
