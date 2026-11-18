# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Couple the two passes of an unmodified upstream matcher through hooks.

The right disparity is obtained the standard way -- horizontally flip and swap
the input pair -- so both views are left-convention problems for the same
weights. Stacking the two problems in the batch lets the upstream ``forward``
run both passes in lock-step, untouched. A forward pre-hook on the module that
consumes the current disparity at every refinement step reads the stacked
disparity and computes each half's inconsistency with the other half (Eq. 2);
a second, backbone-specific hook injects that cue into the GRU input.

Two update orders are supported (``ci_update_scheme``):

* ``"jacobi"`` (default, the released checkpoints): both cues of a step are
  computed from the previous step's disparities and the two views are updated
  simultaneously, in one call.
* ``"gauss_seidel"``: the left view is updated first, and the right view's cue
  is recomputed from the just-updated left disparity before the right view is
  updated. Upstream still runs both views in one batch, so the per-step module
  is called a second time on the stacked batch with the refreshed cue, and the
  left half of the first call is combined with the right half of the second.
  The two halves of the batch never interact, so this is exact; it costs one
  extra refinement call per step.

A backbone joins by subclassing the upstream model together with
:class:`CrossInjectionMixin`. The subclass must

1. set ``ci_channels`` (2 for the masked cue, 1 for the bare cue) and
   ``ci_erosion_iters``;
2. call :meth:`CrossInjectionMixin._ci_attach` with the module upstream calls
   once per refinement step and the position of the current disparity in that
   module's positional arguments;
3. inject ``self._pending_cue`` (``None`` while uncoupled) where its GRU reads
   its input features, with a hook of its own;
4. for ``"gauss_seidel"``, implement :meth:`CrossInjectionMixin._ci_updated_disp`,
   which reads the updated disparity off one call of that module;
5. run upstream's ``forward`` through :meth:`CrossInjectionMixin._ci_run`,
   which validates and stacks the inputs, pads them, couples the views for the
   call and returns both disparities. Without it no cue is ever computed, and
   nothing reports an error.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .geometry import inconsistency, nonocc_single_view, warp_disp

__all__ = [
    "UPDATE_SCHEMES",
    "CrossInjectionMixin",
    "check_stereo_pair",
    "check_zero_init_is_noop",
    "cross_view_cue",
    "split_views",
    "stack_views",
]

# Both backbones downsample by up to 32 internally.
STRIDE = 32

UPDATE_SCHEMES = ("jacobi", "gauss_seidel")

# Frames in these directories are skipped when a warning picks its source line.
_INTERNAL_DIRS = (os.path.dirname(os.path.abspath(__file__)) + os.sep,
                  os.path.dirname(os.path.abspath(nn.modules.module.__file__)) + os.sep)


def stack_views(left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
    """Stack the left problem and the flipped-and-swapped right problem.

    Returns ``(reference, target)`` images of batch ``2B``: the first ``B``
    entries are the left view's pair, the last ``B`` the right view's pair
    re-posed as a left-convention problem.
    """
    return torch.cat([left, right.flip(-1)]), torch.cat([right, left.flip(-1)])


def split_views(disp: Tensor) -> tuple[Tensor, Tensor]:
    """Undo :func:`stack_views`; the right map is flipped back to its own frame."""
    disp_l, disp_r = disp.chunk(2, dim=0)
    return disp_l, disp_r.flip(-1)


def cross_view_cue(disp: Tensor, erosion_iters: int = 1,
                   with_mask: bool = True, pad: float = 0.0) -> Tensor:
    """The cue for both stacked views at once.

    Args:
        disp: ``(2B, 1, h, w)``, stacked as by :func:`stack_views`.
        erosion_iters: Erosion applied to the co-visibility mask.
        with_mask: Append the mask as a second cue channel.
        pad: Columns of padding on the right edge of each stacked half, at the
            resolution of ``disp`` (may be fractional); see
            :meth:`CrossInjectionMixin._ci_run`.

    Returns:
        ``(2B, 2|1, h, w)``: each view's inconsistency with its partner, in
        fp32. The partner is detached -- cross-injection is an input cue, never
        a gradient bridge between the two passes.
    """
    disp = disp.float()
    disp_l, disp_r = disp.chunk(2, dim=0)
    # Both halves live in left convention, so unflip the partner into this
    # view's frame before comparing.
    partner = torch.cat([disp_r, disp_l]).detach().flip(-1)
    if pad <= 0:
        return inconsistency(disp, partner, direction=-1.0,
                             erosion_iters=erosion_iters, with_mask=with_mask)

    # Each half is padded on its right edge, i.e. image content occupies
    # columns [0, w - pad). The flip moves the partner's padding to its left
    # edge, so its content starts at column `pad`: a pixel x with disparity d
    # matches partner column x - d + pad, and the padding counts as out of view.
    w = disp.shape[-1]
    cols = torch.arange(w, device=disp.device, dtype=torch.float32)
    content = math.ceil(w - pad)
    warped = warp_disp(partner, disp, direction=-1.0, offset=pad)
    partner_content = (cols >= pad).float().expand_as(partner)
    in_bounds = warp_disp(partner_content, disp, direction=-1.0, offset=pad) > 0.5
    # Occlusion test on the image content only, so padding cannot shadow it.
    nonocc = nonocc_single_view(disp[:, 0, :, :content], erosion_iters=erosion_iters)
    nonocc = F.pad(nonocc.float(), (0, w - content)).unsqueeze(1)
    mask = in_bounds.float() * nonocc * (cols < w - pad).float()
    cue = (disp - warped) * mask
    if with_mask:
        return torch.cat([cue, mask], dim=1)
    return cue


def _copy_containers(value: Any) -> Any:
    """Copy lists and tuples (recursively) but not the tensors in them."""
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_containers(v) for v in value)
    if isinstance(value, dict):
        return {k: _copy_containers(v) for k, v in value.items()}
    return value


def _left_from_first(first: Any, second: Any, batch: int) -> Any:
    """Left half of the stacked batch from ``first``, right half from ``second``.

    Only tensors with the stacked batch size ``batch`` are split; anything else
    is taken from ``first``.
    """
    if isinstance(first, Tensor):
        if first.ndim == 0 or first.shape[0] != batch:
            return first
        return torch.cat([first[:batch // 2], second[batch // 2:]])
    if isinstance(first, (list, tuple)):
        return type(first)(_left_from_first(a, b, batch) for a, b in zip(first, second))
    return first


def check_stereo_pair(left: Tensor, right: Tensor) -> None:
    """Raise a readable error for inputs the adapters cannot take.

    Both images must be ``(B, 3, H, W)`` RGB tensors in [0, 255] and of equal
    shape; inputs that look like [0, 1] images trigger a warning, since they
    run without error but give wrong disparities.
    """
    for name, image in (("left", left), ("right", right)):
        if not isinstance(image, Tensor):
            raise TypeError(
                f"{name} image is a {type(image).__module__}.{type(image).__name__}; "
                "expected a torch.Tensor (B, 3, H, W) -- e.g. "
                "torch.from_numpy(rgb).permute(2, 0, 1)[None].float() for an "
                "(H, W, 3) array, or crossinj.io.load_image"
            )
        if image.ndim == 4 and image.shape[1] != 3 and image.shape[-1] in (1, 3, 4):
            raise ValueError(
                f"{name} image has shape {tuple(image.shape)}; expected "
                "(B, 3, H, W) RGB in [0, 255] -- this looks channels-last: "
                "move the channels first with .permute(0, 3, 1, 2)"
                + ("" if image.shape[-1] == 3 else " and convert it to RGB")
            )
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"{name} image has shape {tuple(image.shape)}; expected "
                "(B, 3, H, W) RGB in [0, 255] -- add a batch dimension with "
                "[None] and convert grayscale/RGBA images to RGB "
                "(crossinj.io.load_image does both)"
            )
        if image.numel() == 0:
            raise ValueError(f"{name} image is empty: shape {tuple(image.shape)}")
    if left.shape != right.shape:
        raise ValueError(
            f"left and right images differ in shape: {tuple(left.shape)} vs "
            f"{tuple(right.shape)}"
        )
    for name, image in (("left", left), ("right", right)):
        if image.is_floating_point() and float(image.max()) <= 1.0:
            warnings.warn(
                f"the {name} image has max <= 1.0; the adapters expect pixel "
                "values in [0, 255] (not [0, 1]) and give wrong disparities "
                "otherwise",
                stacklevel=_caller_stacklevel(),
            )


def _caller_stacklevel() -> int:
    """``stacklevel`` that points a warning at the first frame outside
    crossinj and ``torch.nn``, i.e. at the caller's own code."""
    frame, level = sys._getframe(1), 1
    while frame is not None and frame.f_code.co_filename.startswith(_INTERNAL_DIRS):
        frame, level = frame.f_back, level + 1
    return level


class CrossInjectionMixin:
    """Shared hook plumbing; mixed into a subclass of the upstream model."""

    ci_channels: int
    ci_erosion_iters: int
    ci_update_scheme: str = "jacobi"

    def _ci_attach(self, disp_consumer: nn.Module, disp_arg: int) -> None:
        """Read the stacked disparity whenever ``disp_consumer`` is called.

        Args:
            disp_consumer: The module upstream calls once per refinement step
                with the current disparity.
            disp_arg: Position of the disparity in its positional arguments.
        """
        if self.ci_update_scheme not in UPDATE_SCHEMES:
            raise ValueError(f"ci_update_scheme must be one of {UPDATE_SCHEMES}, "
                             f"got {self.ci_update_scheme!r}")
        self._ci_disp_arg = disp_arg
        self._ci_coupled = False
        self._ci_pad_fraction = 0.0
        self._pending_cue: Tensor | None = None
        self._ci_cue_override: Tensor | None = None
        self._ci_inputs: tuple[tuple, dict] | None = None
        disp_consumer.register_forward_pre_hook(self._ci_read_disp, with_kwargs=True)
        disp_consumer.register_forward_hook(self._ci_gauss_seidel, with_kwargs=True)

    def _ci_cue(self, disp: Tensor) -> Tensor:
        return cross_view_cue(
            disp, self.ci_erosion_iters,
            with_mask=self.ci_channels == 2,
            pad=self._ci_pad_fraction * disp.shape[-1],
        )

    def _ci_read_disp(self, _module: Any, args: tuple, kwargs: dict) -> None:
        if not self._ci_coupled:
            return
        if self._ci_cue_override is not None:  # second Gauss-Seidel call
            self._pending_cue = self._ci_cue_override
            return
        self._pending_cue = self._ci_cue(args[self._ci_disp_arg])
        if self.ci_update_scheme == "gauss_seidel":
            # Upstream may update list arguments in place (FoundationStereo's
            # hidden states), so keep the inputs as they were before this call.
            self._ci_inputs = (_copy_containers(args), _copy_containers(kwargs))

    def _ci_gauss_seidel(self, module: nn.Module, _args: tuple, _kwargs: dict,
                         output: Any) -> Any:
        """Redo the right view's update with a cue from the updated left view."""
        if (not self._ci_coupled or self.ci_update_scheme != "gauss_seidel"
                or self._ci_cue_override is not None):
            return None
        args, kwargs = self._ci_inputs
        disp_l, _ = self._ci_updated_disp(args, output).chunk(2)
        _, disp_r = args[self._ci_disp_arg].chunk(2)
        cue_l, _ = self._pending_cue.chunk(2)
        _, cue_r = self._ci_cue(torch.cat([disp_l, disp_r])).chunk(2)
        self._ci_cue_override = torch.cat([cue_l, cue_r])
        try:
            second = module(*_copy_containers(args), **_copy_containers(kwargs))
        finally:
            self._ci_cue_override = None
            self._ci_inputs = None
        return _left_from_first(output, second, args[self._ci_disp_arg].shape[0])

    def _ci_updated_disp(self, args: tuple, output: Any) -> Tensor:
        """The stacked disparity after one call of the per-step module.

        Needed for ``ci_update_scheme="gauss_seidel"`` only.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support ci_update_scheme='gauss_seidel'")

    def _ci_prepare_state_dict(self, state_dict: dict[str, Tensor],
                               strict: bool) -> dict[str, Tensor]:
        """Adapt a checkpoint before :func:`crossinj.io.load_checkpoint` loads it."""
        return state_dict

    def _ci_load_hint(self, state_dict: dict[str, Tensor],
                      missing: list[str]) -> str | None:
        """A one-line hint for when a checkpoint does not match the model."""
        return None

    @contextmanager
    def _ci_coupling(self, enabled: bool) -> Iterator[None]:
        """Couple the stacked views for one upstream forward call."""
        self._ci_coupled = enabled
        try:
            yield
        finally:
            self._ci_coupled = False
            self._pending_cue = None
            self._ci_cue_override = None
            self._ci_inputs = None

    def _ci_run(self, left: Tensor, right: Tensor, enabled: bool,
                upstream_forward: Callable[[Tensor, Tensor], Tensor]
                ) -> tuple[Tensor, Tensor]:
        """Run ``upstream_forward`` on both views, coupled, at any input size.

        The two problems are stacked first (:func:`stack_views`), so both are
        posed in left convention. Each is then padded to a multiple of 32 on
        its *right* edge -- away from the left edge, where a left-convention
        view's matches run out of the image -- and split evenly at top and
        bottom; the output is cropped back. Inputs that are already multiples
        of 32 pass through unpadded. The inputs are cast to the model's dtype.

        Args:
            left, right: ``(B, 3, H, W)`` RGB in [0, 255].
            enabled: ``False`` runs the two passes uncoupled.
            upstream_forward: ``(reference, target) -> (2B, 1, H', W')``
                disparity, i.e. upstream's ``forward`` on the stacked batch.

        Returns:
            ``(left_disparity, right_disparity)``, each ``(B, 1, H, W)``, the
            right one in right-image convention.
        """
        check_stereo_pair(left, right)
        # Match the model's dtype (e.g. float64 inputs into an fp32 model, or
        # fp32 inputs into a model cast with .to(torch.bfloat16)).
        dtype = next(self.parameters()).dtype
        left, right = left.to(dtype), right.to(dtype)
        height, width = left.shape[-2:]
        pad_w = -width % STRIDE
        pad_h = -height % STRIDE
        top = pad_h // 2

        reference, target = stack_views(left, right)
        if pad_w or pad_h:
            pad = [0, pad_w, top, pad_h - top]
            reference = F.pad(reference, pad, mode="replicate")
            target = F.pad(target, pad, mode="replicate")
        self._ci_pad_fraction = pad_w / (width + pad_w)
        try:
            with self._ci_coupling(enabled):
                disp = upstream_forward(reference, target)
        finally:
            self._ci_pad_fraction = 0.0
        return split_views(disp[..., top:top + height, :width])


@torch.no_grad()
def check_zero_init_is_noop(
    model: nn.Module,
    left: Tensor,
    right: Tensor,
    atol: float = 0.0,
    **kwargs: Any,
) -> tuple[bool, float]:
    """Assert the bolt-on property: with a zero-init encoder, CI changes nothing.

    Runs the model with cross-injection on and off and compares. Immediately
    after attaching an untrained encoder to a pretrained model the two must
    agree exactly; any drift means the encoder is not actually zero-initialised
    or the cue is reaching the network by some other path.

    Args:
        model: A cross-injection adapter.
        left, right: Input pair, as the adapter's ``forward`` expects it.
        atol: Allowed absolute difference.
        **kwargs: Forwarded to the adapter's ``forward`` (e.g. ``iters``).

    Returns:
        ``(is_noop, max_abs_difference)``.
    """
    on_l, on_r = model(left, right, enabled=True, **kwargs)
    off_l, off_r = model(left, right, enabled=False, **kwargs)
    delta = max(
        (on_l - off_l).abs().max().item(),
        (on_r - off_r).abs().max().item(),
    )
    return delta <= atol, delta
