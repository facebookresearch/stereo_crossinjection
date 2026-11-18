# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-injection adapter for FoundationStereo (Wen et al., CVPR 2025).

Imports upstream FoundationStereo from the ``third_party/FoundationStereo`` git
submodule (see :mod:`crossinj.upstream`). No upstream source is copied here:
the adapter calls upstream's ``forward`` and reaches into it through hooks.

FoundationStereo takes the cue by a different route than S2M2. S2M2 adds an
encoded cue residually through a separate zero-init module; here the cue is
**concatenated** into the finest GRU's input, widening ``gru04`` from 384 to 386
channels (the figure quoted in the paper). Both give the same bolt-on property,
by different means: there, a zero-initialised output conv; here,
:func:`zero_pad_gru04`, which slots the pretrained weights into the wider
tensors and leaves zeros on the new channels.

The widening is not a plain end-pad. ``conv0`` sees the raw concatenated input,
so its new channels really are at the end -- but ``conv1`` sees
``cat([x, hidden])``, which puts the new channels *in the middle*, at offset
``input_dim``. Appending there would silently misalign every hidden-state
channel. :func:`zero_pad_gru04` inserts at the right offset. (In the released
checkpoint only ``conv0``'s two new input columns carry trained weights; the
other new slices stayed at zero.)

The released FoundationStereo+CI checkpoint (``foundationstereo_ci.pth``,
weights under ``"model"`` as in upstream's checkpoints) loads with
``load_checkpoint(model, path)``, strictly. The public upstream checkpoint (a
full training checkpoint, ``23-51-11/model_best_bp2.pth``) loads with
``load_checkpoint(model, path, strict=False)``, which reads it without
unpickling arbitrary objects and applies :func:`zero_pad_gru04`; it bundles the
DepthAnythingV2 backbone, so build the adapter with ``load_depth_anything=False``
to skip the separate download.

Unlike upstream's checkpoints, ours omits FoundationStereo's frozen
DepthAnythingV2 ViT-L backbone: fine-tuning left it bit-identical to the public
DepthAnythingV2-Large weights, so the adapter loads those at construction
instead (``load_depth_anything``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..upstream import import_upstream

FoundationStereo = import_upstream(
    "foundation_stereo", "core.foundation_stereo"
).FoundationStereo
SelectiveConvGRU = import_upstream("foundation_stereo", "core.update").SelectiveConvGRU

from ..loop import CrossInjectionMixin  # noqa: E402

__all__ = [
    "DEPTH_ANYTHING_V2_URL",
    "CrossInjectionFoundationStereo",
    "foundation_stereo_cfg",
    "zero_pad_gru04",
]


# Public DepthAnythingV2-Large weights (CC BY-NC 4.0), the frozen ViT-L inside
# FoundationStereo, pinned to a Hub commit (file sha256 a7ea19fa0ed9...9eb9345).
DEPTH_ANYTHING_V2_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/"
    "resolve/cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9/depth_anything_v2_vitl.pth"
)


def foundation_stereo_cfg(**overrides: Any) -> Any:
    """The upstream config for the public ViT-L FoundationStereo architecture.

    Upstream reads these from the ``cfg.yaml`` next to each checkpoint. The
    defaults are the public ViT-L model's architecture, run in fp32
    (``mixed_precision=False``; upstream's ``cfg.yaml`` enables it), and equal
    the architecture keys of the released ``foundationstereo_ci_cfg.yaml``; its two extra
    keys, ``ci_channels`` and ``ci_erosion_iters``, are constructor arguments
    of :class:`CrossInjectionFoundationStereo`. Requires ``omegaconf`` (part of the
    ``[foundationstereo]`` extra).
    """
    from omegaconf import OmegaConf

    cfg = dict(
        hidden_dims=[128, 128, 128], n_gru_layers=3, n_downsample=2,
        max_disp=416, corr_levels=2, corr_radius=4, mixed_precision=False,
        vit_size="vitl", low_memory=0, valid_iters=32, train_iters=22,
    )
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def zero_pad_gru04(state_dict: dict[str, Tensor], model: torch.nn.Module) -> int:
    """Fit vanilla ``gru04`` weights into the CI-widened ones, zeros on the new channels.

    Mutates ``state_dict`` in place. Returns the number of tensors padded.
    """
    target = model.state_dict()
    conv0 = [k for k in state_dict if "gru04" in k and "conv0.0.weight" in k]
    base_input_dim = state_dict[conv0[0]].shape[0] if conv0 else None
    padded = 0

    for key in list(state_dict):
        if "gru04" not in key or key not in target:
            continue
        weight = state_dict[key]
        want = target[key].shape
        if weight.shape == want:
            continue
        deltas = [want[i] - weight.shape[i] for i in range(weight.dim())]
        if all(d <= 0 for d in deltas):
            continue

        if "conv1.0.weight" in key and base_input_dim is not None:
            # conv1's input is cat([x, hidden]): the new channels belong at
            # offset base_input_dim, between the two, not after the hidden state.
            at, n_new = base_input_dim, deltas[1]
            zeros = weight.new_zeros(weight.shape[0], n_new, *weight.shape[2:])
            weight = torch.cat([weight[:, :at], zeros, weight[:, at:]], dim=1)
            if deltas[0] > 0:
                rows = weight.new_zeros(deltas[0], weight.shape[1], *weight.shape[2:])
                weight = torch.cat([weight, rows], dim=0)
        else:
            pad: list[int] = []
            for d in reversed(deltas):
                pad.extend([0, max(d, 0)])
            weight = F.pad(weight, pad)

        state_dict[key] = weight
        padded += 1
    return padded


class CrossInjectionFoundationStereo(CrossInjectionMixin, FoundationStereo):
    """Upstream FoundationStereo with the inconsistency cue fed into ``gru04``.

    Runs in fp32, or under upstream's ``mixed_precision`` autocast. Casting the
    weights themselves to bf16/fp16 fails inside upstream FoundationStereo,
    which casts its features back to fp32 in places, with or without the
    adapter; ``forward`` therefore rejects a cast model with a readable error.

    Args:
        cfg: The upstream config object (``args``).
        ci_channels: 2 for the masked cue, 1 for the bare cue.
        ci_erosion_iters: Erosion on the co-visibility mask (1 for the
            released FoundationStereo checkpoint).
        load_depth_anything: Load the public DepthAnythingV2-Large weights into
            the frozen ViT-L backbone (downloaded once to the torch hub cache).
            The released +CI checkpoint relies on this; turn it off only when a
            checkpoint that bundles the backbone, like upstream's, is loaded next.
        ci_update_scheme: ``"jacobi"`` (default; the released checkpoint) or
            ``"gauss_seidel"``; see :mod:`crossinj.loop`.
    """

    def __init__(self, cfg: Any, ci_channels: int = 2,
                 ci_erosion_iters: int = 1, load_depth_anything: bool = True,
                 ci_update_scheme: str = "jacobi") -> None:
        super().__init__(cfg)
        self.ci_channels = ci_channels
        self.ci_erosion_iters = ci_erosion_iters
        self.ci_update_scheme = ci_update_scheme

        # Keys load_checkpoint() need not find in a checkpoint file.
        self.preloaded_prefixes: tuple[str, ...] = ()
        if load_depth_anything:
            if self.args.vit_size != "vitl":
                raise ValueError("public DepthAnythingV2 weights are wired up for vitl only")
            state = torch.hub.load_state_dict_from_url(
                DEPTH_ANYTHING_V2_URL, map_location="cpu", weights_only=True
            )
            self.feature.dino.depth_anything.load_state_dict(state, strict=True)
            self.preloaded_prefixes = ("feature.dino.depth_anything.",)

        # Rebuild gru04 `ci_channels` wider. Everything downstream of input_dim
        # grows with it, which is why loading needs zero_pad_gru04.
        hidden = self.args.hidden_dims[0]
        base = hidden * (self.args.n_gru_layers > 1) + hidden * 2
        self.update_block.gru04 = SelectiveConvGRU(hidden, base + ci_channels)

        # update_block(net, inp, corr, disp, att) runs once per refinement step.
        self._ci_attach(self.update_block, disp_arg=3)
        self.update_block.gru04.register_forward_pre_hook(self._append_cue)

    def _ci_updated_disp(self, args: tuple, output: Any) -> Tensor:
        """``update_block`` returns ``(net, mask, delta_disp)``; upstream adds the delta."""
        return args[3] + output[2].float()

    def _append_cue(self, _module: Any, args: tuple) -> tuple:
        """Append the cue to ``gru04``'s varargs.

        ``SelectiveConvGRU.forward(att, h, *x)`` concatenates everything after
        ``h``, so one extra positional argument is the whole injection.
        """
        cue = self._pending_cue
        if cue is None:
            # Single-view use of a CI-trained model: feed zeros so the reserved
            # channels stay well defined rather than changing the tensor width.
            reference = args[1]
            cue = reference.new_zeros(
                reference.shape[0], self.ci_channels, *reference.shape[2:]
            )
        return (*args, cue.to(args[1].dtype))

    def _ci_prepare_state_dict(self, state_dict: dict[str, Tensor],
                               strict: bool) -> dict[str, Tensor]:
        """Zero-pad a vanilla checkpoint's narrow ``gru04`` (non-strict loads only)."""
        if not strict:
            state_dict = dict(state_dict)
            zero_pad_gru04(state_dict, self)
        return state_dict

    def _ci_load_hint(self, state_dict: dict[str, Tensor],
                      missing: list[str]) -> str | None:
        """Point at the constructor flag or the loading mode that fixes it."""
        backbone = "feature.dino.depth_anything."
        target = self.state_dict()
        # A smaller ViT (e.g. upstream's 11-33-40) bundles a backbone too, just
        # a narrower one; its missing keys are only the blocks it lacks.
        key = backbone + "pretrained.cls_token"
        width = state_dict[key].shape[-1] if key in state_dict else None
        if width is not None and key in target and width != target[key].shape[-1]:
            size = {384: "vits", 768: "vitb", 1024: "vitl"}.get(width, f"width {width}")
            return (f"checkpoint looks like vit_size={size!r}, but the adapter "
                    f"was built with vit_size={self.args.vit_size!r}; build it "
                    f"with foundation_stereo_cfg(vit_size={size!r}) and "
                    f"load_depth_anything=False (the released model is ViT-L)")
        if missing and all(k.startswith(backbone) for k in missing):
            return ("the checkpoint does not bundle the DepthAnythingV2 backbone; "
                    "construct the adapter with load_depth_anything=True")
        narrow = [k for k, v in state_dict.items() if "gru04" in k and k in target
                  and v.shape != target[k].shape]
        if narrow:
            return ("gru04 is narrower than the cross-injection model's: this "
                    "looks like a vanilla FoundationStereo checkpoint; load it "
                    "with strict=False to zero-pad gru04")
        return None

    def forward(  # type: ignore[override]  # returns both views
        self, image1: Tensor, image2: Tensor, iters: int = 32,
        enabled: bool = True, **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """Predict both disparities jointly.

        Args:
            image1, image2: ``(B, 3, H, W)`` RGB in **[0, 255]**. Any H and W;
                the adapter pads to a multiple of 32 and crops back.
            iters: GRU iterations; FoundationStereo evaluates with 32.
            enabled: ``False`` runs the two passes uncoupled, with zeros on the
                cue channels: the same weights with the cue switched off (an
                inference-time ablation, not the paper's separately fine-tuned
                control).
            **kwargs: Forwarded to upstream ``forward`` (e.g. ``low_memory``).

        Returns:
            ``(left_disparity, right_disparity)``, each ``(B, 1, H, W)`` in
            pixels, the right one in right-image convention.
        """
        dtype = next(self.parameters()).dtype
        if dtype != torch.float32:
            raise TypeError(
                f"the model's weights are {dtype}; FoundationStereo runs in "
                "fp32 only; for reduced precision keep the weights in fp32 "
                "and build it with foundation_stereo_cfg(mixed_precision=True), "
                "which runs upstream's CUDA autocast")

        def upstream_forward(reference: Tensor, target: Tensor) -> Tensor:
            return super(CrossInjectionFoundationStereo, self).forward(
                reference, target, iters=iters, test_mode=True, **kwargs)

        return self._ci_run(image1, image2, enabled, upstream_forward)
