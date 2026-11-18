# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Image and checkpoint loading."""

from __future__ import annotations

import os
import pickle
import warnings
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "colorize_disparity",
    "load_checkpoint",
    "load_image",
    "read_state_dict",
]


def load_image(path: str, device: str = "cpu", scale: float = 1.0) -> Tensor:
    """Read an image as ``(1, 3, H, W)`` float in **[0, 255]**.

    S2M2 normalises internally from this range; do not pre-scale to [0, 1].

    Args:
        scale: Optional resize factor, handy for fitting a large benchmark
            image into limited memory.
    """
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1)[None]
    return tensor.to(device)


def colorize_disparity(disp: Tensor, vmin: float | None = None,
                       vmax: float | None = None) -> "np.ndarray":
    """``(H, W)`` disparity -> ``(H, W, 3)`` uint8 RGB for quick inspection."""
    import cv2

    d = disp.detach().float().cpu().numpy()
    lo = float(np.nanpercentile(d, 2)) if vmin is None else vmin
    hi = float(np.nanpercentile(d, 98)) if vmax is None else vmax
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _training_checkpoint_globals() -> list:
    """Non-tensor types in upstream FoundationStereo's training checkpoint.

    Its config (omegaconf) and bookkeeping (numpy scalars) are rebuilt as
    plain data; allowlisting them lets ``torch.load(weights_only=True)`` read
    the file without unpickling arbitrary code.
    """
    import collections
    import typing

    allowed: list = [typing.Any, collections.defaultdict, dict, int, list,
                     np.dtype, np.core.multiarray.scalar]
    # numpy >= 1.25 pickles dtypes by their per-type class (Float64DType, ...).
    allowed += [cls for cls in vars(getattr(np, "dtypes", object)).values()
                if isinstance(cls, type) and issubclass(cls, np.dtype)]
    try:
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode
    except ImportError:
        return allowed
    return allowed + [AnyNode, ContainerMetadata, DictConfig, ListConfig, Metadata]


def read_state_dict(path: str | os.PathLike) -> dict[str, Tensor]:
    """Read the model weights of a checkpoint in its upstream repository's layout.

    S2M2 checkpoints keep the state dict under ``"state_dict"``,
    FoundationStereo checkpoints under ``"model"`` -- as upstream does for its
    own releases. Upstream FoundationStereo's public checkpoint is a full
    training checkpoint (optimizer state, omegaconf config, numpy scalars);
    those types are allowlisted, so it too loads with ``weights_only=True``
    and only the weights are kept.
    """
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        with torch.serialization.safe_globals(_training_checkpoint_globals()):
            blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, Mapping):
        raise ValueError(f"{path}: expected a dict of tensors, got {type(blob).__name__}")
    for key in ("state_dict", "model"):
        if isinstance(blob.get(key), Mapping):
            return dict(blob[key])
    return dict(blob)


def _summarise(keys: list[str], limit: int = 5) -> str:
    more = f", ... (+{len(keys) - limit} more)" if len(keys) > limit else ""
    return ", ".join(keys[:limit]) + more


def load_checkpoint(model: torch.nn.Module,
                    path: str | os.PathLike | Mapping[str, Tensor],
                    strict: bool = True) -> list[str]:
    """Load a checkpoint into a cross-injection adapter.

    Args:
        model: A cross-injection adapter.
        path: Local ``.pth`` in its upstream repository's layout (see
            :func:`read_state_dict`), or an already loaded state dict.
        strict: Keep this on for the released +CI checkpoints -- a silent
            partial load of a stereo model gives plausible-looking but wrong
            disparities. With ``strict=False`` a vanilla upstream checkpoint
            loads too: the cross-injection weights may be missing and keep
            their zero initialisation, FoundationStereo's narrower ``gru04``
            is zero-padded, and keys the model does not have (the public S2M2
            files carry 14 that upstream itself never uses) are skipped with a
            warning. Any other missing key, and any shape mismatch, is an error
            either way.

    Keys under the model's ``preloaded_prefixes`` (set by the FoundationStereo
    adapter for its DepthAnythingV2 backbone) may be absent from the file.

    Returns:
        The missing keys (empty for a strict load).
    """
    if isinstance(path, Mapping):
        state_dict = dict(path)
    else:
        state_dict = read_state_dict(path)
    prepare = getattr(model, "_ci_prepare_state_dict", None)
    if prepare is not None:
        state_dict = prepare(state_dict, strict)

    target = model.state_dict()
    # Weights the model already loaded itself, e.g. FoundationStereo's
    # DepthAnythingV2 backbone, which the released +CI checkpoint omits.
    preloaded = tuple(getattr(model, "preloaded_prefixes", ()))
    missing = [k for k in target if k not in state_dict and not k.startswith(preloaded)]
    unexpected = [k for k in state_dict if k not in target]
    mismatched = [k for k in state_dict
                  if k in target and state_dict[k].shape != target[k].shape]

    if (mismatched or (strict and (missing or unexpected))
            or any("ci_encoder" not in k for k in missing)):
        lines = ["checkpoint does not match the model."]
        hint = getattr(model, "_ci_load_hint", lambda *_: None)(state_dict, missing)
        if hint:
            lines.append(f"  hint: {hint}")
        if mismatched:
            shapes = [f"{k}: {tuple(state_dict[k].shape)} in checkpoint vs "
                      f"{tuple(target[k].shape)} in model" for k in mismatched[:3]]
            lines.append(f"  shape mismatch ({len(mismatched)}): " + "; ".join(shapes)
                         + (" ..." if len(mismatched) > 3 else ""))
        if missing:
            lines.append(f"  missing ({len(missing)}): {_summarise(missing)}")
        if unexpected:
            lines.append(f"  unexpected ({len(unexpected)}): {_summarise(unexpected)}")
        raise RuntimeError("\n".join(lines))

    if unexpected:
        warnings.warn(
            f"load_checkpoint: skipped {len(unexpected)} checkpoint keys the model "
            f"does not have: {_summarise(unexpected)}",
            stacklevel=2,
        )
        state_dict = {k: v for k, v in state_dict.items() if k in target}
    model.load_state_dict(state_dict, strict=False)
    return missing
