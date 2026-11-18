# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-injection adapter for S2M2 (Min et al., ICCV 2025).

Imports upstream S2M2 from the ``third_party/s2m2`` git submodule (see
:mod:`crossinj.upstream`). :class:`CrossInjectionS2M2` calls upstream's own,
unmodified ``forward`` and reaches into it through forward hooks, plus -- in
the default ``cost_volume="release"`` mode -- one scoped substitution of
upstream's correlation lookup. The only S2M2-derived code in this project
is that lookup, adapted in :mod:`crossinj.compat`.

* **Both views in one call.** The left pair and the flipped-and-swapped right
  pair are stacked in the batch (:func:`crossinj.loop.stack_views`), so the
  upstream forward runs both passes in lock-step.
* **The cue is read and injected with hooks.** A pre-hook on the local refiner
  sees the stacked disparity at every refinement step and computes each view's
  inconsistency with the other. S2M2's ``LocalRefiner.forward`` computes
  ``refine_feat = self.refine_unet(...)[0]`` and immediately feeds it to the
  GRU; a forward hook on ``refine_unet`` adds the encoded cue to that tensor,
  which is exactly the injection point of the paper (Sec. 3.1).
* **Release cost volume.** With ``cost_volume="release"`` the ``CostVolume``
  name upstream's forward looks up is bound to
  :class:`crossinj.compat.ReleaseCostVolume` for the duration of the call.
  The released checkpoint needs it; start from the public S2M2 weights with
  ``cost_volume="upstream"`` (see :mod:`crossinj.compat`). The binding is a
  module-global swap, so do not run another S2M2 in a second thread meanwhile.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from torch import Tensor

from ..upstream import import_upstream

_s2m2 = import_upstream("s2m2", "s2m2.core.model.s2m2")
S2M2 = _s2m2.S2M2

from ..compat import ReleaseCostVolume, build_coarse_cost_volume  # noqa: E402
from ..encoder import CrossInjectionEncoder  # noqa: E402
from ..loop import CrossInjectionMixin  # noqa: E402

__all__ = ["CrossInjectionS2M2", "MODEL_SIZES"]

# Public S2M2 release variants: (feature_channels, num_transformer).
# Checkpoint filenames on HuggingFace encode these as CH<channels>NTR<n>.
MODEL_SIZES: dict[str, tuple[int, int]] = {
    "S": (128, 1),
    "M": (192, 2),
    "L": (256, 3),
    "XL": (384, 3),
}


class CrossInjectionS2M2(CrossInjectionMixin, S2M2):
    """Upstream S2M2 plus a cross-injection encoder on its local refiner.

    Args:
        model_size: One of ``MODEL_SIZES``. Defaults to ``"XL"``, the size of
            the released checkpoint; pass the size of the public S2M2 weights
            you start from otherwise.
        ci_channels: 2 for the masked cue ``[c, m]`` (the paper's setting), 1
            for the bare cue.
        ci_erosion_iters: Erosion applied to the co-visibility mask. The
            released checkpoint uses 2.
        refine_iter: GRU iterations. S2M2 evaluates with 3.
        ci_update_scheme: ``"jacobi"`` (default; the released checkpoint) or
            ``"gauss_seidel"``; see :mod:`crossinj.loop`.
        cost_volume: ``"release"`` (default) is the correlation lookup
            the released checkpoint was trained with and is required
            for it to give the paper's numbers. ``"upstream"`` uses the public
            repo's own lookup; use it to add cross-injection to the public S2M2
            weights, which then predict exactly as upstream does until
            fine-tuned. Nothing checks the pairing at load time: the
            released checkpoint under ``"upstream"`` loses most of its
            left-right consistency, and public weights under ``"release"``
            compute a different function from upstream's. See
            :mod:`crossinj.compat`.
        **kwargs: Forwarded to upstream ``S2M2``, except
            ``output_upsample=True``, which the adapter does not support (it
            crops the output back to the input size); upsample the returned
            disparities yourself instead.
    """

    def __init__(
        self,
        model_size: str = "XL",
        ci_channels: int = 2,
        ci_erosion_iters: int = 2,
        refine_iter: int = 3,
        cost_volume: str = "release",
        ci_update_scheme: str = "jacobi",
        **kwargs: Any,
    ) -> None:
        if cost_volume not in ("release", "upstream"):
            raise ValueError(f"cost_volume must be 'release' or 'upstream', "
                             f"got {cost_volume!r}")
        if model_size not in MODEL_SIZES:
            raise ValueError(
                f"unknown model_size {model_size!r}; expected one of "
                f"{sorted(MODEL_SIZES)}"
            )
        if kwargs.get("output_upsample"):
            raise ValueError(
                "output_upsample=True is not supported: the adapter crops the "
                "output back to the input size; upsample the returned "
                "disparities (and scale them by the same factor) instead")
        channels, num_transformer = MODEL_SIZES[model_size]
        kwargs.setdefault("dim_expansion", 1)
        kwargs.setdefault("use_positivity", True)
        super().__init__(
            feature_channels=channels,
            num_transformer=num_transformer,
            refine_iter=refine_iter,
            **kwargs,
        )
        self.model_size = model_size
        self.ci_channels = ci_channels
        self.ci_erosion_iters = ci_erosion_iters
        self.ci_update_scheme = ci_update_scheme
        self.cost_volume_mode = cost_volume

        # Lives under the refiner so its checkpoint keys read
        # `refiner.ci_encoder.{0.weight,0.bias,2.weight}` -- the exact three
        # tensors the released checkpoint adds on top of a vanilla S2M2.
        self.refiner.ci_encoder = CrossInjectionEncoder(ci_channels, channels)

        self._ci_attach(self.refiner, disp_arg=2)
        self.refiner.refine_unet.register_forward_hook(self._inject)
        # Release mode builds its coarse volume from the transformed features.
        self._features: Tensor | None = None
        self.transformer.register_forward_hook(self._keep_features)

    def _inject(self, _module: Any, _args: Any, output: Any) -> Any:
        """Add the encoded cue to the GRU's input features.

        ``refine_unet`` returns a sequence whose element 0 is the tensor that
        goes into the GRU, so we rebuild that sequence with element 0 shifted.
        """
        if self._pending_cue is None:
            return output
        # The cue is computed in fp32; match the encoder's dtype if the model
        # was cast (a no-op in fp32 and under autocast).
        cue = self._pending_cue.to(self.refiner.ci_encoder[0].weight.dtype)
        encoded = self.refiner.ci_encoder(cue)
        shifted = output[0] + encoded.to(output[0].dtype)
        return (shifted, *output[1:])

    def _ci_updated_disp(self, args: tuple, output: Any) -> Tensor:
        """``refiner`` returns ``(hidden, disp, conf, occ)``; upstream then clamps."""
        disp = output[1]
        return disp.clamp(min=0) if self.use_positivity else disp

    def _keep_features(self, _module: Any, _args: Any, output: Tensor) -> None:
        if self.cost_volume_mode == "release":
            self._features = output

    def _make_release_cost_volume(self, cv: Tensor, coords: Tensor,
                                  radius: int) -> ReleaseCostVolume:
        coarse = build_coarse_cost_volume(self.disp_init.layer_norm, self._features)
        return ReleaseCostVolume(cv, coarse, coords, radius=radius)

    def _ci_load_hint(self, state_dict: dict[str, Tensor],
                      missing: list[str]) -> str | None:
        """Name the model size a mismatching checkpoint was saved with, or
        point a public S2M2 checkpoint at the non-strict load."""
        norm = state_dict.get("cnn_backbone.norm1.weight")
        if norm is not None:
            sizes = {channels: name for name, (channels, _) in MODEL_SIZES.items()}
            size = sizes.get(norm.shape[0])
            if size is not None and size != self.model_size:
                return (f"checkpoint looks like model_size={size!r} (feature_"
                        f"channels={norm.shape[0]}), but the model was built "
                        f"with model_size={self.model_size!r}")
        if missing and all("ci_encoder" in k for k in missing):
            return ("this looks like a public S2M2 checkpoint without "
                    "cross-injection weights; load it with load_checkpoint(..., "
                    "strict=False) into a model built with cost_volume="
                    "'upstream' (in verify.py: --baseline-checkpoint with "
                    "--baseline-cost-volume upstream)")
        return None

    @contextmanager
    def _cost_volume(self) -> Iterator[None]:
        """Bind upstream's ``CostVolume`` name to the release lookup, if asked."""
        if self.cost_volume_mode != "release":
            yield
            return
        upstream_cost_volume = _s2m2.CostVolume
        _s2m2.CostVolume = self._make_release_cost_volume
        try:
            yield
        finally:
            _s2m2.CostVolume = upstream_cost_volume
            self._features = None

    def forward(  # type: ignore[override]  # returns both views
        self, img0: Tensor, img1: Tensor, enabled: bool = True
    ) -> tuple[Tensor, Tensor]:
        """Predict both disparities jointly.

        Args:
            img0, img1: Left and right images, ``(B, 3, H, W)`` RGB in
                **[0, 255]** -- upstream S2M2 normalises internally. Any H and
                W; the adapter pads to a multiple of 32 and crops back.
            enabled: ``False`` runs the two passes uncoupled: the same weights
                with the cue switched off (an inference-time ablation, not the
                paper's separately fine-tuned control).

        Returns:
            ``(left_disparity, right_disparity)``, each ``(B, 1, H, W)`` in
            pixels, the right one in right-image convention.
        """
        def upstream_forward(reference: Tensor, target: Tensor) -> Tensor:
            with self._cost_volume():
                disp, _occ, _conf = super(CrossInjectionS2M2, self).forward(
                    reference, target)
            return disp

        return self._ci_run(img0, img1, enabled, upstream_forward)
