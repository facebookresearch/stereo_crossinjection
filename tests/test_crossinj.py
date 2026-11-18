# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the parts that can break silently.

The geometry tests need only torch. The model tests need the upstream S2M2
submodule (``git submodule update --init``) and are skipped without it; the
checkpoint test also needs ``CROSSINJ_S2M2_CKPT`` pointing at a released +CI
checkpoint. The FoundationStereo tests also need its extra dependencies
(``pip install -e .[foundationstereo]``) and, on first use, network access:
building the model downloads timm's ``edgenext_small`` and fetches DINOv2's
code through ``torch.hub`` (they are skipped if that fails).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from crossinj.encoder import CrossInjectionEncoder
from crossinj.geometry import covisibility, inconsistency, nonocc_single_view, warp_disp
from crossinj.loop import cross_view_cue
from crossinj.upstream import add_to_path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S2M2_CHECKED_OUT = add_to_path("s2m2")
FS_CHECKED_OUT = add_to_path("foundation_stereo")

try:  # needs the third_party/s2m2 submodule
    from s2m2.core.model import s2m2
except ImportError:  # pragma: no cover
    s2m2 = None

requires_s2m2 = pytest.mark.skipif(
    s2m2 is None,
    reason="third_party/s2m2 submodule not checked out: git submodule update --init",
)


def _load_script(name: str):
    """Import ``scripts/<name>.py`` as a module."""
    spec = importlib.util.spec_from_file_location(
        f"_script_{name}", os.path.join(REPO_ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def test_warp_roundtrip_on_constant_disparity() -> None:
    """A constant disparity field warps a constant map onto itself."""
    disp = torch.full((1, 1, 16, 32), 5.0)
    other = torch.full((1, 1, 16, 32), 5.0)
    warped = warp_disp(other, disp, direction=-1.0)
    # Only the left 5 columns fall outside the grid and read as zero.
    assert torch.allclose(warped[..., 5:], other[..., 5:], atol=1e-5)


def test_inconsistency_is_zero_when_views_agree() -> None:
    """Two views agreeing on a constant disparity have zero inconsistency."""
    disp = torch.full((1, 1, 16, 32), 4.0)
    cue = inconsistency(disp, disp, direction=-1.0, erosion_iters=0, with_mask=True)
    assert cue.shape == (1, 2, 16, 32)
    covis = cue[:, 1:2] > 0.5
    assert covis.any(), "expected some co-visible pixels"
    assert cue[:, 0:1][covis].abs().max() < 1e-4


def test_inconsistency_recovers_a_known_offset() -> None:
    """If the partner is off by a constant, the cue reports that constant."""
    own = torch.full((1, 1, 8, 64), 6.0)
    partner = torch.full((1, 1, 8, 64), 4.0)
    cue = inconsistency(own, partner, direction=-1.0, erosion_iters=0, with_mask=True)
    covis = cue[:, 1:2] > 0.5
    assert torch.allclose(cue[:, 0:1][covis], torch.tensor(2.0), atol=1e-4)


def test_cue_is_signed() -> None:
    """Sign follows which view is larger -- this is the paper's key cue property."""
    own = torch.full((1, 1, 8, 64), 4.0)
    partner = torch.full((1, 1, 8, 64), 6.0)
    cue = inconsistency(own, partner, direction=-1.0, erosion_iters=0, with_mask=True)
    covis = cue[:, 1:2] > 0.5
    assert cue[:, 0:1][covis].mean() < 0


def test_covisibility_rejects_right_convention() -> None:
    disp = torch.full((1, 1, 8, 16), 2.0)
    with pytest.raises(ValueError, match="left-convention"):
        covisibility(disp, disp, direction=+1.0)


def test_occlusion_test_flags_a_foreground_step() -> None:
    """A disparity step occludes the background pixels it warps over."""
    disp = torch.zeros(1, 8, 64)
    disp[..., 32:] = 10.0  # foreground slab on the right half
    nonocc = nonocc_single_view(disp, erosion_iters=0)
    # The 10 columns just left of the step are shadowed by the slab.
    assert nonocc[0, :, 22:32].float().mean() < 0.5
    assert nonocc[0, :, :20].float().mean() > 0.9


def test_erosion_only_shrinks() -> None:
    disp = torch.zeros(1, 8, 64)
    disp[..., 32:] = 10.0
    loose = nonocc_single_view(disp, erosion_iters=0)
    tight = nonocc_single_view(disp, erosion_iters=2)
    assert (tight <= loose).all()
    assert tight.sum() < loose.sum()


def test_padded_cue_matches_the_unpadded_one() -> None:
    """Right-edge padding of the stacked views leaves the cue on the image unchanged.

    The flip puts each partner's padding on its left edge; the cue must shift
    its warp to match and treat the padding as out of view.
    """
    torch.manual_seed(0)
    width, pad = 40, 8
    disp_l = 2 + 6 * torch.rand(1, 1, 6, width).cumsum(-1) / width
    disp_r = 2 + 6 * torch.rand(1, 1, 6, width).cumsum(-1) / width
    stacked = torch.cat([disp_l, disp_r.flip(-1)])
    expected = cross_view_cue(stacked, erosion_iters=1)
    padded = cross_view_cue(F.pad(stacked, (0, pad)), erosion_iters=1, pad=float(pad))
    torch.testing.assert_close(padded[..., :width], expected, rtol=0, atol=1e-5)
    assert padded[..., width:].abs().max() == 0


@pytest.mark.parametrize("pad", [2.5, 1.25])
def test_fractionally_padded_cue_is_zero_for_consistent_views(pad) -> None:
    """At 1/4 resolution the padding is usually fractional (e.g. 22 px -> 5.5).

    Two exactly consistent linear disparity fields must give a zero cue on
    every co-visible pixel; the same fields with the pad rounded to a whole
    column give a visible error, so the test notices an off-by-a-fraction warp.
    """
    width, a, b = 32, 3.0, 0.1

    def stacked_cue(true_pad: float, cue_pad: float) -> torch.Tensor:
        x = torch.arange(width, dtype=torch.float32)
        disp_l = a + b * x
        # Right view, flipped into the stacked frame: column j is right pixel
        # x_r = (width - pad) - 1 - j, which matches left pixel (x_r + a) / (1 - b).
        x_r = (width - true_pad) - 1 - x
        disp_r = a + b * (x_r + a) / (1 - b)
        stacked = torch.stack([disp_l, disp_r]).view(2, 1, 1, width).expand(2, 1, 4, width)
        return cross_view_cue(stacked.contiguous(), erosion_iters=0, pad=cue_pad)

    cue = stacked_cue(pad, pad)
    covisible = cue[:, 1] > 0.5
    assert covisible[0].sum() > 50 and covisible[1].sum() > 50
    assert cue[:, 0][covisible].abs().max() < 1e-4
    # Padding columns (from width - pad on) are never co-visible.
    assert not covisible[..., int(np.ceil(width - pad)):].any()
    rounded = stacked_cue(pad, float(round(pad)))
    assert rounded[:, 0][rounded[:, 1] > 0.5].abs().max() > 1e-2


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------

def test_encoder_is_exactly_zero_at_init() -> None:
    """The bolt-on property depends on this being exactly, not nearly, zero."""
    enc = CrossInjectionEncoder(in_channels=2, dim=16)
    out = enc(torch.randn(2, 2, 8, 8))
    assert out.abs().max().item() == 0.0


def test_encoder_checkpoint_keys_match_the_release_layout() -> None:
    """Released checkpoints carry exactly these three tensors."""
    enc = CrossInjectionEncoder(in_channels=2, dim=16)
    assert set(enc.state_dict()) == {"0.weight", "0.bias", "2.weight"}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.rand(1, 3, 96, 128) * 255, torch.rand(1, 3, 96, 128) * 255


@requires_s2m2
def test_adapter_matches_upstream_when_cue_is_zero(small_pair) -> None:
    """Bolting CI onto a pretrained model must not change its predictions.

    Same weights, upstream's own forward per view vs both views stacked through
    the adapter with a zero-init encoder. Runs in ``upstream`` cost-volume mode
    because that is what upstream does.

    The tolerance is floating-point noise, not a model difference: batching
    the two views changes the reduction order of CPU GEMMs, which depends on
    the thread count (up to ~3e-5 px at 20-32 OpenMP threads, below 1e-5 at
    most others) -- the same residual upstream shows between batch sizes.
    """
    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    left, right = small_pair
    ref = s2m2.S2M2(feature_channels=128, dim_expansion=1, num_transformer=1,
                    use_positivity=True).eval()
    ours = CrossInjectionS2M2(model_size="S", refine_iter=3, cost_volume="upstream",
                              use_positivity=True).eval()
    ours.load_state_dict(
        {**ref.state_dict(),
         **{k: v for k, v in ours.state_dict().items() if "ci_encoder" in k}},
        strict=True,
    )
    with torch.inference_mode():
        expected_l, _, _ = ref(left, right)
        expected_r, _, _ = ref(right.flip(-1), left.flip(-1))
        got_l, got_r = ours(left, right, enabled=True)
    torch.testing.assert_close(got_l, expected_l, rtol=0, atol=1e-4)
    torch.testing.assert_close(got_r, expected_r.flip(-1), rtol=0, atol=1e-4)


@requires_s2m2
def test_adapter_pads_each_view_away_from_its_out_of_view_edge() -> None:
    """Any input size works: each view is padded on the side its matches do not leave.

    In its own frame the left view is padded on the right, the right view on
    the left; the result must equal upstream run on exactly those inputs.
    """
    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    torch.manual_seed(0)
    left, right = torch.rand(1, 3, 90, 120) * 255, torch.rand(1, 3, 90, 120) * 255
    ref = s2m2.S2M2(feature_channels=128, dim_expansion=1, num_transformer=1,
                    use_positivity=True).eval()
    ours = CrossInjectionS2M2(model_size="S", cost_volume="upstream").eval()
    ours.load_state_dict({**ref.state_dict(), **{
        k: v for k, v in ours.state_dict().items() if "ci_encoder" in k}})

    pad = (0, 8, 3, 3)  # 120 -> 128 wide, 90 -> 96 high
    with torch.inference_mode():
        got_l, got_r = ours(left, right, enabled=False)
        exp_l, _, _ = ref(F.pad(left, pad, mode="replicate"),
                          F.pad(right, pad, mode="replicate"))
        exp_r, _, _ = ref(F.pad(right.flip(-1), pad, mode="replicate"),
                          F.pad(left.flip(-1), pad, mode="replicate"))
    assert got_l.shape == got_r.shape == (1, 1, 90, 120)
    torch.testing.assert_close(got_l, exp_l[..., 3:93, :120], rtol=0, atol=1e-4)
    torch.testing.assert_close(got_r, exp_r[..., 3:93, :120].flip(-1), rtol=0, atol=1e-4)


@requires_s2m2
def test_adapter_hands_the_cue_a_fractional_pad(monkeypatch) -> None:
    """118 px wide pads to 128; at the refinement resolution that is fractional."""
    import crossinj.loop as loop
    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    seen = []

    def recording_cue(disp, *args, pad=0.0, **kwargs):
        cue = cross_view_cue(disp, *args, pad=pad, **kwargs)
        seen.append((disp.shape[-1], pad, cue))
        return cue

    monkeypatch.setattr(loop, "cross_view_cue", recording_cue)
    torch.manual_seed(0)
    left, right = torch.rand(1, 3, 90, 118) * 255, torch.rand(1, 3, 90, 118) * 255
    model = CrossInjectionS2M2(model_size="S").eval()
    torch.nn.init.normal_(model.refiner.ci_encoder[2].weight, std=0.05)
    with torch.inference_mode():
        model(left, right)
    assert seen
    for width, pad, cue in seen:
        assert pad == pytest.approx(10 / 128 * width)
        assert not cue[..., int(np.ceil(width - pad)):].any()
    assert any(pad != int(pad) for _, pad, _ in seen)


@requires_s2m2
def test_bolt_on_and_hooks_hold_at_any_input_size() -> None:
    """The padded path keeps the no-op property, and the cue still reaches the GRU."""
    from crossinj.adapters.s2m2 import CrossInjectionS2M2
    from crossinj.loop import check_zero_init_is_noop

    torch.manual_seed(0)
    left, right = torch.rand(1, 3, 90, 120) * 255, torch.rand(1, 3, 90, 120) * 255
    model = CrossInjectionS2M2(model_size="S").eval()
    is_noop, delta = check_zero_init_is_noop(model, left, right)
    assert is_noop, f"zero-init encoder changed the output by {delta}"
    torch.nn.init.normal_(model.refiner.ci_encoder[2].weight, std=0.05)
    with torch.inference_mode():
        on, _ = model(left, right, enabled=True)
        off, _ = model(left, right, enabled=False)
    assert (on - off).abs().max() > 1e-6


@requires_s2m2
def test_cross_injection_is_a_noop_at_zero_init(small_pair) -> None:
    """CI on and CI off must agree exactly while the encoder is zero."""
    from crossinj.adapters.s2m2 import CrossInjectionS2M2
    from crossinj.loop import check_zero_init_is_noop

    left, right = small_pair
    model = CrossInjectionS2M2(model_size="S", refine_iter=3).eval()
    is_noop, delta = check_zero_init_is_noop(model, left, right)
    assert is_noop, f"zero-init encoder changed the output by {delta}"


@requires_s2m2
def test_trained_encoder_actually_changes_the_output(small_pair) -> None:
    """Guards against the hooks silently not firing."""
    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    left, right = small_pair
    model = CrossInjectionS2M2(model_size="S", refine_iter=3).eval()
    torch.nn.init.normal_(model.refiner.ci_encoder[2].weight, std=0.05)
    with torch.inference_mode():
        on, _ = model(left, right, enabled=True)
        off, _ = model(left, right, enabled=False)
    assert (on - off).abs().max() > 1e-6


class _ToyStep(torch.nn.Module):
    """Per-step module that, like FoundationStereo's, updates a list in place."""

    def forward(self, state: list, disp: torch.Tensor):
        state[0] = state[0] + 1.0
        cue = self.get_cue()
        return state, disp + 0.5 * cue[:, :1] + 0.1 * state[0]


class _ToyCI(torch.nn.Module):
    def __init__(self, scheme: str, iters: int = 3) -> None:
        from crossinj.loop import CrossInjectionMixin

        class Model(CrossInjectionMixin, torch.nn.Module):
            pass

        super().__init__()
        self.inner = Model()
        self.inner.ci_channels, self.inner.ci_erosion_iters = 2, 0
        self.inner.ci_update_scheme = scheme
        self.inner.dummy = torch.nn.Parameter(torch.zeros(()))
        self.inner.step = _ToyStep()
        self.inner.step.get_cue = lambda: self.inner._pending_cue
        self.inner._ci_attach(self.inner.step, disp_arg=1)
        self.inner._ci_updated_disp = lambda args, output: output[1]
        self.iters = iters

    def upstream(self, reference: torch.Tensor, _target: torch.Tensor) -> torch.Tensor:
        disp, state = reference.mean(1, keepdim=True) * 0.05, [torch.zeros(())]
        for _ in range(self.iters):
            state, disp = self.inner.step(state, disp)
        return disp

    def forward(self, left, right):
        return self.inner._ci_run(left, right, True, self.upstream)


def _toy_reference(left, right, scheme: str, iters: int = 3):
    """The two update orders written out by hand."""
    from crossinj.loop import cross_view_cue, split_views, stack_views

    reference, _ = stack_views(left, right)
    disp, state = reference.mean(1, keepdim=True) * 0.05, torch.zeros(())
    for _ in range(iters):
        state = state + 1.0
        disp_l, disp_r = disp.chunk(2)
        cue_l, cue_r = cross_view_cue(disp, erosion_iters=0).chunk(2)
        new_l = disp_l + 0.5 * cue_l[:, :1] + 0.1 * state
        if scheme == "gauss_seidel":
            _, cue_r = cross_view_cue(torch.cat([new_l, disp_r]), erosion_iters=0).chunk(2)
        new_r = disp_r + 0.5 * cue_r[:, :1] + 0.1 * state
        disp = torch.cat([new_l, new_r])
    return split_views(disp)


def test_update_schemes_match_their_definitions() -> None:
    """Jacobi and Gauss-Seidel through the hooks equal the loops written by hand."""
    torch.manual_seed(0)
    left, right = torch.rand(1, 3, 32, 64) * 255, torch.rand(1, 3, 32, 64) * 255
    out = {}
    for scheme in ("jacobi", "gauss_seidel"):
        with torch.no_grad():
            out[scheme] = _ToyCI(scheme)(left, right)
        expected = _toy_reference(left, right, scheme)
        for got, want in zip(out[scheme], expected):
            assert torch.allclose(got, want, atol=1e-5), scheme
    assert (out["jacobi"][1] - out["gauss_seidel"][1]).abs().max() > 1e-4


def test_unknown_update_scheme_is_rejected() -> None:
    with pytest.raises(ValueError, match="ci_update_scheme"):
        _ToyCI("alternating")


@requires_s2m2
def test_s2m2_gauss_seidel_changes_only_the_right_view_in_one_step(small_pair) -> None:
    """After one step the left update is the same under both orders; the right is not."""
    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    left, right = small_pair
    out = {}
    for scheme in ("jacobi", "gauss_seidel"):
        torch.manual_seed(0)
        model = CrossInjectionS2M2(model_size="S", refine_iter=1,
                                   ci_update_scheme=scheme).eval()
        torch.nn.init.normal_(model.refiner.ci_encoder[2].weight, std=0.05)
        with torch.inference_mode():
            out[scheme] = model(left, right)
    assert torch.equal(out["jacobi"][0], out["gauss_seidel"][0])
    assert (out["jacobi"][1] - out["gauss_seidel"][1]).abs().max() > 1e-6


@requires_s2m2
def test_release_cost_volume_is_scoped_to_the_call(small_pair) -> None:
    """The release lookup is swapped in for one forward and restored after."""
    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    upstream_cost_volume = s2m2.CostVolume
    left, right = small_pair
    model = CrossInjectionS2M2(model_size="S", refine_iter=3).eval()
    with torch.inference_mode():
        release, _ = model(left, right)
        model.cost_volume_mode = "upstream"
        upstream, _ = model(left, right)
    assert s2m2.CostVolume is upstream_cost_volume
    assert model._features is None
    assert (release - upstream).abs().max() > 1e-6


@requires_s2m2
def test_right_view_is_returned_in_right_convention(small_pair) -> None:
    """Both outputs must be full-size and non-negative under use_positivity."""
    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    left, right = small_pair
    model = CrossInjectionS2M2(model_size="S", refine_iter=3).eval()
    with torch.inference_mode():
        disp_l, disp_r = model(left, right)
    assert disp_l.shape == disp_r.shape == (1, 1, 96, 128)
    assert disp_l.min() >= 0 and disp_r.min() >= 0


@requires_s2m2
@pytest.mark.parametrize("shape", [(96, 128), (1, 96, 128), (1, 1, 96, 128),
                                   (1, 4, 96, 128)])
def test_bad_input_shapes_raise_a_readable_error(shape) -> None:
    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    model = CrossInjectionS2M2(model_size="S").eval()
    image = torch.rand(*shape) * 255
    with pytest.raises(ValueError, match=r"expected \(B, 3, H, W\) RGB in \[0, 255\]"):
        model(image, image)


@requires_s2m2
def test_channels_last_and_empty_inputs_raise_a_readable_error() -> None:
    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    model = CrossInjectionS2M2(model_size="S").eval()
    image = torch.rand(1, 64, 96, 3) * 255
    with pytest.raises(ValueError, match=r"channels-last.*permute\(0, 3, 1, 2\)$"):
        model(image, image)
    with pytest.raises(ValueError, match="convert it to RGB"):
        model(image[..., :1], image[..., :1])
    with pytest.raises(ValueError, match="empty"):
        model(torch.zeros(1, 3, 0, 64), torch.zeros(1, 3, 0, 64))


@requires_s2m2
def test_output_upsample_is_rejected() -> None:
    """Upstream's 2x output would be silently cropped to its top-left quarter."""
    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    with pytest.raises(ValueError, match="output_upsample=True is not supported"):
        CrossInjectionS2M2(model_size="S", output_upsample=True)


@requires_s2m2
def test_mismatched_views_and_unit_range_inputs_are_caught() -> None:
    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    model = CrossInjectionS2M2(model_size="S").eval()
    with pytest.raises(ValueError, match="differ in shape"):
        model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 96))
    with pytest.warns(UserWarning, match=r"\[0, 255\]"), torch.inference_mode():
        model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
    with pytest.warns(UserWarning, match="right image"), torch.inference_mode():
        model(torch.rand(1, 3, 64, 64) * 255, torch.rand(1, 3, 64, 64))
    with pytest.raises(TypeError, match="torch.from_numpy"):
        model(np.zeros((64, 64, 3)), np.zeros((64, 64, 3)))


@requires_s2m2
def test_inputs_are_cast_to_the_model_dtype(small_pair) -> None:
    """float64 or integer inputs into an fp32 model run as their fp32 values."""
    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    left, right = small_pair
    model = CrossInjectionS2M2(model_size="S").eval()
    with torch.inference_mode():
        expected, _ = model(left, right)
        from_double, _ = model(left.double(), right.double())
        from_uint8, _ = model(left.to(torch.uint8), right.to(torch.uint8))
        rounded, _ = model(left.to(torch.uint8).float(), right.to(torch.uint8).float())
    assert from_double.dtype == torch.float32
    torch.testing.assert_close(from_double, expected, rtol=0, atol=0)
    torch.testing.assert_close(from_uint8, rounded, rtol=0, atol=0)


@requires_s2m2
@pytest.mark.parametrize("cost_volume", ["release", "upstream"])
def test_bfloat16_cast_model_runs(small_pair, cost_volume) -> None:
    """A model cast with .to(torch.bfloat16) runs, cue and all."""
    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    left, right = small_pair
    model = CrossInjectionS2M2(model_size="S", cost_volume=cost_volume).eval()
    torch.nn.init.normal_(model.refiner.ci_encoder[2].weight, std=0.05)
    model = model.to(torch.bfloat16)
    with torch.inference_mode():
        disp_l, disp_r = model(left.bfloat16(), right.bfloat16())
        from_fp32, _ = model(left, right)  # cast to the model's dtype
    assert disp_l.dtype == torch.bfloat16 and disp_l.shape == (1, 1, 96, 128)
    assert torch.isfinite(disp_l).all() and torch.isfinite(disp_r).all()
    torch.testing.assert_close(from_fp32, disp_l, rtol=0, atol=0)


@pytest.mark.skipif(
    not os.environ.get("CROSSINJ_S2M2_CKPT"),
    reason="set CROSSINJ_S2M2_CKPT to the released S2M2-XL +CI checkpoint",
)
@requires_s2m2
def test_released_checkpoint_loads_strictly() -> None:
    """1220 upstream tensors plus exactly the 3 cross-injection ones."""
    from crossinj.adapters.s2m2 import CrossInjectionS2M2
    from crossinj.io import load_checkpoint

    model = CrossInjectionS2M2(model_size="XL", ci_erosion_iters=2, refine_iter=3)
    assert load_checkpoint(model, os.environ["CROSSINJ_S2M2_CKPT"]) == []
    assert model.refiner.ci_encoder[2].weight.abs().sum() > 0, "encoder still zero"


@requires_s2m2
def test_load_checkpoint_handles_ci_and_vanilla_files(tmp_path) -> None:
    """+CI files load strictly; vanilla ones only with the CI encoder missing."""
    from crossinj.adapters.s2m2 import CrossInjectionS2M2
    from crossinj.io import load_checkpoint

    source = CrossInjectionS2M2(model_size="S", refine_iter=3)
    torch.nn.init.normal_(source.refiner.ci_encoder[2].weight, std=0.05)
    state = source.state_dict()
    torch.save({"state_dict": state}, tmp_path / "ci.pth")
    vanilla = {k: v for k, v in state.items() if "ci_encoder" not in k}
    torch.save({"state_dict": vanilla}, tmp_path / "vanilla.pth")
    torch.save({"state_dict": {**state, "stray.weight": torch.zeros(1)}},
               tmp_path / "stray.pth")

    model = CrossInjectionS2M2(model_size="S", refine_iter=3)
    assert load_checkpoint(model, str(tmp_path / "ci.pth")) == []
    assert torch.equal(model.refiner.ci_encoder[2].weight,
                       source.refiner.ci_encoder[2].weight)

    model = CrossInjectionS2M2(model_size="S", refine_iter=3)
    missing = load_checkpoint(model, str(tmp_path / "vanilla.pth"), strict=False)
    assert missing and all("ci_encoder" in k for k in missing)
    assert model.refiner.ci_encoder[2].weight.abs().sum() == 0

    with pytest.raises(RuntimeError, match="unexpected"):
        load_checkpoint(model, str(tmp_path / "stray.pth"))
    with pytest.raises(RuntimeError, match="missing"):
        load_checkpoint(model, str(tmp_path / "vanilla.pth"))
    partial = {k: v for k, v in vanilla.items() if not k.startswith("refiner.")}
    with pytest.raises(RuntimeError, match="missing"):
        load_checkpoint(model, partial, strict=False)


# The 14 tensors the public S2M2 checkpoints (CH128NTR1.pth ... CH384NTR3.pth)
# carry for modules the upstream model no longer has; upstream skips them too.
UPSTREAM_S2M2_EXTRA_KEYS = [
    "disp_init.layer_norm1.weight", "disp_init.layer_norm1.bias",
    "feature_flow_attn.q_proj.0.weight", "feature_flow_attn.q_proj.0.bias",
    "feature_flow_attn.k_proj.0.weight", "feature_flow_attn.k_proj.0.bias",
    "upsample_mask_4x.conv_x.weight", "upsample_mask_4x.conv_x.bias",
    "upsample_mask_4x.conv_y.weight", "upsample_mask_4x.conv_y.bias",
    "upsample_mask_4x.conv_concat.0.weight", "upsample_mask_4x.conv_concat.0.bias",
    "upsample_mask_4x.conv_concat.2.weight", "upsample_mask_4x.conv_concat.2.bias",
]


@requires_s2m2
def test_public_s2m2_checkpoint_layout_loads_non_strictly(tmp_path) -> None:
    """Public S2M2 files load with strict=False: unused keys skipped, CI at zero."""
    from crossinj.adapters.s2m2 import CrossInjectionS2M2
    from crossinj.io import load_checkpoint

    source = CrossInjectionS2M2(model_size="S")
    state = {k: v for k, v in source.state_dict().items() if "ci_encoder" not in k}
    state.update({k: torch.zeros(4) for k in UPSTREAM_S2M2_EXTRA_KEYS})
    torch.save({"state_dict": state}, tmp_path / "CH128NTR1.pth")

    model = CrossInjectionS2M2(model_size="S", cost_volume="upstream")
    with pytest.warns(UserWarning, match="skipped 14 checkpoint keys"):
        missing = load_checkpoint(model, str(tmp_path / "CH128NTR1.pth"), strict=False)
    assert sorted(missing) == sorted(k for k in model.state_dict() if "ci_encoder" in k)
    assert torch.equal(model.cnn_backbone.norm1.weight, source.cnn_backbone.norm1.weight)
    assert model.refiner.ci_encoder[2].weight.abs().sum() == 0
    # A strict load (e.g. verify.py --checkpoint) points at the non-strict one.
    with pytest.raises(RuntimeError, match=r"unexpected \(14\)") as err:
        load_checkpoint(model, str(tmp_path / "CH128NTR1.pth"))
    assert "public S2M2 checkpoint" in str(err.value)
    assert "--baseline-checkpoint" in str(err.value)


@requires_s2m2
def test_wrong_model_size_names_the_checkpoint_size(tmp_path) -> None:
    """A size mismatch gives a short summary that says which size the file is."""
    from crossinj.adapters.s2m2 import CrossInjectionS2M2
    from crossinj.io import load_checkpoint

    torch.save({"state_dict": CrossInjectionS2M2(model_size="S").state_dict()},
               tmp_path / "s.pth")
    with pytest.raises(RuntimeError, match="model_size='S'") as err:
        load_checkpoint(CrossInjectionS2M2(model_size="M"), str(tmp_path / "s.pth"))
    assert len(str(err.value).splitlines()) <= 6


def test_read_state_dict_accepts_training_checkpoints(tmp_path) -> None:
    """Numpy scalars and omegaconf configs, as in upstream FoundationStereo's file."""
    from crossinj.io import read_state_dict

    blob = {"model": {"w": torch.ones(2)}, "global_step": 3, "epoch": 1,
            "best_val_loss": {"bp2": np.float64(1.5), "epe": np.float32(0.3)},
            "optimizer": {"state": {}, "param_groups": [{"lr": 1e-4}]}}
    try:
        from omegaconf import OmegaConf
        blob["cfg"] = OmegaConf.create({"hidden_dims": [128, 128], "max_disp": 416})
    except ImportError:  # pragma: no cover -- part of the [foundationstereo] extra
        pass
    torch.save(blob, tmp_path / "train.pth")
    assert torch.equal(read_state_dict(tmp_path / "train.pth")["w"], torch.ones(2))


def test_load_checkpoint_accepts_preloaded_backbone(tmp_path) -> None:
    """The FoundationStereo+CI file omits DepthAnythingV2, which the adapter preloads."""
    from crossinj.io import load_checkpoint

    def make(preloaded: bool) -> torch.nn.Module:
        model = torch.nn.Module()
        model.feature = torch.nn.Module()
        model.feature.dino = torch.nn.Module()
        model.feature.dino.depth_anything = torch.nn.Linear(4, 4)
        model.head = torch.nn.Linear(4, 1)
        model.preloaded_prefixes = ("feature.dino.depth_anything.",) if preloaded else ()
        return model

    source = make(preloaded=True)
    head = {k: v for k, v in source.state_dict().items() if k.startswith("head.")}
    torch.save({"model": head}, tmp_path / "fs_ci.pth")

    model = make(preloaded=True)
    assert load_checkpoint(model, str(tmp_path / "fs_ci.pth")) == []
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in head.items())

    with pytest.raises(RuntimeError):
        load_checkpoint(make(preloaded=False), str(tmp_path / "fs_ci.pth"))


def test_download_checkpoint_rejects_unknown_names() -> None:
    from crossinj.hub import download_checkpoint

    with pytest.raises(ValueError, match="s2m2_xl"):
        download_checkpoint("s2m2_l")


# --------------------------------------------------------------------------
# scripts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["s2m2", "foundation_stereo"])
def test_verify_reference_check(arch) -> None:
    """verify.py's reference comparison: match, mismatch, NaN, and the skips."""
    verify = _load_script("verify")
    row = dict(verify.REFERENCE[arch])
    assert verify.compare_to_reference(row, arch) == {}
    assert "matches" in verify.reference_verdict(row, arch, True, True)

    near = {**row, "bp2_l": row["bp2_l"] + 0.01, "epe_r": row["epe_r"] - 0.001}
    assert verify.compare_to_reference(near, arch) == {}
    tampered = {**row, "lr_bp1": row["lr_bp1"] + 0.05, "epe_l": float("nan")}
    assert set(verify.compare_to_reference(tampered, arch)) == {"lr_bp1", "epe_l"}
    verdict = verify.reference_verdict(tampered, arch, True, True)
    assert verdict.startswith("Motorcycle-H: differs") and "lr_bp1" in verdict

    assert "skipped" in verify.reference_verdict(tampered, arch, True, False)
    assert "skipped" in verify.reference_verdict(tampered, arch, False, True)
    assert "--device cpu" in verify.reference_verdict(row, arch, True, True,
                                                      device="cuda")


def test_verify_reference_settings_gate() -> None:
    """The reference check runs only at the settings its values were made with."""
    import argparse

    verify = _load_script("verify")
    s2m2_args = dict(arch="s2m2", scale=1.0, iters=3, erosion_iters=2,
                     model_size="XL", cost_volume="release", max_disp=416,
                     update_scheme="jacobi")
    fs_args = {**s2m2_args, "arch": "foundation_stereo", "iters": 32,
               "erosion_iters": 1, "model_size": "S", "cost_volume": "upstream"}
    for defaults, changes in (
        (s2m2_args, [("scale", 0.5), ("iters", 4), ("erosion_iters", 1),
                     ("model_size", "L"), ("cost_volume", "upstream"),
                     ("update_scheme", "gauss_seidel")]),
        (fs_args, [("scale", 0.5), ("iters", 3), ("erosion_iters", 2),
                   ("max_disp", 192), ("update_scheme", "gauss_seidel")]),
    ):
        assert verify.uses_reference_settings(argparse.Namespace(**defaults))
        for key, value in changes:
            changed = argparse.Namespace(**{**defaults, key: value})
            assert not verify.uses_reference_settings(changed), key
    # FoundationStereo ignores the S2M2-only flags, and vice versa.
    assert verify.uses_reference_settings(
        argparse.Namespace(**{**s2m2_args, "max_disp": 192}))


@requires_s2m2
@pytest.mark.parametrize("ablate", [False, True])
def test_verify_main_runs_ci_on_only_unless_ablating(tmp_path, monkeypatch,
                                                      capsys, ablate) -> None:
    """verify.py end to end on a tiny sample, with a random S2M2-S model."""
    import hashlib

    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    verify = _load_script("verify")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (2, 64, 96, 3), dtype=np.uint8)
    gt = rng.uniform(0, 8, (2, 64, 96)).astype(np.float32)
    np.save(tmp_path / "sample.npy", {"images": images, "gt": {"disparity": gt},
                                      "mask": np.ones((2, 64, 96), bool)})
    state = CrossInjectionS2M2(model_size="S").state_dict()
    torch.save({"state_dict": state}, tmp_path / "ci.pth")
    vanilla = {k: v for k, v in state.items() if "ci_encoder" not in k}
    torch.save({"state_dict": vanilla}, tmp_path / "public.pth")

    argv = ["verify.py", "--checkpoint", str(tmp_path / "ci.pth"),
            "--sample", str(tmp_path / "sample.npy"), "--model-size", "S",
            "--device", "cpu"]
    if ablate:
        argv += ["--ablate-ci", "--baseline-checkpoint", str(tmp_path / "public.pth")]
    monkeypatch.setattr(sys, "argv", argv)
    verify.main()
    out = capsys.readouterr().out
    arms = [line.split("  ")[0] for line in out.splitlines()
            if line.startswith(("+CI ckpt", "public"))]
    if ablate:
        assert arms == ["public (no CI)", "+CI ckpt, CI off", "+CI ckpt, CI on"]
        assert "cross-injection on vs off (same weights): BP-2 L" in out
        assert "ours vs public baseline: BP-2 L" in out
    else:
        assert arms == ["+CI ckpt"]
        assert "on vs off" not in out and "public baseline" not in out
    assert "Reference check skipped: not the Motorcycle_H sample" in out

    # Recognised as the reference sample, the check is still skipped for S.
    monkeypatch.setattr(verify, "MOTORCYCLE_H_IMAGES_SHA256",
                        hashlib.sha256(images.tobytes()).hexdigest())
    verify.main()
    assert ("Reference check skipped: the reference values apply to the "
            "default settings only") in capsys.readouterr().out


@requires_s2m2
def test_demo_build_model_uses_the_released_settings(tmp_path, monkeypatch) -> None:
    """S2M2-XL, erosion 2, 3 iterations; a refused download exits readably."""
    import crossinj.adapters.s2m2 as s2m2_adapter
    import crossinj.hub

    demo = _load_script("demo")
    real = s2m2_adapter.CrossInjectionS2M2
    built = {}

    def small(**kwargs):  # records the settings, builds the cheap S size
        built.update(kwargs)
        return real(**{**kwargs, "model_size": "S"})

    monkeypatch.setattr(s2m2_adapter, "CrossInjectionS2M2", small)
    torch.save({"state_dict": real(model_size="S").state_dict()}, tmp_path / "ci.pth")
    model = demo.build_model("s2m2", str(tmp_path / "ci.pth"))
    assert built == {"model_size": "XL", "ci_erosion_iters": 2, "refine_iter": 3}
    assert not model.training

    def refuse(name):
        raise PermissionError("request access on the model page")

    monkeypatch.setattr(crossinj.hub, "download_checkpoint", refuse)
    with pytest.raises(SystemExit, match="error: request access"):
        demo.build_model("s2m2", None)


@requires_s2m2
def test_demo_writes_both_views(tmp_path, monkeypatch) -> None:
    """demo.py end to end on a tiny pair, with a random S2M2-S model."""
    import cv2

    from crossinj.adapters.s2m2 import CrossInjectionS2M2

    demo = _load_script("demo")
    torch.manual_seed(0)
    monkeypatch.setattr(demo, "build_model",
                        lambda arch, checkpoint, max_disp: CrossInjectionS2M2(
                            model_size="S").eval())
    rng = np.random.default_rng(0)
    for name in ("left", "right"):
        cv2.imwrite(str(tmp_path / f"{name}.png"),
                    rng.integers(0, 256, (70, 100, 3), dtype=np.uint8))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "demo.py", "--left", str(tmp_path / "left.png"),
        "--right", str(tmp_path / "right.png"), "--out", str(out), "--device", "cpu"])
    demo.main()
    for view in ("left", "right"):
        disp = np.load(out / f"disp_{view}.npy")
        assert disp.shape == (70, 100) and disp.dtype == np.float32
        assert cv2.imread(str(out / f"disp_{view}.png")).shape == (70, 100, 3)


# --------------------------------------------------------------------------
# FoundationStereo (skipped unless the upstream repo is on the path)
# --------------------------------------------------------------------------

try:  # optional: needs the submodule plus pip install -e .[foundationstereo]
    from core import foundation_stereo as fs
    _FS_SKIP = ""
except ImportError as _err:  # pragma: no cover
    fs = None
    _FS_SKIP = (
        f"FoundationStereo extras not installed ({_err}): "
        "pip install -e '.[foundationstereo]'"
        if FS_CHECKED_OUT else
        "third_party/FoundationStereo submodule not checked out: "
        "git submodule update --init"
    )

requires_fs = pytest.mark.skipif(fs is None, reason=_FS_SKIP)


def _build_or_skip(build):
    """Construct a FoundationStereo model, skipping if its downloads fail."""
    try:
        return build()
    except OSError as err:  # urllib / huggingface_hub network errors
        pytest.skip("building FoundationStereo downloads timm's edgenext_small "
                    f"and DINOv2's code (torch.hub) on first use: {err}")


@pytest.fixture(scope="module")
def fs_cfg():
    from crossinj.adapters.foundation_stereo import foundation_stereo_cfg

    return foundation_stereo_cfg(max_disp=192)


@requires_fs
def test_demo_build_model_foundation_stereo_settings(monkeypatch) -> None:
    """Erosion 1 and --max-disp reach the FoundationStereo config."""
    import crossinj.adapters.foundation_stereo as fs_adapter
    import crossinj.io

    demo = _load_script("demo")
    built = {}

    def record(cfg, **kwargs):
        built.update(kwargs, max_disp=cfg.max_disp)
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(fs_adapter, "CrossInjectionFoundationStereo", record)
    monkeypatch.setattr(crossinj.io, "load_checkpoint", lambda model, path: [])
    demo.build_model("foundation_stereo", "fs_ci.pth", max_disp=300)
    assert built == {"ci_erosion_iters": 1, "max_disp": 300}


@requires_fs
def test_fs_widens_gru04_to_386(fs_cfg) -> None:
    """The paper states 384 -> 386 for FoundationStereo; hold that exactly."""
    from crossinj.adapters.foundation_stereo import CrossInjectionFoundationStereo

    model = _build_or_skip(lambda: CrossInjectionFoundationStereo(
        fs_cfg, ci_channels=2, load_depth_anything=False))
    assert model.update_block.gru04.conv0[0].weight.shape[0] == 386


@requires_fs
def test_fs_bolt_on_reproduces_vanilla(fs_cfg, small_pair) -> None:
    """Zero-padded vanilla weights in the widened model must predict as before."""
    from crossinj.adapters.foundation_stereo import (
        CrossInjectionFoundationStereo,
        zero_pad_gru04,
    )

    torch.manual_seed(0)
    base = _build_or_skip(lambda: fs.FoundationStereo(fs_cfg).eval())
    model = CrossInjectionFoundationStereo(fs_cfg, ci_channels=2, load_depth_anything=False).eval()
    state = dict(base.state_dict())
    assert zero_pad_gru04(state, model) == 10
    model.load_state_dict(state, strict=True)

    left, right = small_pair
    with torch.inference_mode():
        expected = base(left, right, iters=3, test_mode=True)
        got, _ = model(left, right, iters=3, enabled=True)
    assert (got - expected).abs().max() < 1e-3

    # Not a multiple of 32: each view is padded away from its out-of-view edge
    # and must match upstream run on exactly those padded inputs.
    left, right = left[..., :90, :120], right[..., :90, :120]
    pad = (0, 8, 3, 3)  # 120 -> 128 wide, 90 -> 96 high
    with torch.inference_mode():
        got_l, got_r = model(left, right, iters=3)
        exp_l = base(F.pad(left, pad, mode="replicate"),
                     F.pad(right, pad, mode="replicate"), iters=3, test_mode=True)
        exp_r = base(F.pad(right.flip(-1), pad, mode="replicate"),
                     F.pad(left.flip(-1), pad, mode="replicate"), iters=3, test_mode=True)
    assert got_l.shape == got_r.shape == (1, 1, 90, 120)
    assert (got_l - exp_l[..., 3:93, :120]).abs().max() < 1e-3
    assert (got_r - exp_r[..., 3:93, :120].flip(-1)).abs().max() < 1e-3


@requires_fs
def test_fs_cue_reaches_the_gru(fs_cfg, small_pair) -> None:
    """Guards against the hooks silently not firing for FoundationStereo."""
    from crossinj.adapters.foundation_stereo import CrossInjectionFoundationStereo

    # Freshly built, gru04's cue channels carry random (non-zero) weights.
    torch.manual_seed(0)
    model = _build_or_skip(lambda: CrossInjectionFoundationStereo(
        fs_cfg, ci_channels=2, load_depth_anything=False).eval())
    left, right = small_pair
    with torch.inference_mode():
        on, _ = model(left, right, iters=3, enabled=True)
        off, _ = model(left, right, iters=3, enabled=False)
    assert (on - off).abs().max() > 1e-6

    # Upstream fails on bf16/fp16 weights; the adapter says so up front.
    with pytest.raises(TypeError, match="fp32 only"):
        model.to(torch.bfloat16)(left, right, iters=3)


@requires_fs
def test_fs_gauss_seidel_changes_only_the_right_view_in_one_step(fs_cfg, small_pair) -> None:
    """Also checks that the in-place update of ``net`` does not leak into the redo."""
    from crossinj.adapters.foundation_stereo import CrossInjectionFoundationStereo

    left, right = small_pair
    out = {}
    for scheme in ("jacobi", "gauss_seidel"):
        torch.manual_seed(0)
        model = _build_or_skip(lambda: CrossInjectionFoundationStereo(
            fs_cfg, ci_channels=2, load_depth_anything=False,
            ci_update_scheme=scheme).eval())
        with torch.inference_mode():
            out[scheme] = model(left, right, iters=1)
    assert torch.equal(out["jacobi"][0], out["gauss_seidel"][0])
    assert (out["jacobi"][1] - out["gauss_seidel"][1]).abs().max() > 1e-6


@requires_fs
def test_fs_upstream_training_checkpoint_loads_non_strictly(fs_cfg, small_pair, tmp_path) -> None:
    """Upstream's file layout: a training checkpoint, narrow gru04, bundled backbone."""
    from omegaconf import OmegaConf

    from crossinj.adapters.foundation_stereo import CrossInjectionFoundationStereo
    from crossinj.io import load_checkpoint

    torch.manual_seed(0)
    base = _build_or_skip(lambda: fs.FoundationStereo(fs_cfg).eval())
    torch.save({"model": base.state_dict(), "optimizer": {"state": {}},
                "global_step": 200000, "best_val_loss": {"bp2": np.float64(1.0)},
                "cfg": OmegaConf.create({"max_disp": 192})},
               tmp_path / "model_best_bp2.pth")

    model = CrossInjectionFoundationStereo(fs_cfg, load_depth_anything=False).eval()
    with pytest.raises(RuntimeError, match="strict=False to zero-pad gru04"):
        load_checkpoint(model, str(tmp_path / "model_best_bp2.pth"))
    assert load_checkpoint(model, str(tmp_path / "model_best_bp2.pth"), strict=False) == []

    left, right = small_pair
    with torch.inference_mode():
        expected = base(left, right, iters=3, test_mode=True)
        got, _ = model(left, right, iters=3)
    assert (got - expected).abs().max() < 1e-3

    # The released +CI layout leaves the backbone out: hint at the flag.
    released = {k: v for k, v in model.state_dict().items()
                if not k.startswith("feature.dino.depth_anything.")}
    with pytest.raises(RuntimeError, match="load_depth_anything=True"):
        load_checkpoint(model, released)

    # Upstream's ViT-S checkpoint bundles a narrower backbone (and lacks its
    # deeper blocks): name the size rather than the backbone flag.
    vits = {k: v for k, v in model.state_dict().items() if ".blocks.23." not in k}
    vits["feature.dino.depth_anything.pretrained.cls_token"] = torch.zeros(1, 1, 384)
    with pytest.raises(RuntimeError, match="vit_size='vits'"):
        load_checkpoint(model, vits, strict=False)
