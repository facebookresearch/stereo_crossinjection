# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end check: run a released +CI checkpoint and score it against GT.

By default it runs the +CI checkpoint once, with cross-injection on, and
prints its accuracy and binocular consistency. On the Middlebury-H
``Motorcycle`` sample built by scripts/make_middlebury_sample.py, with the
default settings and --device cpu, it also compares the result with reference
values for the released checkpoints (CPU, fp32) and says whether your install
reproduces them; otherwise (including on a GPU) it says that the check was
skipped.

    python scripts/verify.py --checkpoint s2m2_xl_ci.pth --sample Motorcycle_H.npy

    python scripts/verify.py --arch foundation_stereo \\
        --checkpoint foundationstereo_ci.pth --sample Motorcycle_H.npy

Two optional extra arms, both through identical code:

* --baseline-checkpoint: a public upstream checkpoint (the base model of the
  paper's Table 2), scored through the same code. Public S2M2 files (e.g.
  CH384NTR3.pth) and upstream FoundationStereo's training checkpoint
  (23-51-11/model_best_bp2.pth) both load, with the cross-injection weights
  left at zero. An S2M2 baseline runs with the release correlation lookup by
  default, as every S2M2 row in the paper does; --baseline-cost-volume
  upstream runs it exactly as upstream S2M2 does instead.
* --ablate-ci: the same +CI weights with cross-injection switched off, an
  inference-time ablation. It is not the paper's control, which is a
  separate fine-tune without cross-injection and is not released.

scripts/make_middlebury_sample.py builds such a sample from public data. A
sample is a .npy holding a dict: "images" (2, H, W, 3) uint8 RGB (left,
right), gt["disparity"] (2, H, W) float32 per-view ground truth (NaN where
unknown), and "mask" (2, H, W) bool.

Runs on CPU or GPU, at any image size. On CPU at Middlebury-H (1000x1482, half
resolution), expect about 1.5 min per forward pass and 14 GB of RAM for
S2M2-XL, 3-4 min and 24 GB for FoundationStereo (24 threads); each arm is one
pass, and --scale 0.5 is about 4x cheaper (and skips the reference check).
"""

from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np
import torch

from crossinj.geometry import covisibility, warp_disp
from crossinj.io import load_checkpoint, read_state_dict

# sha256 of the "images" array of the Middlebury-H Motorcycle sample that
# scripts/make_middlebury_sample.py builds from Motorcycle-perfect.zip.
MOTORCYCLE_H_IMAGES_SHA256 = (
    "3b624b848a79da10f02f2502f5898c6ff8e10550d053ee03e9fc41646105dc1e")

# The released checkpoints on that sample, CI on, default settings, CPU fp32.
REFERENCE = {
    "s2m2": {"bp2_l": 1.599, "bp2_r": 1.769, "epe_l": 0.434, "epe_r": 0.458,
             "lr_bp1": 9.522, "lr_bp1_cv": 0.789},
    "foundation_stereo": {"bp2_l": 2.149, "bp2_r": 2.198, "epe_l": 0.509,
                          "epe_r": 0.513, "lr_bp1": 10.062, "lr_bp1_cv": 0.880},
}
# Absolute tolerances: percentage points for the rates, px for EPE.
REFERENCE_TOL = {"bp2_l": 0.02, "bp2_r": 0.02, "epe_l": 0.002, "epe_r": 0.002,
                 "lr_bp1": 0.02, "lr_bp1_cv": 0.02}


def bad_n(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, n: float) -> float:
    """Percentage of valid pixels whose disparity error exceeds ``n`` px."""
    return ((pred - gt).abs()[mask] > n).float().mean().item() * 100.0


def epe(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    return (pred - gt).abs()[mask].mean().item()


def lr_bad1(disp_l: torch.Tensor, disp_r: torch.Tensor,
            covisible_only: bool = False) -> float:
    """LR-BP-1, the paper's binocular-consistency metric. Needs no ground truth.

    Percent of left-view pixels whose match ``x - D_L`` falls inside the right
    image and where the left disparity and the right disparity sampled at that
    match disagree by more than 1 px (Sec. 4 of the paper). Occluded pixels are
    *not* excluded, so they add a floor near the occluded fraction of the image.

    ``covisible_only`` additionally drops occluded pixels (the single-view
    z-buffer test, with one erosion). That stricter variant is much lower and
    is not the number the paper reports.
    """
    warped = warp_disp(disp_r, disp_l, direction=-1.0)
    xs = torch.arange(disp_l.shape[-1], device=disp_l.device, dtype=disp_l.dtype)
    match_x = xs - disp_l
    valid = (match_x >= 0) & (match_x < disp_l.shape[-1])
    if covisible_only:
        valid = covisibility(disp_l, disp_r, direction=-1.0, erosion_iters=1) > 0.5
    return ((disp_l - warped).abs()[valid] > 1.0).float().mean().item() * 100.0


def compare_to_reference(row: dict, arch: str) -> dict[str, tuple[float, float]]:
    """Metrics of ``row`` outside ``REFERENCE_TOL`` of ``REFERENCE[arch]``.

    Returns:
        ``{metric: (measured, reference)}``; empty if the run matches.
    """
    ref = REFERENCE[arch]
    # `not <=` so that a NaN counts as a mismatch.
    return {k: (row[k], ref[k]) for k in ref
            if not abs(row[k] - ref[k]) <= REFERENCE_TOL[k]}


def reference_verdict(row: dict, arch: str, is_reference_sample: bool,
                      default_settings: bool, device: str = "cpu") -> str:
    """The line ``main`` prints about the reference check (run or skipped)."""
    if not is_reference_sample:
        return ("Reference check skipped: not the Motorcycle_H sample built by "
                "scripts/make_middlebury_sample.py (its images hash differs).")
    if not default_settings:
        return ("Reference check skipped: the reference values apply to the "
                "default settings only.")
    if not str(device).startswith("cpu"):
        # GPU kernels (e.g. TF32 convolutions) shift results slightly, which
        # the tolerances are too tight to absorb.
        return ("Reference check skipped: the reference values were computed "
                "on CPU; rerun with --device cpu for the check.")
    off_by = compare_to_reference(row, arch)
    if not off_by:
        return ("Motorcycle-H: matches the reference for the released checkpoint "
                "(CI on, CPU fp32).")
    return ("Motorcycle-H: differs from the reference for the released "
            "checkpoint (CI on, CPU fp32): "
            + ", ".join(f"{k} {got:.3f} vs {ref:.3f}"
                        for k, (got, ref) in off_by.items()))


def uses_reference_settings(args: argparse.Namespace) -> bool:
    """Whether ``args`` are the settings the reference values were computed with.

    ``args.iters`` and ``args.erosion_iters`` must already be resolved (not
    ``None``).
    """
    if args.update_scheme != "jacobi":
        return False
    if args.arch == "foundation_stereo":
        return (args.scale == 1.0 and args.iters == 32 and args.erosion_iters == 1
                and args.max_disp == 416)
    return (args.scale == 1.0 and args.iters == 3 and args.erosion_iters == 2
            and args.model_size == "XL" and args.cost_volume == "release")


def _photometric_residual(ref: np.ndarray, other: np.ndarray,
                          disp: np.ndarray, sign: int) -> float:
    """Mean |ref - other(x + sign*disp)|, used to identify view conventions."""
    h, w = disp.shape
    xs = np.clip(np.arange(w)[None, :] + sign * disp, 0, w - 1)
    x0 = np.floor(xs).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    frac = (xs - x0)[..., None]
    ys = np.arange(h)[:, None].repeat(w, 1)
    warped = other[ys, x0] * (1 - frac) + other[ys, x1] * frac
    return float(np.abs(warped - ref).mean())


def pair_gt_to_views(images: np.ndarray, gt: np.ndarray) -> tuple[int, int]:
    """Work out which GT plane belongs to which image, by photometric test.

    Sample files built by other tools need not store the two GT planes in the
    same order as the two images. Guessing here silently produces a
    plausible-looking but ~40%-BP-2 result, so the pairing is measured rather
    than assumed.

    Returns:
        ``(index of GT for images[0], index of GT for images[1])``.
    """
    left, right = images[0].astype(np.float32), images[1].astype(np.float32)
    as_left = [
        _photometric_residual(left, right, np.nan_to_num(gt[i]), -1) for i in (0, 1)
    ]
    idx_for_left = int(np.argmin(as_left))
    return idx_for_left, 1 - idx_for_left


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="released +CI checkpoint, loaded strictly")
    ap.add_argument("--baseline-checkpoint",
                    help="public upstream checkpoint to score through the same code")
    ap.add_argument("--ablate-ci", action="store_true",
                    help="also run --checkpoint with cross-injection switched off "
                         "(an inference-time ablation of the same weights, not the "
                         "paper's control, which is a separate fine-tune)")
    ap.add_argument("--sample", required=True, help="npy with images/gt/mask")
    ap.add_argument("--arch", default="s2m2", choices=["s2m2", "foundation_stereo"],
                    help="backbone (default: s2m2)")
    ap.add_argument("--model-size", default="XL", choices=["S", "M", "L", "XL"],
                    help="S2M2 size (default: XL, the released checkpoint)")
    ap.add_argument("--cost-volume", default="release", choices=["release", "upstream"],
                    help="S2M2 correlation lookup for --checkpoint (default: release, "
                         "which the released checkpoint needs); see crossinj/compat.py")
    ap.add_argument("--baseline-cost-volume", choices=["release", "upstream"],
                    help="S2M2 correlation lookup for --baseline-checkpoint (default: "
                         "same as --cost-volume, as the paper scored every S2M2 row, "
                         "the public baseline included). Pass 'upstream' to score the "
                         "public weights exactly as upstream S2M2 runs them")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="resize factor for images and GT, e.g. 0.5 to save memory")
    ap.add_argument("--iters", type=int, default=None,
                    help="GRU iterations; default 3 for S2M2, 32 for FoundationStereo")
    ap.add_argument("--erosion-iters", type=int, default=None,
                    help="default 2 for S2M2, 1 for FoundationStereo")
    ap.add_argument("--max-disp", type=int, default=416, help="FoundationStereo only")
    ap.add_argument("--update-scheme", default="jacobi", choices=["jacobi", "gauss_seidel"],
                    help="cross-injection update order (default: jacobi, as the "
                         "released checkpoints were trained); see crossinj/loop.py")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="default: cuda if available, else cpu (the reference "
                         "check runs on cpu only)")
    args = ap.parse_args()

    is_fs = args.arch == "foundation_stereo"
    default_iters, default_erosion = (32, 1) if is_fs else (3, 2)
    if args.iters is None:
        args.iters = default_iters
    if args.erosion_iters is None:
        args.erosion_iters = default_erosion
    if args.baseline_cost_volume is None:
        args.baseline_cost_volume = args.cost_volume
    # The reference values only apply to the default settings.
    default_settings = uses_reference_settings(args)

    import cv2

    blob = np.load(args.sample, allow_pickle=True).item()
    images, gt, valid = blob["images"], blob["gt"]["disparity"], blob["mask"]
    is_reference_sample = hashlib.sha256(
        np.ascontiguousarray(images).tobytes()).hexdigest() == MOTORCYCLE_H_IMAGES_SHA256

    gt_l_idx, gt_r_idx = pair_gt_to_views(images, gt)
    print(f"GT pairing (measured photometrically): images[0]<-gt[{gt_l_idx}], "
          f"images[1]<-gt[{gt_r_idx}]")

    def prep_img(img: np.ndarray) -> torch.Tensor:
        if args.scale != 1.0:
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                             interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1)[None]

    def prep_gt(idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        d, m = gt[idx], valid[idx]
        if args.scale != 1.0:
            d = cv2.resize(d, None, fx=args.scale, fy=args.scale,
                           interpolation=cv2.INTER_NEAREST) * args.scale
            m = cv2.resize(m.astype(np.uint8), None, fx=args.scale, fy=args.scale,
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        return torch.from_numpy(np.nan_to_num(d)), torch.from_numpy(m & np.isfinite(d))

    left = prep_img(images[0]).to(args.device)
    right = prep_img(images[1]).to(args.device)
    gt_l, mask_l = prep_gt(gt_l_idx)
    gt_r, mask_r = prep_gt(gt_r_idx)

    print(f"input {tuple(left.shape[-2:])}  scale={args.scale}  device={args.device}")
    print(f"GT disparity range: {gt_l[mask_l].min():.1f} .. {gt_l[mask_l].max():.1f} px")

    def build(path: str, baseline: bool) -> torch.nn.Module:
        # The +CI checkpoint loads strictly. A public baseline has no
        # cross-injection weights, which then keep their zero initialisation
        # -- the "bolt-on, no-op at init" state.
        state_dict = read_state_dict(path)
        if is_fs:
            from crossinj.adapters.foundation_stereo import (
                CrossInjectionFoundationStereo,
                foundation_stereo_cfg,
            )

            cfg = foundation_stereo_cfg(max_disp=args.max_disp,
                                        valid_iters=args.iters)
            # Upstream's checkpoint bundles the DepthAnythingV2 backbone; ours
            # leaves it out and the adapter loads the public weights instead.
            bundled = any(k.startswith("feature.dino.depth_anything.") for k in state_dict)
            model = CrossInjectionFoundationStereo(
                cfg, ci_channels=2, ci_erosion_iters=args.erosion_iters,
                load_depth_anything=not bundled, ci_update_scheme=args.update_scheme,
            )
        else:
            from crossinj.adapters.s2m2 import CrossInjectionS2M2

            model = CrossInjectionS2M2(
                model_size=args.model_size, ci_channels=2,
                ci_erosion_iters=args.erosion_iters, refine_iter=args.iters,
                cost_volume=args.baseline_cost_volume if baseline else args.cost_volume,
                ci_update_scheme=args.update_scheme,
            )
        load_checkpoint(model, state_dict, strict=not baseline)
        return model.to(args.device).eval()

    def run(model: torch.nn.Module, enabled: bool, label: str) -> dict:
        start = time.time()
        kwargs = {"iters": args.iters} if is_fs else {}
        with torch.inference_mode():
            # Any input size: the adapter pads internally and crops back.
            disp_l, disp_r = model(left, right, enabled=enabled, **kwargs)
        disp_l = disp_l[0, 0].float().cpu()
        disp_r = disp_r[0, 0].float().cpu()
        return {
            "arm": label,
            "bp2_l": bad_n(disp_l, gt_l, mask_l, 2.0),
            "bp2_r": bad_n(disp_r, gt_r, mask_r, 2.0),
            "epe_l": epe(disp_l, gt_l, mask_l),
            "epe_r": epe(disp_r, gt_r, mask_r),
            "lr_bp1": lr_bad1(disp_l[None, None], disp_r[None, None]),
            "lr_bp1_cv": lr_bad1(disp_l[None, None], disp_r[None, None],
                                 covisible_only=True),
            "secs": time.time() - start,
        }

    rows = []
    if args.baseline_checkpoint:
        base = build(args.baseline_checkpoint, baseline=True)
        rows.append(run(base, False, "public (no CI)"))
        del base
    ours = build(args.checkpoint, baseline=False)
    if args.ablate_ci:
        rows.append(run(ours, False, "+CI ckpt, CI off"))
    rows.append(run(ours, True, "+CI ckpt, CI on" if args.ablate_ci else "+CI ckpt"))

    print(f"\n{'arm':<20}{'BP-2 L':>9}{'BP-2 R':>9}{'EPE L':>9}{'EPE R':>9}"
          f"{'LR-BP-1':>10}{'(co-vis)':>10}{'s':>7}")
    for r in rows:
        print(f"{r['arm']:<20}{r['bp2_l']:>9.2f}{r['bp2_r']:>9.2f}"
              f"{r['epe_l']:>9.3f}{r['epe_r']:>9.3f}{r['lr_bp1']:>10.2f}"
              f"{r['lr_bp1_cv']:>10.2f}{r['secs']:>7.0f}")

    on = rows[-1]
    print()
    if args.ablate_ci:
        off = rows[-2]
        print(f"cross-injection on vs off (same weights): "
              f"BP-2 L {on['bp2_l'] - off['bp2_l']:+.2f}, "
              f"LR-BP-1 {on['lr_bp1'] - off['lr_bp1']:+.2f}")
    if args.baseline_checkpoint:
        pub = rows[0]
        print(f"ours vs public baseline: "
              f"BP-2 L {on['bp2_l'] - pub['bp2_l']:+.2f}, "
              f"LR-BP-1 {on['lr_bp1'] - pub['lr_bp1']:+.2f}")

    print(reference_verdict(on, args.arch, is_reference_sample, default_settings,
                            args.device))


if __name__ == "__main__":
    main()
