# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Check both adapters on a GPU (or, slowly, on CPU).

Runs a subset of the test suite's checks with CUDA kernels, plus the released
S2M2 checkpoint on a whole sample (no downscaling), in fp32:

* the zero-init no-op check (random 128x160 input);
* a strict load of the released S2M2 checkpoint, and that its encoder is
  trained;
* a forward pass on the whole sample, scored against its ground truth
  (BP-2 of both views must stay below --max-bp2);
* with --with-foundationstereo, that cross-injection bolts onto a randomly
  initialised FoundationStereo as a no-op (needs no checkpoint).

    python scripts/smoke_test_gpu.py \\
        --s2m2-checkpoint /path/to/s2m2_xl_ci.pth --sample Motorcycle_H.npy

Pass --device cpu to run it without a GPU (about 1.5 min and 14 GB of RAM at
Middlebury-H).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from verify import bad_n, pair_gt_to_views

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))


def check_s2m2(args, device: str) -> None:
    from crossinj.adapters.s2m2 import CrossInjectionS2M2
    from crossinj.io import load_checkpoint
    from crossinj.loop import check_zero_init_is_noop

    fresh = CrossInjectionS2M2(model_size=args.model_size, refine_iter=3)
    fresh = fresh.to(device).eval()
    left = torch.rand(1, 3, 128, 160, device=device) * 255
    right = torch.rand(1, 3, 128, 160, device=device) * 255
    is_noop, delta = check_zero_init_is_noop(fresh, left, right)
    record("S2M2 zero-init cross-injection is a no-op", is_noop, f"max|d|={delta:.3e}")
    del fresh

    model = CrossInjectionS2M2(
        model_size=args.model_size, ci_channels=2,
        ci_erosion_iters=args.erosion_iters, refine_iter=3,
    )
    try:
        load_checkpoint(model, args.s2m2_checkpoint, strict=True)
    except Exception as err:  # noqa: BLE001 -- report, do not crash
        record("S2M2 released checkpoint loads strictly", False, str(err))
        return
    keys = model.state_dict()
    n_ci = sum("ci_encoder" in k for k in keys)
    record("S2M2 released checkpoint loads strictly", True,
           f"{len(keys) - n_ci} upstream tensors + {n_ci} cross-injection")
    record("S2M2 cross-injection encoder is trained",
           model.refiner.ci_encoder[2].weight.abs().sum().item() > 0)
    model = model.to(device).eval()

    blob = np.load(args.sample, allow_pickle=True).item()
    images, gt, valid = blob["images"], blob["gt"]["disparity"], blob["mask"]

    def to_tensor(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1)
        return t[None].to(device)

    left, right = to_tensor(images[0]), to_tensor(images[1])

    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    try:
        with torch.inference_mode():
            disp_l, disp_r = model(left, right, enabled=True)
    except Exception as err:  # noqa: BLE001 -- report, do not crash
        record("S2M2 forward on the whole sample", False, f"{type(err).__name__}: {err}")
        return
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - start
    record("S2M2 forward on the whole sample", True,
           f"{tuple(left.shape[-2:])} in {elapsed:.2f}s")
    record("S2M2 output shapes and finiteness",
           disp_l.shape == disp_r.shape == (1, 1, *left.shape[-2:])
           and bool(torch.isfinite(disp_l).all() and torch.isfinite(disp_r).all()),
           f"left {disp_l.min():.1f}..{disp_l.max():.1f} px")

    bp2 = []
    for disp, idx in zip((disp_l, disp_r), pair_gt_to_views(images, gt)):
        d = torch.from_numpy(np.nan_to_num(gt[idx]))
        m = torch.from_numpy(valid[idx] & np.isfinite(gt[idx]))
        bp2.append(bad_n(disp[0, 0].float().cpu(), d, m, 2.0))
    record("S2M2 accuracy against ground truth", max(bp2) < args.max_bp2,
           f"BP-2 L {bp2[0]:.2f}, R {bp2[1]:.2f} (limit {args.max_bp2})")

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"S2M2 peak GPU memory: {peak:.2f} GB")


def check_foundation_stereo(device: str) -> None:
    from crossinj.adapters.foundation_stereo import (
        CrossInjectionFoundationStereo,
        FoundationStereo,
        foundation_stereo_cfg,
        zero_pad_gru04,
    )

    cfg = foundation_stereo_cfg(max_disp=192)
    torch.manual_seed(0)
    base = FoundationStereo(cfg).to(device).eval()
    model = CrossInjectionFoundationStereo(
        cfg, ci_channels=2, load_depth_anything=False).to(device).eval()
    record("FoundationStereo gru04 widened 384 -> 386",
           model.update_block.gru04.conv0[0].weight.shape[0] == 386)

    state = {k: v.cpu() for k, v in base.state_dict().items()}
    padded = zero_pad_gru04(state, model)
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    record("FoundationStereo vanilla weights zero-pad into the widened model",
           padded == 10, f"{padded} tensors")

    left = torch.rand(1, 3, 128, 160, device=device) * 255
    right = torch.rand(1, 3, 128, 160, device=device) * 255
    with torch.inference_mode():
        expected = base(left, right, iters=4, test_mode=True)
        got, _ = model(left, right, iters=4, enabled=True)
    delta = float((got - expected).abs().max())
    record("FoundationStereo bolt-on reproduces vanilla", delta < 1e-3,
           f"max|d|={delta:.3e}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s2m2-checkpoint", required=True,
                    help="released S2M2-XL + CI checkpoint")
    ap.add_argument("--sample", required=True,
                    help="sample .npy from scripts/make_middlebury_sample.py")
    ap.add_argument("--model-size", default="XL", choices=["S", "M", "L", "XL"],
                    help="S2M2 size (default: XL, the released checkpoint)")
    ap.add_argument("--erosion-iters", type=int, default=2,
                    help="co-visibility erosion (default: 2, as released)")
    ap.add_argument("--max-bp2", type=float, default=10.0,
                    help="largest BP-2 (%%) of either view that passes (default: 10)")
    ap.add_argument("--with-foundationstereo", action="store_true",
                    help="also check the FoundationStereo adapter (downloads its "
                         "backbone parts on first use)")
    ap.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available -- run this on a GPU host, or pass --device cpu.",
              file=sys.stderr)
        return 2
    if args.device == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}\n")

    check_s2m2(args, args.device)
    if args.with_foundationstereo:
        print()
        check_foundation_stereo(args.device)

    failed = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
