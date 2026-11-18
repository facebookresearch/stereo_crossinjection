# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run a released +CI model on your own rectified stereo pair.

Writes both views' disparities to --out: disp_left.npy / disp_right.npy
(float32, pixels; the right map in right-image convention, i.e. a right pixel
x matches left pixel x + d) and colourised disp_left.png / disp_right.png on a
shared colour scale.

    python scripts/demo.py --left im0.png --right im1.png --out out/

    python scripts/demo.py --arch foundation_stereo \\
        --checkpoint checkpoints/foundationstereo_ci.pth \\
        --left im0.png --right im1.png --out out/

Without --checkpoint the released checkpoint is downloaded from the Hugging
Face Hub (gated: request access on the model page -- requests are approved
manually -- and run `hf auth login` with the approved account first). Images
can be any size, but FoundationStereo's cost volume covers disparities up to
--max-disp only (416 px by default, as trained); downscale pairs with larger
disparities with --scale. On CPU, a 1000x1500 pair takes about 1.5 min and
14 GB of RAM with S2M2-XL, 3.5-4.5 min and 24 GB with FoundationStereo.
--scale 0.5 is about 4x cheaper; the saved disparities are then at half
resolution, in half-resolution pixels.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch


def build_model(arch: str, checkpoint: str | None,
                max_disp: int = 416) -> torch.nn.Module:
    from crossinj.hub import download_checkpoint
    from crossinj.io import load_checkpoint

    if checkpoint is None:
        try:
            checkpoint = download_checkpoint(
                "foundation_stereo" if arch == "foundation_stereo" else "s2m2_xl")
        except PermissionError as err:
            raise SystemExit(f"error: {err}") from err
    if arch == "foundation_stereo":
        from crossinj.adapters.foundation_stereo import (
            CrossInjectionFoundationStereo,
            foundation_stereo_cfg,
        )

        model = CrossInjectionFoundationStereo(
            foundation_stereo_cfg(max_disp=max_disp), ci_erosion_iters=1)
    else:
        from crossinj.adapters.s2m2 import CrossInjectionS2M2

        model = CrossInjectionS2M2(model_size="XL", ci_erosion_iters=2, refine_iter=3)
    load_checkpoint(model, checkpoint)
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", required=True, help="left image")
    ap.add_argument("--right", required=True, help="right image")
    ap.add_argument("--arch", default="s2m2", choices=["s2m2", "foundation_stereo"],
                    help="released model (default: s2m2, i.e. S2M2-XL + CI)")
    ap.add_argument("--checkpoint",
                    help="local checkpoint; default: download the released one")
    ap.add_argument("--out", default="out", help="output directory (default: out)")
    ap.add_argument("--device", default="auto",
                    help="auto (default: cuda if available, else cpu), cuda or cpu")
    ap.add_argument("--iters", type=int, default=None,
                    help="GRU iterations; default 3 for S2M2, 32 for FoundationStereo")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="resize both images first (disparities are then at that scale)")
    ap.add_argument("--max-disp", type=int, default=416,
                    help="FoundationStereo only: disparity range of its cost "
                         "volume, in px after --scale (default: 416, as trained)")
    args = ap.parse_args()

    from crossinj.io import colorize_disparity, load_image

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    left = load_image(args.left, device=device, scale=args.scale)
    right = load_image(args.right, device=device, scale=args.scale)
    print(f"input {tuple(left.shape[-2:])} on {device}")

    model = build_model(args.arch, args.checkpoint, args.max_disp).to(device)
    if args.arch == "foundation_stereo":
        kwargs = {"iters": args.iters or 32}
    else:
        model.refine_iter = args.iters or model.refine_iter
        kwargs = {}

    start = time.time()
    with torch.inference_mode():
        disp_l, disp_r = model(left, right, **kwargs)
    print(f"forward pass: {time.time() - start:.1f} s")

    os.makedirs(args.out, exist_ok=True)
    disp_l, disp_r = disp_l[0, 0].float().cpu(), disp_r[0, 0].float().cpu()
    # One colour scale for both views, so they can be compared side by side.
    vmin = float(np.percentile(disp_l.numpy(), 2))
    vmax = float(np.percentile(disp_l.numpy(), 98))
    import cv2

    for view, disp in (("left", disp_l), ("right", disp_r)):
        np.save(os.path.join(args.out, f"disp_{view}.npy"), disp.numpy())
        rgb = colorize_disparity(disp, vmin=vmin, vmax=vmax)
        cv2.imwrite(os.path.join(args.out, f"disp_{view}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"{view:>5}: {tuple(disp.shape)}, {float(disp.min()):.2f} .. "
              f"{float(disp.max()):.2f} px -> {args.out}/disp_{view}.npy, .png")


if __name__ == "__main__":
    main()
