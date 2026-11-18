# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Turn a Middlebury 2014 scene into a half-resolution ``verify.py`` sample.

Middlebury is not redistributed here; download a scene from
https://vision.middlebury.edu/stereo/data/scenes2014/ first, e.g.

    wget https://vision.middlebury.edu/stereo/data/scenes2014/zip/Motorcycle-perfect.zip
    unzip Motorcycle-perfect.zip
    python scripts/make_middlebury_sample.py Motorcycle-perfect Motorcycle_H.npy

Half resolution is the paper's Middlebury-H setting: images are area-averaged,
ground truth is subsampled and its disparities halved. Both views' ground truth
(``disp0.pfm``, ``disp1.pfm``) is kept, so ``verify.py`` scores both.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        if f.readline().strip() != b"Pf":
            raise ValueError(f"{path}: not a single-channel PFM")
        width, height = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.fromfile(f, "<f4" if scale < 0 else ">f4").reshape(height, width)
    return np.flipud(data)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="unzipped Middlebury 2014 scene directory")
    ap.add_argument("out", help="output .npy")
    args = ap.parse_args()

    images, gts = [], []
    for view in (0, 1):
        image = cv2.cvtColor(cv2.imread(f"{args.scene}/im{view}.png"), cv2.COLOR_BGR2RGB)
        h, w = image.shape[0] // 2, image.shape[1] // 2
        images.append(cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA))
        disp = read_pfm(f"{args.scene}/disp{view}.pfm").astype(np.float32)
        disp[~np.isfinite(disp)] = np.nan
        gts.append(disp[::2, ::2][:h, :w] / 2.0)

    gt = np.stack(gts)
    np.save(args.out, {"images": np.stack(images), "gt": {"disparity": gt},
                       "mask": np.isfinite(gt)}, allow_pickle=True)
    print(f"wrote {args.out}: {gt.shape[1]}x{gt.shape[2]}, "
          f"{np.isfinite(gt).mean():.1%} pixels with ground truth")


if __name__ == "__main__":
    main()
