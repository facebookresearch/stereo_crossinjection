# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Left-right-consistent iterative stereo matching via cross-injection.

A bolt-on module for iterative stereo matchers. See README.md.
"""

from .encoder import CrossInjectionEncoder
from .geometry import covisibility, inconsistency, nonocc_single_view, warp_disp
from .io import load_checkpoint
from .loop import check_zero_init_is_noop, cross_view_cue, split_views, stack_views

__version__ = "0.1.0"

__all__ = [
    "CrossInjectionEncoder",
    "check_zero_init_is_noop",
    "covisibility",
    "cross_view_cue",
    "inconsistency",
    "load_checkpoint",
    "nonocc_single_view",
    "split_views",
    "stack_views",
    "warp_disp",
]
