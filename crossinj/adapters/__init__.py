# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-backbone adapters.

Each adapter imports its upstream repo at module level, from the git submodule
under ``third_party/`` (``git submodule update --init``), so importing one
without it raises ImportError. That is deliberate: the upstream models are
linked against, not copied here. The one exception is S2M2's correlation
lookup, of which :mod:`crossinj.compat` carries an adapted, attributed copy.
"""
