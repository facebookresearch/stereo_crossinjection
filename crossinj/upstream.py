# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Make the upstream stereo matchers importable from their git submodules.

S2M2 and FoundationStereo are git submodules under ``third_party/``, pinned to
the commits this code was tested against. They are linked against, not copied:
their code stays in their own repositories under their own licences (the one
adapted excerpt, S2M2's correlation lookup, is attributed in
:mod:`crossinj.compat`).

Neither upstream is a regular installable dependency -- FoundationStereo has no
package metadata and imports its modules as top-level ``core`` and ``Utils`` --
so the adapters import them through :func:`import_upstream`, which puts the
submodule on ``sys.path`` first. This needs the repository checkout itself
(``git clone --recursive`` and ``pip install -e .``); a non-editable install
does not ship ``third_party/``, so it falls back to whatever is already on
``PYTHONPATH``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

__all__ = ["THIRD_PARTY", "add_to_path", "import_upstream"]

THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"

# Import root of each upstream inside its submodule.
_ROOTS = {
    "s2m2": THIRD_PARTY / "s2m2" / "src",
    "foundation_stereo": THIRD_PARTY / "FoundationStereo",
}

# How to install each upstream's own dependencies.
_EXTRAS = {
    "s2m2": "pip install -e .",
    "foundation_stereo": "pip install -e '.[foundationstereo]'",
}


def add_to_path(name: str) -> bool:
    """Prepend upstream ``name``'s submodule to ``sys.path``.

    Args:
        name: ``"s2m2"`` or ``"foundation_stereo"``.

    Returns:
        True if the submodule is checked out (and is now on the path), False if
        it is missing -- run ``git submodule update --init`` in that case.
    """
    root = _ROOTS[name]
    if not root.is_dir() or not any(root.iterdir()):
        return False
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return True


def import_upstream(name: str, module: str) -> ModuleType:
    """Import ``module`` from upstream ``name``, with an actionable error.

    Args:
        name: ``"s2m2"`` or ``"foundation_stereo"``.
        module: Dotted module path inside that upstream, e.g.
            ``"core.foundation_stereo"``.
    """
    checked_out = add_to_path(name)
    try:
        imported = importlib.import_module(module)
    except ModuleNotFoundError as err:
        if checked_out:
            # A genuinely missing dependency of the upstream code.
            raise ModuleNotFoundError(
                f"cannot import {module!r}: upstream {name} needs the "
                f"{err.name!r} package, which is not installed. Install its "
                f"dependencies with `{_EXTRAS[name]}`.",
                name=err.name,
            ) from err
        submodule = _ROOTS[name].relative_to(THIRD_PARTY.parent).parts[:2]
        raise ImportError(
            f"cannot import {module!r}: the {'/'.join(submodule)} submodule is "
            "not checked out. Run `git submodule update --init` in the "
            "repository and install it with `pip install -e .`."
        ) from err

    # FoundationStereo's top-level packages (`core`, ...) are namespace
    # packages, which any regular package of the same name shadows no matter
    # where it sits on sys.path.
    root = _ROOTS[name]
    if checked_out and root.resolve() not in Path(imported.__file__).resolve().parents:
        raise ImportError(
            f"{module!r} resolved to {imported.__file__}, not to the submodule "
            f"under {root}: another package of the same name shadows it. "
            "Remove it from sys.path (or the current directory)."
        )
    return imported
