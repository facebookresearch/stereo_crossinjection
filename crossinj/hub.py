# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download the released checkpoints from the Hugging Face Hub."""

from __future__ import annotations

__all__ = ["CHECKPOINTS", "REPO_ID", "download_checkpoint"]

REPO_ID = "facebook/stereo_crossinjection"

# Released checkpoints: name -> files in the Hub repo, the checkpoint first.
CHECKPOINTS: dict[str, tuple[str, ...]] = {
    "s2m2_xl": ("s2m2_xl_ci.pth",),
    "foundation_stereo": ("foundationstereo_ci.pth", "foundationstereo_ci_cfg.yaml"),
}


def download_checkpoint(name: str, cache_dir: str | None = None,
                        revision: str | None = None) -> str:
    """Download a released checkpoint (once) and return its local path.

    Files go to the Hugging Face cache (``$HF_HOME``) unless ``cache_dir`` is
    given; FoundationStereo's reference config is fetched along with its
    checkpoint.

    Args:
        name: ``"s2m2_xl"`` or ``"foundation_stereo"``.
        cache_dir: Optional cache directory.
        revision: Optional Hub revision (branch, tag or commit) to pin.

    Raises:
        PermissionError: The Hub refused access. The repository is gated:
            request access on the model page (requests are approved
            manually), then log in with the approved account.
    """
    if name not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint {name!r}; expected one of {sorted(CHECKPOINTS)}")
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
    except ImportError as err:
        raise ImportError("downloading checkpoints needs huggingface_hub: "
                          "pip install huggingface_hub") from err

    paths = []
    for filename in CHECKPOINTS[name]:
        try:
            paths.append(hf_hub_download(REPO_ID, filename, cache_dir=cache_dir,
                                         revision=revision))
        except (RepositoryNotFoundError, HfHubHTTPError) as err:
            # Gated and private repos answer 401/403, or 404 to a token without
            # access, rather than a dedicated error.
            status = getattr(getattr(err, "response", None), "status_code", None)
            if not isinstance(err, RepositoryNotFoundError) and status not in (401, 403):
                raise
            raise PermissionError(
                f"could not download {filename} from {REPO_ID}: access denied "
                f"(HTTP {status}). The repository is gated: request access on "
                f"https://huggingface.co/{REPO_ID} (requests are approved "
                "manually), then log in with `hf auth login` (or set HF_TOKEN) "
                "using the approved account."
            ) from err
    return paths[0]
