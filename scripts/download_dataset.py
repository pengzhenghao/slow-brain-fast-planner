#!/usr/bin/env python
"""Download a Slow Brain, Fast Planner dataset split from HuggingFace.

Both splits live in a single dataset repo (default
``pengzhenghao97/slow-brain-fast-planner-dataset``) as top-level subfolders ``mini/``,
``hard/``. ``--split <name>`` fetches just one subfolder via
``allow_patterns``; ``--split all`` fetches the whole repo.

The repo is mirrored into ``--out`` (default ``data/slow-brain-fast-planner``), so a split
ends up at ``<out>/<split>`` — e.g. ``data/slow-brain-fast-planner/mini`` — which is the
path to pass as ``--dataset`` to the eval CLIs.

Usage:
    python scripts/download_dataset.py --split mini
    python scripts/download_dataset.py --split hard --out /path/to/data
    python scripts/download_dataset.py --split all

The dataset is private. Use ``HF_TOKEN`` (or ``hf auth login``) for an account
with an explicit access grant. Authentication alone does not grant access.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "pengzhenghao97/slow-brain-fast-planner-dataset"
SPLITS = ("mini", "hard")


def _available_splits(repo_id: str, revision: str | None, token: str | None) -> list[str]:
    from huggingface_hub import HfApi

    entries = HfApi(token=token).list_repo_tree(
        repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=False,
    )
    top_level = {
        str(path).split("/", 1)[0] for entry in entries if (path := getattr(entry, "path", None))
    }
    return [s for s in SPLITS if s in top_level]


def download(
    split: str,
    out_dir: Path,
    repo_id: str,
    revision: str | None,
) -> Path:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    available = _available_splits(repo_id, revision, token)
    if not available:
        raise SystemExit(f"No splits found in {repo_id}; nothing to download.")
    if split != "all" and split not in available:
        raise SystemExit(
            f"Split '{split}' is not available in {repo_id} yet "
            f"(available: {', '.join(available)})."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    if split == "all":
        allow_patterns = None
        logger.info("Downloading entire repo %s -> %s", repo_id, out_dir)
    else:
        allow_patterns = [f"{split}/**"]
        logger.info("Downloading %s/%s/** -> %s", repo_id, split, out_dir)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(out_dir),
        token=token,
        allow_patterns=allow_patterns,
    )

    if split == "all":
        for name in available:
            logger.info("Dataset ready: %s", out_dir / name)
        return out_dir
    dataset_dir = out_dir / split
    logger.info("Dataset ready: %s (pass this path as --dataset)", dataset_dir)
    return dataset_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--split", choices=[*SPLITS, "all"], required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data") / "slow-brain-fast-planner",
        help=(
            "Directory the repo is mirrored into (default: data/slow-brain-fast-planner). "
            "A single split ends up at <out>/<split>."
        ),
    )
    p.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF dataset repo id (default: {DEFAULT_REPO_ID})",
    )
    p.add_argument("--revision", default=None, help="Optional git revision / tag")
    args = p.parse_args(argv)

    download(args.split, args.out, args.repo_id, args.revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
