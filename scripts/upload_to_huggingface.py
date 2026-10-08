#!/usr/bin/env python
"""Upload a Slow Brain, Fast Planner dataset split to a single HuggingFace dataset repo.

Splits (mini, hard) live in a single repo (default
``pengzhenghao97/slow-brain-fast-planner-dataset``) as top-level subfolders. This script
uploads one split at a time into its corresponding subfolder via
``--path-in-repo``.

Usage:
    # Mini split (~125 MB)
    python scripts/upload_to_huggingface.py --src /path/to/mini --split mini

    # Hard split (~10 GB) — use --large for resumable chunked upload
    python scripts/upload_to_huggingface.py --src /path/to/hard_root --split hard --large

    # Make the entire repo public (do this once, on release day)
    python scripts/upload_to_huggingface.py --make-public

Requires ``hf auth login`` first (or ``HF_TOKEN`` env). The first upload
creates the repo (private by default).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "pengzhenghao97/slow-brain-fast-planner-dataset"
SPLITS = ("mini", "hard")


def _api():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return HfApi(token=token)


def _file_inventory(root: Path) -> list[tuple[str, int]]:
    return sorted(
        (str(path.relative_to(root)), int(path.stat().st_size))
        for path in root.rglob("*")
        if path.is_file()
    )


def _stage_matches_source(src: Path, staged: Path) -> bool:
    if _file_inventory(src) != _file_inventory(staged):
        return False
    for source_path in src.rglob("*"):
        if not source_path.is_file():
            continue
        staged_path = staged / source_path.relative_to(src)
        if not staged_path.is_file() or not os.path.samefile(source_path, staged_path):
            return False
    return True


def _prepare_large_upload_stage(src: Path, split: str, repo_id: str) -> Path:
    """Build a complete hardlink mirror for upload_large_folder."""
    repo_key = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()[:12]
    stage = src.parent / f".hf_stage_{split}_{repo_key}"
    stage_split = stage / split
    if stage_split.is_dir() and _stage_matches_source(src, stage_split):
        return stage

    building = src.parent / f".hf_stage_{split}.building"
    if building.exists():
        shutil.rmtree(building)
    if stage.exists():
        shutil.rmtree(stage)

    try:
        building.mkdir()
        shutil.copytree(src, building / split, copy_function=os.link)
        if not _stage_matches_source(src, building / split):
            raise RuntimeError("Staged upload mirror does not match the source tree.")
        building.replace(stage)
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise

    return stage


def upload(
    src: Path,
    repo_id: str,
    split: str,
    *,
    public: bool,
    card_path: Path | None,
    dry_run: bool,
    commit_message: str,
    large: bool,
) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"src is not a directory: {src}")

    files = sorted(src.rglob("*"))
    n_files = sum(1 for f in files if f.is_file())
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    logger.info(
        "src=%s repo_id=%s path_in_repo=%s files=%d size=%.1f MB",
        src,
        repo_id,
        split,
        n_files,
        total_size / 1e6,
    )

    if dry_run:
        for f in files:
            if f.is_file():
                rel = f.relative_to(src)
                print(f"  {split}/{rel}  ({f.stat().st_size / 1e6:.2f} MB)")
        return

    api = _api()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=not public, exist_ok=True)
    logger.info("Repo ready: https://huggingface.co/datasets/%s", repo_id)

    if card_path is not None:
        if not card_path.is_file():
            raise FileNotFoundError(f"--card-path not found: {card_path}")
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update dataset card",
        )
        logger.info("Uploaded README.md (dataset card)")

    if large:
        # upload_large_folder cannot target a subfolder (no path_in_repo), so upload a
        # validated hardlink mirror at <stage>/<split>/.... The stage lives beside src
        # so hardlinks remain on one filesystem and an unchanged upload can resume.
        stage = _prepare_large_upload_stage(src, split, repo_id)
        logger.info("Staged hardlink mirror at %s", stage / split)
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(stage),
        )
    else:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(src),
            path_in_repo=split,
            commit_message=commit_message,
        )

    logger.info(
        "Upload complete: https://huggingface.co/datasets/%s/tree/main/%s",
        repo_id,
        split,
    )


def make_public(repo_id: str) -> None:
    api = _api()
    api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=False)
    logger.info("Set %s public.", repo_id)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Local dataset root for the split being uploaded (canonical episodes/, "
        "takeover_clips/, ...)",
    )
    p.add_argument(
        "--split",
        choices=SPLITS,
        default=None,
        help="Which subfolder to upload into. Required when --src is given.",
    )
    p.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF dataset repo id (default: {DEFAULT_REPO_ID})",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Create / leave the repo public (default: private on create).",
    )
    p.add_argument(
        "--card-path",
        type=Path,
        default=None,
        help="Path to a README.md dataset card to upload at the repo root.",
    )
    p.add_argument("--dry-run", action="store_true", help="List files without uploading")
    p.add_argument("--commit-message", default="Upload split")
    p.add_argument(
        "--large",
        action="store_true",
        help="Use upload_large_folder (recommended for >5 GB or >10k files; resumes).",
    )
    p.add_argument(
        "--make-public",
        action="store_true",
        help="Flip the repo to public visibility and exit (no upload).",
    )
    args = p.parse_args(argv)

    if args.make_public:
        if args.dry_run:
            logger.info("DRY RUN: would set %s public.", args.repo_id)
            return 0
        make_public(args.repo_id)
        return 0

    if args.src is None or args.split is None:
        p.error("Need --src and --split (or --make-public).")

    upload(
        args.src,
        args.repo_id,
        args.split,
        public=args.public,
        card_path=args.card_path,
        dry_run=args.dry_run,
        commit_message=f"{args.commit_message}: {args.split}",
        large=args.large,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
