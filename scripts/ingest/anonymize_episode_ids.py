#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_json_jsonl_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".json", ".jsonl"):
            continue
        # Skip assets (videos/images).
        parts = {s.lower() for s in p.parts}
        if "assets" in parts:
            continue
        out.append(p)
    out.sort(key=lambda x: str(x))
    return out


def _transform_obj(obj: Any, mapping: dict[str, str]) -> Any:
    # Keys whose string values can leak source timestamps/paths; redact to None by default.
    # (We keep per-episode relative refs like "records_ref" intact.)
    redact_keys = {
        "raw",
        "processed",
        "input_dir",
        "output_dataset_dir",
        "episode_dir",
        "episode_meta_path",
        "scene_dir",
        "dataset",
        "out_dataset",
    }
    if isinstance(obj, str):
        return mapping.get(obj, obj)
    if isinstance(obj, list):
        return [_transform_obj(x, mapping) for x in obj]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, str):
                # Redact common path-like keys (only for absolute paths).
                is_abs = os.path.isabs(v)
                if (k in redact_keys) or k.endswith("_path") or k.endswith("_dir"):
                    if is_abs:
                        out[k] = None
                        continue
            if k == "episode_id" and isinstance(v, str):
                out[k] = mapping.get(v, v)
                continue
            if k == "episodes" and isinstance(v, list) and all(isinstance(x, str) for x in v):
                out[k] = [mapping.get(str(x), str(x)) for x in v]
                continue
            out[k] = _transform_obj(v, mapping)
        return out
    return obj


def _rewrite_json_or_jsonl(path: Path, mapping: dict[str, str]) -> None:
    if path.suffix.lower() == ".json":
        obj = _read_json(path)
        obj2 = _transform_obj(obj, mapping)
        _write_json(path, obj2)
        return

    # jsonl
    tmp = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                fout.write(line)
                continue
            try:
                obj = json.loads(s)
            except Exception:
                fout.write(line)
                continue
            obj2 = _transform_obj(obj, mapping)
            fout.write(json.dumps(obj2, sort_keys=True) + "\n")
    tmp.replace(path)


def _load_episode_ids(dataset_root: Path) -> list[str]:
    manifest = dataset_root / "dataset_manifest.json"
    episodes_dir = dataset_root / "episodes"
    if manifest.exists():
        obj = _read_json(manifest)
        eps = obj.get("episodes", [])
        if isinstance(eps, list) and all(isinstance(x, str) for x in eps):
            return list(eps)
    if episodes_dir.is_dir():
        eps2 = sorted([p.name for p in episodes_dir.iterdir() if p.is_dir()])
        return eps2
    raise FileNotFoundError(f"Dataset missing dataset_manifest.json and episodes/: {dataset_root}")


def _make_ids(
    olds: list[str],
    *,
    prefix: str,
    length: int,
    seed: int | None,
) -> dict[str, str]:
    if length <= 0:
        raise ValueError("--id-length must be > 0")
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    used: set[str] = set()
    out: dict[str, str] = {}
    for old in olds:
        while True:
            s = "".join(rng.choice(alphabet) for _ in range(int(length)))
            new = f"{prefix}{s}" if prefix else s
            if new not in used:
                used.add(new)
                out[str(old)] = new
                break
    return out


def _copy_dataset_skeleton(dataset_root: Path, out_root: Path, *, overwrite: bool) -> None:
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists (use --overwrite): {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Copy everything except episodes/ (handled separately to rename dirs).
    for child in dataset_root.iterdir():
        if child.name == "episodes":
            continue
        dst = out_root / child.name
        if child.is_dir():
            shutil.copytree(child, dst)
        elif child.is_file():
            shutil.copy2(child, dst)

    (out_root / "episodes").mkdir(parents=True, exist_ok=True)


def anonymize_dataset(
    *,
    dataset_root: Path,
    out_root: Path,
    mapping: dict[str, str],
    overwrite: bool,
) -> None:
    dataset_root = dataset_root.resolve()
    out_root = out_root.resolve()
    _copy_dataset_skeleton(dataset_root, out_root, overwrite=overwrite)

    episodes_dir = dataset_root / "episodes"
    out_episodes_dir = out_root / "episodes"

    for old_id, new_id in mapping.items():
        src_ep = episodes_dir / old_id
        if not src_ep.is_dir():
            continue
        dst_ep = out_episodes_dir / new_id
        shutil.copytree(src_ep, dst_ep)

        # Update episode.json if present.
        ep_json = dst_ep / "episode.json"
        if ep_json.exists():
            try:
                obj = _read_json(ep_json)
                obj2 = _transform_obj(obj, mapping)
                # Ensure top-level episode_id matches directory.
                if isinstance(obj2, dict):
                    obj2["episode_id"] = str(new_id)
                _write_json(ep_json, obj2)
            except Exception:
                pass

    # Rewrite all json/jsonl metadata (including manifests/reports/splits).
    for p in _iter_json_jsonl_files(out_root):
        _rewrite_json_or_jsonl(p, mapping)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Copy a canonical dataset and anonymize episode IDs.")
    ap.add_argument("--dataset", required=True, help="Input canonical dataset root.")
    ap.add_argument("--out-dataset", required=True, help="Output canonical dataset root.")
    ap.add_argument(
        "--mapping-in",
        default=None,
        help="Optional JSON mapping old_episode_id->new_episode_id to reuse.",
    )
    ap.add_argument(
        "--mapping-out",
        default=None,
        help="Optional path to write the mapping JSON (keep private).",
    )
    ap.add_argument(
        "--prefix", default="ep_", help="Prefix for generated episode IDs (default: ep_)."
    )
    ap.add_argument("--id-length", type=int, default=12, help="Random suffix length (default: 12).")
    ap.add_argument(
        "--seed", type=int, default=None, help="Optional deterministic seed (default: random)."
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite output dataset if it exists."
    )
    args = ap.parse_args(argv)

    dataset_root = Path(str(args.dataset)).resolve()
    out_root = Path(str(args.out_dataset)).resolve()

    olds = _load_episode_ids(dataset_root)
    if not olds:
        raise SystemExit(f"No episodes found under: {dataset_root}")

    mapping: dict[str, str]
    if args.mapping_in:
        mapping_obj = _read_json(Path(str(args.mapping_in)))
        if not isinstance(mapping_obj, dict):
            raise SystemExit("--mapping-in must be a JSON object {old: new}")
        mapping = {str(k): str(v) for k, v in mapping_obj.items()}
        missing = [ep for ep in olds if ep not in mapping]
        if missing:
            # Extend mapping deterministically (seeded) if provided; else system random.
            extra = _make_ids(
                missing,
                prefix=str(args.prefix),
                length=int(args.id_length),
                seed=(int(args.seed) if args.seed is not None else None),
            )
            mapping.update(extra)
    else:
        mapping = _make_ids(
            olds,
            prefix=str(args.prefix),
            length=int(args.id_length),
            seed=(int(args.seed) if args.seed is not None else None),
        )

    # Ensure mapping is one-to-one.
    inv = {}
    for k, v in mapping.items():
        if v in inv:
            raise SystemExit(f"Duplicate new id generated: {v} for {k} and {inv[v]}")
        inv[v] = k

    anonymize_dataset(
        dataset_root=dataset_root,
        out_root=out_root,
        mapping=mapping,
        overwrite=bool(args.overwrite),
    )

    if args.mapping_out:
        out_p = Path(str(args.mapping_out)).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out_p, mapping)

    # Small stdout summary.
    print(f"input={dataset_root}")
    print(f"output={out_root}")
    print(f"episodes_mapped={len(mapping)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
