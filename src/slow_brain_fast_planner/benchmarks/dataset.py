from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from slow_brain_fast_planner.schema.canonical_episode import (
    CanonicalEpisode,
    OdomRecord,
    PlannerCandidatesRecord,
    RGBRecord,
    RouteRecord,
    StreamSpec,
)
from slow_brain_fast_planner.utils.io import read_json, read_jsonl


@dataclass(frozen=True)
class EpisodeLoadResult:
    episode_id: str
    episode_meta_path: str
    episode_dir: str
    schema_valid: bool
    schema_errors: list[str]
    episode: CanonicalEpisode | None
    planner_candidates: list[PlannerCandidatesRecord]
    planner_candidates_errors: list[str]
    rgb: list[RGBRecord]
    rgb_errors: list[str]
    odom: list[OdomRecord]
    odom_errors: list[str]
    route: list[RouteRecord]
    route_errors: list[str]


def find_episode_metadata_files(dataset_path: Path) -> list[Path]:
    """Discover episode metadata files in a canonical dataset directory.

    Supports:
    - DATASET_ROOT/episodes/<episode_id>/episode.json (recommended)
    - DATASET_ROOT/episodes/<episode_id>.json
    - or a directory directly containing episode JSON files
    """

    dataset_path = dataset_path.resolve()
    episodes_root = dataset_path / "episodes"
    if not episodes_root.is_dir():
        episodes_root = dataset_path

    out: list[Path] = []
    for p in episodes_root.iterdir():
        if p.is_file() and p.suffix == ".json" and p.name != "dataset_manifest.json":
            out.append(p)
        elif p.is_dir():
            ep = p / "episode.json"
            if ep.is_file():
                out.append(ep)
    out.sort(key=lambda x: str(x))
    return out


def _resolve_records_ref(
    records_ref: Any,
    *,
    episode_dir: Path,
    dataset_root: Path,
) -> Path | None:
    """Resolve a `records_ref` into a local file path (best-effort)."""

    if isinstance(records_ref, str):
        ref_path = Path(records_ref)
    elif isinstance(records_ref, dict) and isinstance(records_ref.get("path"), str):
        ref_path = Path(records_ref["path"])
    else:
        return None

    if ref_path.is_absolute():
        return ref_path

    cand1 = (episode_dir / ref_path).resolve()
    if cand1.exists():
        return cand1

    cand2 = (dataset_root / ref_path).resolve()
    if cand2.exists():
        return cand2

    return cand1


def _load_stream_records(
    stream_name: str,
    stream_spec: StreamSpec | None,
    *,
    episode_dir: Path,
    dataset_root: Path,
) -> tuple[list[Any], list[str]]:
    if stream_spec is None:
        return [], [f"missing_stream:{stream_name}"]

    if stream_spec.records is not None:
        return stream_spec.records, []

    ref_path = _resolve_records_ref(
        stream_spec.records_ref, episode_dir=episode_dir, dataset_root=dataset_root
    )
    if ref_path is None:
        return [], [f"unsupported_records_ref:{stream_spec.records_ref!r}"]
    if not ref_path.exists():
        return [], [f"missing_records_ref_file:{ref_path}"]

    try:
        if ref_path.suffix == ".jsonl":
            return read_jsonl(ref_path), []
        if ref_path.suffix == ".json":
            raw_obj = read_json(ref_path)
            if isinstance(raw_obj, list):
                return raw_obj, []
            if isinstance(raw_obj, dict) and isinstance(raw_obj.get("records"), list):
                return raw_obj["records"], []
            return [], [f"bad_json_records_format:{ref_path}"]
        return [], [f"unsupported_records_ref_suffix:{ref_path.suffix}"]
    except Exception as e:  # noqa: BLE001
        return [], [f"failed_to_read_records_ref:{ref_path}:{e}"]


def load_episode(dataset_root: Path, episode_meta_path: Path) -> EpisodeLoadResult:
    """Load episode metadata + planner_candidates + rgb streams (Tier P1)."""

    dataset_root = dataset_root.resolve()
    episode_meta_path = episode_meta_path.resolve()
    episode_dir = episode_meta_path.parent
    episode_id_guess = (
        episode_dir.name if episode_meta_path.name == "episode.json" else episode_meta_path.stem
    )

    schema_errors: list[str] = []
    episode: CanonicalEpisode | None = None
    try:
        episode_obj = read_json(episode_meta_path)
        episode = CanonicalEpisode.model_validate(episode_obj)
    except ValidationError as e:
        schema_errors.append(str(e))
    except Exception as e:  # noqa: BLE001
        schema_errors.append(str(e))

    if episode is None:
        return EpisodeLoadResult(
            episode_id=episode_id_guess,
            episode_meta_path=str(episode_meta_path),
            episode_dir=str(episode_dir),
            schema_valid=False,
            schema_errors=schema_errors,
            episode=None,
            planner_candidates=[],
            planner_candidates_errors=["episode_schema_invalid"],
            rgb=[],
            rgb_errors=["episode_schema_invalid"],
            odom=[],
            odom_errors=["episode_schema_invalid"],
            route=[],
            route_errors=["episode_schema_invalid"],
        )

    # planner_candidates
    raw_records, record_errors = _load_stream_records(
        "planner_candidates",
        episode.streams.get("planner_candidates"),
        episode_dir=episode_dir,
        dataset_root=dataset_root,
    )
    pc_errors: list[str] = list(record_errors)
    pc_records: list[PlannerCandidatesRecord] = []
    for i, r in enumerate(raw_records):
        try:
            pc_records.append(PlannerCandidatesRecord.model_validate(r))
        except ValidationError as e:
            pc_errors.append(f"record[{i}]:{e}")
    pc_records.sort(key=lambda r: float(r.t))

    # rgb
    raw_rgb, rgb_record_errors = _load_stream_records(
        "rgb",
        episode.streams.get("rgb"),
        episode_dir=episode_dir,
        dataset_root=dataset_root,
    )
    rgb_errors: list[str] = list(rgb_record_errors)
    rgb_records: list[RGBRecord] = []
    for i, r in enumerate(raw_rgb):
        try:
            rgb_records.append(RGBRecord.model_validate(r))
        except ValidationError as e:
            rgb_errors.append(f"record[{i}]:{e}")
    rgb_records.sort(key=lambda r: float(r.t))

    # odom
    raw_odom, odom_record_errors = _load_stream_records(
        "odom",
        episode.streams.get("odom"),
        episode_dir=episode_dir,
        dataset_root=dataset_root,
    )
    odom_errors: list[str] = list(odom_record_errors)
    odom_records: list[OdomRecord] = []
    for i, r in enumerate(raw_odom):
        try:
            odom_records.append(OdomRecord.model_validate(r))
        except ValidationError as e:
            odom_errors.append(f"record[{i}]:{e}")
    odom_records.sort(key=lambda r: float(r.t))

    # route
    raw_route, route_record_errors = _load_stream_records(
        "route",
        episode.streams.get("route"),
        episode_dir=episode_dir,
        dataset_root=dataset_root,
    )
    route_errors: list[str] = list(route_record_errors)
    route_records: list[RouteRecord] = []
    for i, r in enumerate(raw_route):
        try:
            route_records.append(RouteRecord.model_validate(r))
        except ValidationError as e:
            route_errors.append(f"record[{i}]:{e}")
    route_records.sort(key=lambda r: float(r.t))

    return EpisodeLoadResult(
        episode_id=episode.episode_id,
        episode_meta_path=str(episode_meta_path),
        episode_dir=str(episode_dir),
        schema_valid=True,
        schema_errors=[],
        episode=episode,
        planner_candidates=pc_records,
        planner_candidates_errors=pc_errors,
        rgb=rgb_records,
        rgb_errors=rgb_errors,
        odom=odom_records,
        odom_errors=odom_errors,
        route=route_records,
        route_errors=route_errors,
    )
