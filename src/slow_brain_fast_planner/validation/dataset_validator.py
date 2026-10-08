from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ValidationError

from slow_brain_fast_planner.schema.canonical_episode import (
    CANONICAL_EPISODE_SCHEMA_VERSION,
    CanonicalEpisode,
    ControlRecord,
    GPSRecord,
    OdomRecord,
    PlannerCandidatesRecord,
    RGBRecord,
    RouteRecord,
    StreamSpec,
)
from slow_brain_fast_planner.utils.io import read_json, read_jsonl

KNOWN_STREAMS: tuple[str, ...] = (
    "gps",
    "odom",
    "rgb",
    "route",
    "planner_candidates",
    "control",
)

TaskName: TypeAlias = Literal["trajectory_selection"]


@dataclass(frozen=True)
class DatasetValidationConfig:
    """Configuration for snapshot validity checks.

    This is intentionally small and benchmark-core-agnostic; it is only used for the
    completeness report.
    """

    horizon_s: float = 5.0
    time_tolerance_s: float = 0.2
    snapshot_stride_s: float | None = None
    max_snapshots_per_episode: int | None = None
    max_episodes: int | None = None
    include_episode_details: bool = True

    def __post_init__(self) -> None:
        if self.horizon_s <= 0:
            raise ValueError("horizon_s must be > 0")
        if self.time_tolerance_s < 0:
            raise ValueError("time_tolerance_s must be >= 0")
        if self.snapshot_stride_s is not None and self.snapshot_stride_s <= 0:
            raise ValueError("snapshot_stride_s must be > 0 if set")
        if self.max_snapshots_per_episode is not None and self.max_snapshots_per_episode <= 0:
            raise ValueError("max_snapshots_per_episode must be > 0 if set")
        if self.max_episodes is not None and self.max_episodes <= 0:
            raise ValueError("max_episodes must be > 0 if set")


@dataclass(frozen=True)
class LoadedStream:
    name: str
    records: list[BaseModel]
    parse_errors: list[str]
    source: str  # "inline" | "ref:<path>" | "missing" | "unreadable"

    @property
    def times(self) -> list[float]:
        out: list[float] = []
        for r in self.records:
            t = getattr(r, "t", None)
            if isinstance(t, (int, float)):
                out.append(float(t))
        return out


@dataclass(frozen=True)
class TimeIndex:
    """Binary-searchable time index over a stream."""

    times: list[float]
    records: list[BaseModel]

    @classmethod
    def from_records(cls, records: Iterable[BaseModel]) -> TimeIndex:
        pairs = [(float(r.t), r) for r in records if hasattr(r, "t")]
        pairs.sort(key=lambda x: x[0])
        return cls(times=[t for t, _ in pairs], records=[r for _, r in pairs])

    def nearest(self, t: float, tol: float) -> BaseModel | None:
        # Simple binary search.
        from bisect import bisect_left

        if not self.times:
            return None
        i = bisect_left(self.times, t)
        candidates: list[tuple[float, int]] = []
        if 0 <= i < len(self.times):
            candidates.append((abs(self.times[i] - t), i))
        if 0 <= i - 1 < len(self.times):
            candidates.append((abs(self.times[i - 1] - t), i - 1))
        if not candidates:
            return None
        dist, idx = min(candidates, key=lambda x: x[0])
        if dist <= tol:
            return self.records[idx]
        return None

    def has_time_at_or_after(self, t: float) -> bool:
        from bisect import bisect_left

        if not self.times:
            return False
        return bisect_left(self.times, t) < len(self.times)

    def max_time(self) -> float | None:
        return self.times[-1] if self.times else None


def _resolve_records_ref(
    records_ref: Any,
    *,
    episode_dir: Path,
    dataset_root: Path,
) -> Path | None:
    """Resolve a `records_ref` into a local file path (best-effort).

    Spec allows arbitrary object-store keys; the validator only understands local paths.
    """

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

    # Fall back to episode-relative even if it doesn't exist, for better error messages.
    return cand1


def _parse_records(stream_name: str, raw_records: list[Any]) -> tuple[list[BaseModel], list[str]]:
    model: type[BaseModel] | None = {
        "gps": GPSRecord,
        "odom": OdomRecord,
        "rgb": RGBRecord,
        "route": RouteRecord,
        "planner_candidates": PlannerCandidatesRecord,
        "control": ControlRecord,
    }.get(stream_name)

    if model is None:
        # Unknown stream: keep raw dicts as-is (but still enforce that `t` exists if present).
        parsed: list[BaseModel] = []
        errors: list[str] = []
        for i, r in enumerate(raw_records):
            if isinstance(r, dict) and isinstance(r.get("t"), (int, float)):
                parsed.append(_DictRecord.model_validate(r))
            else:
                errors.append(f"record[{i}] is not a dict with numeric 't'")
        return parsed, errors

    parsed_typed: list[BaseModel] = []
    errors_typed: list[str] = []
    for i, r in enumerate(raw_records):
        try:
            parsed_typed.append(model.model_validate(r))
        except ValidationError as e:
            errors_typed.append(f"record[{i}] {e.__class__.__name__}: {e}")
    return parsed_typed, errors_typed


class _DictRecord(BaseModel):
    """Helper for unknown streams (only cares about `t`)."""

    t: float

    model_config = {"extra": "allow"}


def _load_stream(
    stream_name: str,
    stream_spec: StreamSpec | None,
    *,
    episode_dir: Path,
    dataset_root: Path,
) -> LoadedStream:
    if stream_spec is None:
        return LoadedStream(name=stream_name, records=[], parse_errors=[], source="missing")

    if stream_spec.records is not None:
        records, errs = _parse_records(stream_name, stream_spec.records)
        return LoadedStream(
            name=stream_name,
            records=records,
            parse_errors=errs,
            source="inline",
        )

    ref_path = _resolve_records_ref(
        stream_spec.records_ref, episode_dir=episode_dir, dataset_root=dataset_root
    )
    if ref_path is None:
        return LoadedStream(
            name=stream_name,
            records=[],
            parse_errors=[
                f"records_ref is not a supported local path: {stream_spec.records_ref!r}"
            ],
            source="unreadable",
        )

    if not ref_path.exists():
        return LoadedStream(
            name=stream_name,
            records=[],
            parse_errors=[f"records_ref path does not exist: {ref_path}"],
            source=f"ref:{ref_path}",
        )

    try:
        if ref_path.suffix == ".jsonl":
            raw_records = read_jsonl(ref_path)
        elif ref_path.suffix == ".json":
            raw_obj = read_json(ref_path)
            if isinstance(raw_obj, list):
                raw_records = raw_obj
            elif isinstance(raw_obj, dict) and isinstance(raw_obj.get("records"), list):
                raw_records = raw_obj["records"]
            else:
                raise ValueError("Expected list or {'records': [...]} JSON")
        else:
            raise ValueError(f"Unsupported records_ref file type: {ref_path.suffix}")
    except Exception as e:  # noqa: BLE001 (we want robust reporting)
        return LoadedStream(
            name=stream_name,
            records=[],
            parse_errors=[f"Failed to read records_ref {ref_path}: {e}"],
            source=f"ref:{ref_path}",
        )

    records, errs = _parse_records(stream_name, raw_records)
    return LoadedStream(
        name=stream_name,
        records=records,
        parse_errors=errs,
        source=f"ref:{ref_path}",
    )


def _find_episode_metadata_files(dataset_path: Path) -> list[Path]:
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


def _data_tier(streams: dict[str, LoadedStream]) -> str:
    has_rgb = len(streams.get("rgb", LoadedStream("rgb", [], [], "missing")).records) > 0
    has_pc = (
        len(
            streams.get(
                "planner_candidates", LoadedStream("planner_candidates", [], [], "missing")
            ).records
        )
        > 0
    )
    has_odom = len(streams.get("odom", LoadedStream("odom", [], [], "missing")).records) > 0
    if has_rgb and has_pc and has_odom:
        return "P0_full"
    if has_rgb and has_pc and not has_odom:
        return "P1_planner_only"
    return "unknown_or_partial"


def _subsample_times(times: list[float], *, stride_s: float | None, max_n: int | None) -> list[int]:
    """Return indices to keep from a sorted times list (deterministic)."""

    if not times:
        return []

    keep: list[int] = []
    if stride_s is None:
        keep = list(range(len(times)))
    else:
        next_t = times[0]
        for i, t in enumerate(times):
            if t + 1e-12 >= next_t:
                keep.append(i)
                next_t = t + stride_s

    if max_n is not None:
        keep = keep[:max_n]
    return keep


def _task_summary_from_counters(total: int, valid: int, reasons: Counter[str]) -> dict[str, Any]:
    invalid = total - valid
    return {
        "snapshots_total": total,
        "snapshots_valid": valid,
        "snapshots_invalid": invalid,
        "invalid_reasons": {k: reasons[k] for k in sorted(reasons) if reasons[k] > 0},
    }


def _validate_trajectory_selection(pc_records: list[PlannerCandidatesRecord]) -> dict[str, Any]:
    total = len(pc_records)
    valid = 0
    reasons: Counter[str] = Counter()
    for r in pc_records:
        if not r.candidates:
            reasons["planner_candidates_empty"] += 1
            continue
        if any((not math.isfinite(c.score)) for c in r.candidates):
            reasons["non_finite_score"] += 1
            continue
        valid += 1
    return _task_summary_from_counters(total, valid, reasons)


def validate_dataset(
    dataset_path: Path, config: DatasetValidationConfig | None = None
) -> dict[str, Any]:
    """Validate a dataset directory of canonical episodes and emit a completeness report."""

    config = config or DatasetValidationConfig()
    dataset_root = dataset_path.resolve()

    episode_files = _find_episode_metadata_files(dataset_root)
    if config.max_episodes is not None:
        episode_files = episode_files[: config.max_episodes]

    # Dataset-level aggregates.
    missing_stream_counts: Counter[str] = Counter()
    unreadable_stream_counts: Counter[str] = Counter()
    schema_version_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()

    task_totals: dict[TaskName, Counter[str]] = {"trajectory_selection": Counter()}
    task_valid_counts: dict[TaskName, int] = defaultdict(int)
    task_total_counts: dict[TaskName, int] = defaultdict(int)

    episodes_out: list[dict[str, Any]] = []

    for ep_path in episode_files:
        ep_dir = ep_path.parent
        episode_id_guess = ep_dir.name if ep_path.name == "episode.json" else ep_path.stem

        ep_errors: list[str] = []
        ep: CanonicalEpisode | None = None
        try:
            ep_obj = read_json(ep_path)
            ep = CanonicalEpisode.model_validate(ep_obj)
            schema_version_counts[ep.schema_version] += 1
        except Exception as e:  # noqa: BLE001 (report robustness)
            ep_errors.append(f"Failed to parse episode metadata {ep_path}: {e}")

        if ep is None:
            # If the metadata is unreadable, count all known streams as missing for this episode.
            for s in KNOWN_STREAMS:
                missing_stream_counts[s] += 1
            if config.include_episode_details:
                episodes_out.append(
                    {
                        "episode_id": episode_id_guess,
                        "episode_path": str(ep_path),
                        "schema_valid": False,
                        "schema_errors": ep_errors,
                        "streams_missing": list(KNOWN_STREAMS),
                        "streams_unreadable": [],
                        "tasks": {},
                    }
                )
            continue

        streams_loaded: dict[str, LoadedStream] = {}
        streams_missing: list[str] = []
        streams_unreadable: list[str] = []
        stream_record_counts: dict[str, int] = {}
        stream_parse_error_counts: dict[str, int] = {}

        for s in KNOWN_STREAMS:
            spec = ep.streams.get(s)
            if spec is None:
                missing_stream_counts[s] += 1
                streams_missing.append(s)
                streams_loaded[s] = LoadedStream(
                    name=s, records=[], parse_errors=[], source="missing"
                )
                continue

            loaded = _load_stream(s, spec, episode_dir=ep_dir, dataset_root=dataset_root)
            streams_loaded[s] = loaded
            stream_record_counts[s] = len(loaded.records)
            stream_parse_error_counts[s] = len(loaded.parse_errors)

            if loaded.source == "missing":
                missing_stream_counts[s] += 1
                streams_missing.append(s)
            elif loaded.parse_errors:
                unreadable_stream_counts[s] += 1
                streams_unreadable.append(s)

        tier = _data_tier(streams_loaded)
        tier_counts[tier] += 1

        # Build indexes (even if partially invalid; empty indexes handled by task validators).
        pc_index = TimeIndex.from_records(streams_loaded["planner_candidates"].records)

        # Subsample anchors deterministically.
        pc_times = pc_index.times
        pc_keep = _subsample_times(
            pc_times,
            stride_s=config.snapshot_stride_s,
            max_n=config.max_snapshots_per_episode,
        )

        pc_records = [pc_index.records[i] for i in pc_keep if i < len(pc_index.records)]

        # Narrow to correct types (unknown stream parsing could produce _DictRecord).
        pc_records_typed = [r for r in pc_records if isinstance(r, PlannerCandidatesRecord)]

        # Task summaries (per episode).
        tasks_ep: dict[TaskName, dict[str, Any]] = {}

        tasks_ep["trajectory_selection"] = _validate_trajectory_selection(pc_records_typed)

        # Accumulate dataset-level task summaries.
        for task_name, summary in tasks_ep.items():
            task_total_counts[task_name] += int(summary["snapshots_total"])
            task_valid_counts[task_name] += int(summary["snapshots_valid"])
            for reason, count in summary["invalid_reasons"].items():
                task_totals[task_name][reason] += int(count)

        if config.include_episode_details:
            episodes_out.append(
                {
                    "episode_id": ep.episode_id,
                    "episode_path": str(ep_path),
                    "schema_valid": True,
                    "schema_version": ep.schema_version,
                    "schema_version_expected": CANONICAL_EPISODE_SCHEMA_VERSION,
                    "data_tier": tier,
                    "streams_missing": sorted(streams_missing),
                    "streams_unreadable": sorted(streams_unreadable),
                    "stream_record_counts": {
                        k: stream_record_counts.get(k, 0) for k in KNOWN_STREAMS
                    },
                    "stream_parse_error_counts": {
                        k: stream_parse_error_counts.get(k, 0) for k in KNOWN_STREAMS
                    },
                    "tasks": tasks_ep,
                    "declared_missing_streams": sorted(
                        ep.quality.missing_streams if ep.quality else []
                    ),
                }
            )

    # Final dataset report (fully deterministic ordering).
    report: dict[str, Any] = {
        "canonical_episode_schema_version_expected": CANONICAL_EPISODE_SCHEMA_VERSION,
        "dataset_path": str(dataset_root),
        "episode_count": len(episode_files),
        "schema_version_counts": {
            k: schema_version_counts[k] for k in sorted(schema_version_counts)
        },
        "data_tier_counts": {k: tier_counts[k] for k in sorted(tier_counts)},
        "missing_stream_counts": {k: missing_stream_counts[k] for k in KNOWN_STREAMS},
        "unreadable_stream_counts": {k: unreadable_stream_counts[k] for k in KNOWN_STREAMS},
        "tasks": {},
    }

    for task_name in sorted(task_totals.keys()):
        total = task_total_counts[task_name]
        valid = task_valid_counts[task_name]
        reasons = task_totals[task_name]
        report["tasks"][task_name] = _task_summary_from_counters(total, valid, reasons)

    if config.include_episode_details:
        # Sort by episode_id first, then by path for stable output when ids collide.
        episodes_out.sort(key=lambda e: (e.get("episode_id", ""), e.get("episode_path", "")))
        report["episodes"] = episodes_out

    return report
