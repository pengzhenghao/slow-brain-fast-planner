from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from slow_brain_fast_planner import constants
from slow_brain_fast_planner.schema.canonical_episode import PlannerCandidatesRecord

A0GtRule = Literal[
    "raw_argmax",
    "planner_v1_nms_softmax_argmax",
    "planner_v2_nms_softmax_threshold_argmax",
]
A0Policy = Literal["raw_argmax", "random"]


@dataclass(frozen=True)
class A0Config:
    gt_rule: A0GtRule = "raw_argmax"
    policy: A0Policy = "raw_argmax"
    nms_max_trajectories: int = 6
    nms_distance_threshold: float = 2.0
    prob_threshold: float = 0.01  # used by planner_v2_* rule

    def __post_init__(self) -> None:
        if self.nms_max_trajectories <= 0:
            raise ValueError("nms_max_trajectories must be > 0")
        if self.nms_distance_threshold < 0:
            raise ValueError("nms_distance_threshold must be >= 0")
        if not (0.0 <= self.prob_threshold <= 1.0):
            raise ValueError("prob_threshold must be in [0,1]")


def subsample_by_time_stride(times: list[float], *, stride_s: float | None) -> list[int]:
    """Deterministically subsample indices from a sorted times list."""

    if not times:
        return []
    if stride_s is None:
        return list(range(len(times)))
    if stride_s <= 0:
        raise ValueError("stride_s must be > 0")

    keep: list[int] = []
    next_t = times[0]
    for i, t in enumerate(times):
        if t + 1e-12 >= next_t:
            keep.append(i)
            next_t = t + stride_s
    return keep


def _scores_and_endpoints(
    record: PlannerCandidatesRecord,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if not record.candidates:
        return None, None, "empty_candidates"

    scores = np.asarray([float(c.score) for c in record.candidates], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        return None, None, "non_finite_scores"

    endpoints = []
    for c in record.candidates:
        if not c.points_xy:
            return None, None, "empty_points_xy"
        xy = c.points_xy[-1]
        if len(xy) != 2:
            return None, None, "bad_endpoint_shape"
        endpoints.append([float(xy[0]), float(xy[1])])
    endpoints_arr = np.asarray(endpoints, dtype=np.float64)
    if endpoints_arr.ndim != 2 or endpoints_arr.shape[1] != 2:
        return None, None, "bad_endpoints_array"

    return scores, endpoints_arr, None


def _trajectory_nms_endpoints(
    *,
    scores: np.ndarray,
    endpoints_xy: np.ndarray,
    max_trajectories: int,
    distance_threshold: float,
) -> np.ndarray:
    """Deterministic NMS based on endpoint distance (mirrors planner script)."""

    K = int(scores.shape[0])
    if K == 0:
        return np.asarray([], dtype=np.int64)

    # Deterministic tie-break: higher score first, then lower index.
    sorted_indices = sorted(range(K), key=lambda i: (-float(scores[i]), int(i)))

    keep: list[int] = []
    for i in sorted_indices:
        if len(keep) >= max_trajectories:
            break
        should_keep = True
        for kept_idx in keep:
            dx = float(endpoints_xy[i, 0] - endpoints_xy[kept_idx, 0])
            dy = float(endpoints_xy[i, 1] - endpoints_xy[kept_idx, 1])
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < distance_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(int(i))

    if not keep:
        keep = [int(sorted_indices[0])]

    while len(keep) < max_trajectories:
        keep.append(int(keep[-1]))

    return np.asarray(keep[:max_trajectories], dtype=np.int64)


def _softmax_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    if s <= 0 or not np.isfinite(s):
        return np.full_like(ex, 1.0 / float(ex.size))
    return ex / s


def planner_postprocess_v2(
    record: PlannerCandidatesRecord,
    *,
    max_trajectories: int = constants.NMS_MAX_TRAJECTORIES,
    distance_threshold: float = constants.NMS_DISTANCE_THRESHOLD,
    prob_threshold: float = constants.PROB_THRESHOLD,
    allowed_indices: list[int] | None = None,
) -> tuple[list[int] | None, list[float] | None, str | None]:
    """Planner-style candidate postprocess:
    1) NMS (endpoint-distance) to keep `max_trajectories`
    2) normalize via softmax (stable)
    3) filter out candidates with normalized prob < prob_threshold

    If `allowed_indices` is provided, it is treated as a **precondition mask** applied
    *before* NMS/softmax/thresholding:
      - NMS is run only on the subset of candidates in `allowed_indices`
      - Returned indices are always in the original record.candidates index space

    Returns:
      kept_indices_filtered, probs_filtered, error
    """

    scores, endpoints, err = _scores_and_endpoints(record)
    if err is not None:
        return None, None, err
    assert scores is not None and endpoints is not None

    if allowed_indices is not None:
        K = int(scores.shape[0])
        allowed: list[int] = []
        seen: set[int] = set()
        for ii in allowed_indices:
            try:
                i = int(ii)
            except Exception:
                continue
            if i < 0 or i >= K:
                continue
            if i in seen:
                continue
            seen.add(i)
            allowed.append(i)
        if not allowed:
            return [], [], "empty_allowed_indices"

        keep_sub = _trajectory_nms_endpoints(
            scores=scores[np.asarray(allowed, dtype=np.int64)],
            endpoints_xy=endpoints[np.asarray(allowed, dtype=np.int64)],
            max_trajectories=int(max_trajectories),
            distance_threshold=float(distance_threshold),
        )
        keep = np.asarray([int(allowed[int(j)]) for j in keep_sub], dtype=np.int64)
    else:
        keep = _trajectory_nms_endpoints(
            scores=scores,
            endpoints_xy=endpoints,
            max_trajectories=int(max_trajectories),
            distance_threshold=float(distance_threshold),
        )
    if keep.size == 0:
        return None, None, "nms_no_candidates"

    probs = _softmax_stable(scores[keep])
    filtered = [
        (int(keep[i]), float(probs[i]))
        for i in range(len(keep))
        if float(probs[i]) >= prob_threshold
    ]
    if not filtered:
        return [], [], "all_below_prob_threshold"

    # Sort by probability (descending), stable tie-break by index.
    filtered.sort(key=lambda x: (-x[1], x[0]))
    return [i for i, _ in filtered], [p for _, p in filtered], None


def a0_label_index(record: PlannerCandidatesRecord, cfg: A0Config) -> tuple[int | None, str | None]:
    scores, endpoints, err = _scores_and_endpoints(record)
    if err is not None:
        return None, err

    assert scores is not None and endpoints is not None

    if cfg.gt_rule == "raw_argmax":
        return int(np.argmax(scores)), None

    if cfg.gt_rule == "planner_v1_nms_softmax_argmax":
        # Planner script:
        #   keep = trajectory_nms(...)
        #   scores_kept = scores[keep]
        #   argmax(softmax(scores_kept)) == argmax(scores_kept)
        keep = _trajectory_nms_endpoints(
            scores=scores,
            endpoints_xy=endpoints,
            max_trajectories=cfg.nms_max_trajectories,
            distance_threshold=cfg.nms_distance_threshold,
        )
        if keep.size == 0:
            return None, "nms_no_candidates"
        scores_kept = scores[keep]
        chosen = int(keep[int(np.argmax(scores_kept))])
        return chosen, None

    if cfg.gt_rule == "planner_v2_nms_softmax_threshold_argmax":
        kept, probs, err2 = planner_postprocess_v2(
            record,
            max_trajectories=cfg.nms_max_trajectories,
            distance_threshold=cfg.nms_distance_threshold,
            prob_threshold=cfg.prob_threshold,
        )
        if kept is None:
            return None, err2
        if kept:
            # Choose argmax among filtered probs.
            assert probs is not None
            chosen = int(kept[int(np.argmax(np.asarray(probs, dtype=np.float64)))])
            return chosen, None
        # If everything is filtered out, fall back to argmax among NMS-kept.
        scores2, endpoints2, err3 = _scores_and_endpoints(record)
        if err3 is not None or scores2 is None or endpoints2 is None:
            return None, err3 or "postprocess_failed"
        keep2 = _trajectory_nms_endpoints(
            scores=scores2,
            endpoints_xy=endpoints2,
            max_trajectories=cfg.nms_max_trajectories,
            distance_threshold=cfg.nms_distance_threshold,
        )
        if keep2.size == 0:
            return None, "nms_no_candidates"
        chosen2 = int(keep2[int(np.argmax(scores2[keep2]))])
        return chosen2, "all_below_prob_threshold_fallback_argmax"

    return None, f"unknown_gt_rule:{cfg.gt_rule}"


def a0_predict_index(
    record: PlannerCandidatesRecord,
    cfg: A0Config,
    rng: np.random.Generator,
) -> tuple[int | None, str | None]:
    scores, endpoints, err = _scores_and_endpoints(record)
    if err is not None:
        return None, err

    assert scores is not None and endpoints is not None
    K = int(scores.shape[0])

    if cfg.policy == "raw_argmax":
        return int(np.argmax(scores)), None
    if cfg.policy == "random":
        return int(rng.integers(0, K)), None
    return None, f"unknown_policy:{cfg.policy}"


def a0_topk_indices(record: PlannerCandidatesRecord, k: int) -> tuple[list[int] | None, str | None]:
    scores, _, err = _scores_and_endpoints(record)
    if err is not None:
        return None, err
    assert scores is not None
    K = int(scores.shape[0])
    kk = min(int(k), K)
    if kk <= 0:
        return [], None

    # Deterministic tie-break: higher score first, then lower index.
    ranked = sorted(range(K), key=lambda i: (-float(scores[i]), int(i)))
    return [int(i) for i in ranked[:kk]], None


def safe_div(n: int, d: int) -> float | None:
    if d == 0:
        return None
    return float(n) / float(d)


def to_jsonable(obj: Any) -> Any:
    """Convert common non-JSON types to JSONable values."""

    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    return obj
