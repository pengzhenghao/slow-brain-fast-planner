#!/usr/bin/env python
"""Build a *static* trajectory-selection candidate set by clustering raw planner trajectories.

This creates a JSON file containing K prototype trajectories (in robot-frame XY),
intended to be used as a baseline candidate set (VLM selects among K).

Default source: prelogged `episodes/*/planner_candidates.jsonl` candidates.
Optionally restrict to takeover-clip t0 times to better match the evaluation distribution.

We support multiple construction methods:
- kmeans_medoids: k-means in full-trajectory space, then choose medoids (existing)
- kdisk_endpoints_medoids: "k-disk / covering" style farthest-first over endpoints (k-center),
  then choose representative trajectories per endpoint cluster (better geometric coverage)
- grid_endpoints_bezier: deterministic grid of endpoints; build smooth Bezier trajectories to each
endpoint
- grid_endpoints_scurve: deterministic grid of endpoints; build smooth S-curve (smoothstep)
trajectories
- grid_endpoints_nearest_real: deterministic endpoint grid; select nearest *real* logged
trajectories per grid point
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _portable_provenance_path(path: Path, *, dataset_root: Path) -> str:
    path = path.resolve()
    dataset_root = dataset_root.resolve()
    try:
        relative = path.relative_to(dataset_root)
    except ValueError:
        return path.name
    return str(Path(dataset_root.name) / relative)


def _resample_polyline_xy(points_xy: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to exactly n points using linear interpolation on index."""
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 1 or pts.shape[1] < 2:
        raise ValueError("points_xy must be (T,2+)")
    if int(n) <= 0:
        raise ValueError("n must be > 0")
    if int(pts.shape[0]) == int(n):
        return pts[:, :2].copy()
    t_src = np.linspace(0.0, 1.0, int(pts.shape[0]), dtype=np.float64)
    t_dst = np.linspace(0.0, 1.0, int(n), dtype=np.float64)
    x = np.interp(t_dst, t_src, pts[:, 0])
    y = np.interp(t_dst, t_src, pts[:, 1])
    return np.stack([x, y], axis=1)


def _kmeanspp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return indices of k initial centers using k-means++."""
    n = int(X.shape[0])
    if n <= 0:
        raise ValueError("X is empty")
    if not (1 <= int(k) <= n):
        raise ValueError("k must be in [1, n]")

    centers = np.empty((int(k),), dtype=np.int64)
    centers[0] = int(rng.integers(0, n))
    # d2: squared distance to nearest chosen center
    c0 = X[centers[0]]
    d2 = np.sum((X - c0) ** 2, axis=1)
    d2 = np.maximum(d2, 0.0)

    for i in range(1, int(k)):
        s = float(np.sum(d2))
        if not math.isfinite(s) or s <= 1e-12:
            # All points identical (or numerical issues). Pick random.
            centers[i] = int(rng.integers(0, n))
            continue
        probs = d2 / s
        centers[i] = int(rng.choice(n, p=probs))
        ci = X[centers[i]]
        d2 = np.minimum(d2, np.sum((X - ci) ** 2, axis=1))
        d2 = np.maximum(d2, 0.0)
    return centers


@dataclass(frozen=True)
class KMeansResult:
    centers: np.ndarray  # (k, d)
    assign: np.ndarray  # (n,)


def _kmeans_lloyd(
    X: np.ndarray, k: int, *, seed: int = 0, max_iter: int = 40, tol: float = 1e-5
) -> KMeansResult:
    """Simple k-means (Lloyd) with k-means++ init. Returns centers and assignments."""
    X = np.asarray(X, dtype=np.float64)
    n, d = int(X.shape[0]), int(X.shape[1])
    if n <= 0 or d <= 0:
        raise ValueError("X must be (n,d) with n,d>0")
    if not (1 <= int(k) <= n):
        raise ValueError("k must be in [1, n]")

    rng = np.random.default_rng(int(seed))
    init_idx = _kmeanspp_init(X, int(k), rng)
    C = X[init_idx].copy()

    x2 = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
    prev_inertia = None
    assign = np.zeros((n,), dtype=np.int64)

    for _it in range(int(max_iter)):
        c2 = np.sum(C * C, axis=1, keepdims=True).T  # (1,k)
        # d2 = ||x||^2 + ||c||^2 - 2 x·c
        d2 = x2 + c2 - 2.0 * (X @ C.T)  # (n,k)
        assign = np.argmin(d2, axis=1).astype(np.int64)
        inertia = float(np.sum(d2[np.arange(n), assign]))

        # Recompute centers
        C_new = np.zeros_like(C)
        counts = np.zeros((int(k),), dtype=np.int64)
        for j in range(int(k)):
            mask = assign == j
            cnt = int(np.sum(mask))
            counts[j] = cnt
            if cnt > 0:
                C_new[j] = np.mean(X[mask], axis=0)
            else:
                # Empty cluster: re-seed to a random point (stable).
                C_new[j] = X[int(rng.integers(0, n))]

        # Convergence check
        shift = float(np.sqrt(np.mean((C_new - C) ** 2)))
        C = C_new
        if prev_inertia is not None:
            rel = abs(prev_inertia - inertia) / max(1e-9, abs(prev_inertia))
            if rel < float(tol) or shift < float(tol):
                break
        prev_inertia = inertia

    return KMeansResult(centers=C, assign=assign)


def _choose_medoids(X: np.ndarray, assign: np.ndarray, centers: np.ndarray) -> list[int]:
    """Choose one medoid (actual datapoint) per cluster, closest to the center."""
    X = np.asarray(X, dtype=np.float64)
    assign = np.asarray(assign, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.float64)
    k = int(centers.shape[0])
    out: list[int] = []
    for j in range(k):
        idxs = np.where(assign == j)[0]
        if idxs.size == 0:
            # Fallback: closest overall point to center.
            d2 = np.sum((X - centers[j]) ** 2, axis=1)
            out.append(int(np.argmin(d2)))
            continue
        d2 = np.sum((X[idxs] - centers[j]) ** 2, axis=1)
        out.append(int(idxs[int(np.argmin(d2))]))
    return out


def _kcenter_farthest_first(X: np.ndarray, k: int, *, seed: int = 0) -> list[int]:
    """Greedy k-center / farthest-first traversal (a simple 'covering' baseline).

    This is a reasonable approximation of the "k-disk" covering intuition: iteratively add the point
    farthest from the current set of centers.
    """
    X = np.asarray(X, dtype=np.float64)
    n = int(X.shape[0])
    if n <= 0:
        raise ValueError("X is empty")
    if not (1 <= int(k) <= n):
        raise ValueError("k must be in [1, n]")

    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(0, n))
    centers: list[int] = [first]
    # d2 to nearest center
    d2 = np.sum((X - X[first]) ** 2, axis=1)
    d2 = np.maximum(d2, 0.0)
    for _ in range(1, int(k)):
        i = int(np.argmax(d2))
        centers.append(i)
        # update distances
        d2 = np.minimum(d2, np.sum((X - X[i]) ** 2, axis=1))
        d2 = np.maximum(d2, 0.0)
    return centers


def _assign_to_centers(X: np.ndarray, centers_idx: list[int]) -> np.ndarray:
    """Return cluster assignments for each X row to nearest center (by Euclidean distance)."""
    X = np.asarray(X, dtype=np.float64)
    C = X[np.asarray(centers_idx, dtype=np.int64)]
    x2 = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
    c2 = np.sum(C * C, axis=1, keepdims=True).T  # (1,k)
    d2 = x2 + c2 - 2.0 * (X @ C.T)
    return np.argmin(d2, axis=1).astype(np.int64)


def _bezier_curve(
    P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, P3: np.ndarray, n: int
) -> np.ndarray:
    """Cubic Bezier, n points, inclusive of endpoints."""
    t = np.linspace(0.0, 1.0, int(n), dtype=np.float64).reshape(-1, 1)
    omt = 1.0 - t
    pts = (omt**3) * P0 + 3.0 * (omt**2) * t * P1 + 3.0 * omt * (t**2) * P2 + (t**3) * P3
    return pts.astype(np.float64)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Smoothstep 3t^2 - 2t^3 with t clamped to [0,1]."""
    tt = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return tt * tt * (3.0 - 2.0 * tt)


def _scurve_xy_to_endpoint(x_end: float, y_end: float, n: int) -> np.ndarray:
    """Monotonic-x S-curve to endpoint using smoothstep in y(x)."""
    xe = float(x_end)
    if not math.isfinite(xe) or xe <= 1e-6:
        xe = 1.0
    x = np.linspace(0.0, xe, int(n), dtype=np.float64)
    t = x / xe
    s = _smoothstep(t)
    y = float(y_end) * s
    return np.stack([x, y], axis=1)


def _parse_csv_floats(s: str | None) -> list[float] | None:
    if s is None:
        return None
    out: list[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out if out else None


def _load_takeover_times(takeover_jsonl: Path) -> dict[str, set[float]]:
    """Map episode_id -> set of t0 times (rounded to 1e-6)."""
    out: dict[str, set[float]] = {}
    with takeover_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            eid = obj.get("episode_id")
            t0 = obj.get("t0")
            if not isinstance(eid, str) or not isinstance(t0, (int, float)):
                continue
            out.setdefault(str(eid), set()).add(round(float(t0), 6))
    return out


def _iter_candidate_trajs_from_dataset(
    dataset_root: Path,
    *,
    restrict_to_times: dict[str, set[float]] | None,
) -> Iterable[list[list[float]]]:
    episodes_dir = (dataset_root / "episodes").resolve()
    for ep_dir in sorted(episodes_dir.glob("*")):
        if not ep_dir.is_dir():
            continue
        eid = ep_dir.name
        pc_path = (ep_dir / "planner_candidates.jsonl").resolve()
        if not pc_path.exists():
            continue
        allowed_t = restrict_to_times.get(eid) if restrict_to_times is not None else None
        try:
            with pc_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    if allowed_t is not None:
                        t = rec.get("t")
                        if not isinstance(t, (int, float)) or round(float(t), 6) not in allowed_t:
                            continue
                    cands = rec.get("candidates")
                    if not isinstance(cands, list) or not cands:
                        continue
                    for c in cands:
                        if not isinstance(c, dict):
                            continue
                        pts = c.get("points_xy")
                        if isinstance(pts, list) and len(pts) >= 2:
                            yield pts
        except Exception:
            continue


def _reservoir_sample(
    it: Iterable[list[list[float]]],
    *,
    max_samples: int,
    seed: int,
) -> list[list[list[float]]]:
    rng = random.Random(int(seed))
    sample: list[list[list[float]]] = []
    seen = 0
    for traj in it:
        seen += 1
        if len(sample) < int(max_samples):
            sample.append(traj)
        else:
            j = rng.randint(0, seen - 1)
            if j < int(max_samples):
                sample[j] = traj
    return sample


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build static Task2 candidate set via clustering.")
    p.add_argument(
        "--dataset",
        required=True,
        help="Processed dataset root (contains episodes/*/planner_candidates.jsonl).",
    )
    p.add_argument(
        "--out", required=True, help="Output JSON path OR output directory (if --ks has multiple)."
    )
    p.add_argument(
        "--ks",
        default="6,12,18,24",
        help="Comma-separated K values to build (default: 6,12,18,24).",
    )
    p.add_argument(
        "--method",
        default="kmeans_medoids",
        choices=[
            "kmeans_medoids",
            "kdisk_endpoints_medoids",
            "grid_endpoints_bezier",
            "grid_endpoints_scurve",
            "grid_endpoints_nearest_real",
        ],
        help=(
            "How to construct the static candidate set.\n"
            "  kmeans_medoids: k-means in trajectory space, choose medoids\n"
            "  kdisk_endpoints_medoids: farthest-first (k-center) in endpoint space, choose "
            "representative traj per cluster\n"
            "  grid_endpoints_bezier: deterministic endpoint grid + smooth Bezier trajectories\n"
            "  grid_endpoints_scurve: deterministic endpoint grid + smoothstep S-curve "
            "trajectories\n"
            "  grid_endpoints_nearest_real: deterministic endpoint grid + nearest real logged "
            "trajectories"
        ),
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=120_000,
        help="Max number of trajectories to sample for clustering.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-points", type=int, default=20, help="Trajectory waypoint count (resampled if needed)."
    )
    p.add_argument(
        "--grid-x",
        default="1.8,2.8,3.8",
        help="Grid X (forward) coordinates for grid_* methods (comma-separated). Default: "
        "1.8,2.8,3.8",
    )
    p.add_argument(
        "--grid-y",
        default="-1.0,-0.4,0.4,1.0",
        help="Grid Y (left) coordinates for grid_* methods (comma-separated). Default: "
        "-1.0,-0.4,0.4,1.0",
    )
    p.add_argument(
        "--takeover-clips",
        default=None,
        help="Optional takeover_clips.jsonl path; if provided, only cluster candidates at clip t0 "
        "times.",
    )
    args = p.parse_args(argv)

    dataset_root = Path(args.dataset).resolve()
    out_path = Path(args.out).resolve()
    ks = [int(x) for x in str(args.ks).split(",") if str(x).strip() != ""]
    if not ks:
        raise SystemExit("--ks is empty")
    method = str(args.method)

    restrict = (
        _load_takeover_times(Path(args.takeover_clips).resolve()) if args.takeover_clips else None
    )
    n_pts = int(args.n_points)
    raw: list[list[list[float]]] = []
    T: np.ndarray | None = None
    X: np.ndarray | None = None
    endpoints: np.ndarray | None = None

    if method not in ("grid_endpoints_bezier", "grid_endpoints_scurve"):
        it = _iter_candidate_trajs_from_dataset(dataset_root, restrict_to_times=restrict)
        raw = _reservoir_sample(it, max_samples=int(args.max_samples), seed=int(args.seed))
        if not raw:
            raise SystemExit(
                "No trajectories found to cluster (check dataset path and optional takeover "
                "filter)."
            )

        trajs = []
        ends = []
        for pts in raw:
            try:
                arr = np.asarray(pts, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
                    continue
                arr2 = _resample_polyline_xy(arr[:, :2], n_pts)
                if not np.all(np.isfinite(arr2)):
                    continue
                trajs.append(arr2.astype(np.float32))
                ends.append(arr2[-1, :2].astype(np.float64))
            except Exception:
                continue
        if not trajs:
            raise SystemExit("All sampled trajectories were invalid after resampling.")

        T = np.stack(trajs, axis=0)  # (N, n_pts, 2)
        X = T.reshape(T.shape[0], -1).astype(np.float64)  # (N, 2*n_pts)
        endpoints = np.stack(ends, axis=0).astype(np.float64)  # (N, 2)

    out_dir: Path
    multi = len(ks) > 1 or out_path.suffix.lower() != ".json"
    if multi:
        out_dir = out_path if out_path.suffix.lower() != ".json" else out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

    for k in ks:
        protos: list[dict[str, object]] = []

        if method in ("grid_endpoints_bezier", "grid_endpoints_scurve"):
            if int(k) != 12:
                raise SystemExit("grid_* methods currently support only K=12 (requested K!=12)")
            # Endpoint grid: 3 forward distances × 4 lateral offsets = 12.
            x_vals = _parse_csv_floats(str(args.grid_x)) or [1.8, 2.8, 3.8]
            y_vals = _parse_csv_floats(str(args.grid_y)) or [-1.0, -0.4, 0.4, 1.0]
            if len(x_vals) * len(y_vals) != 12:
                raise SystemExit(
                    f"--grid-x * --grid-y must yield 12 endpoints, got {len(x_vals)}*{len(y_vals)}"
                )
            endpoints_grid = [(float(x), float(y)) for x in x_vals for y in y_vals]
            assert len(endpoints_grid) == 12
            for x, y in endpoints_grid:
                if method == "grid_endpoints_bezier":
                    P0 = np.asarray([0.0, 0.0], dtype=np.float64)
                    P3 = np.asarray([float(x), float(y)], dtype=np.float64)
                    # Smooth, forward-biased curve: start tangent along +x, end tangent roughly
                    # toward endpoint.
                    P1 = np.asarray([max(0.2, float(x) * 0.33), 0.0], dtype=np.float64)
                    P2 = np.asarray([max(0.4, float(x) * 0.66), float(y)], dtype=np.float64)
                    pts = _bezier_curve(P0, P1, P2, P3, n_pts).astype(np.float32)
                else:
                    pts = _scurve_xy_to_endpoint(float(x), float(y), n_pts).astype(np.float32)
                protos.append(
                    {"points_xy": [[float(px), float(py)] for px, py in pts.tolist()], "score": 0.0}
                )

        elif method == "kmeans_medoids":
            assert X is not None and T is not None
            if int(k) > int(X.shape[0]):
                raise SystemExit(
                    f"K={k} > num_samples={int(X.shape[0])} (increase --max-samples or reduce K)"
                )
            km = _kmeans_lloyd(X, int(k), seed=int(args.seed), max_iter=40, tol=1e-5)
            medoids = _choose_medoids(X, km.assign, km.centers)
            for idx in medoids:
                pts = T[int(idx)].astype(np.float32)
                protos.append(
                    {"points_xy": [[float(x), float(y)] for x, y in pts.tolist()], "score": 0.0}
                )

        elif method == "kdisk_endpoints_medoids":
            # "K-disk" intuition: maximize geometric coverage in endpoint space, then pick a real
            # trajectory per cell.
            assert endpoints is not None and T is not None
            if int(k) > int(endpoints.shape[0]):
                raise SystemExit(
                    f"K={k} > num_samples={int(endpoints.shape[0])} "
                    "(increase --max-samples or reduce K)"
                )
            centers_idx = _kcenter_farthest_first(endpoints, int(k), seed=int(args.seed))
            assign = _assign_to_centers(endpoints, centers_idx)
            for j, c_i in enumerate(centers_idx):
                # Choose a representative trajectory: closest endpoint to the chosen center
                # endpoint (medoid-ish).
                # This keeps trajectories "realistic" (picked from data) but improves coverage vs
                # kmeans.
                idxs = np.where(assign == int(j))[0]
                if idxs.size == 0:
                    rep = int(c_i)
                else:
                    center_xy = endpoints[int(c_i)]
                    d2 = np.sum((endpoints[idxs] - center_xy.reshape(1, 2)) ** 2, axis=1)
                    rep = int(idxs[int(np.argmin(d2))])
                pts = T[int(rep)].astype(np.float32)
                protos.append(
                    {"points_xy": [[float(x), float(y)] for x, y in pts.tolist()], "score": 0.0}
                )

        elif method == "grid_endpoints_nearest_real":
            assert endpoints is not None and T is not None
            if int(k) != 12:
                raise SystemExit(
                    "grid_endpoints_nearest_real currently supports only K=12 (requested K!=12)"
                )
            x_vals = _parse_csv_floats(str(args.grid_x)) or [1.8, 2.8, 3.8]
            y_vals = _parse_csv_floats(str(args.grid_y)) or [-1.0, -0.4, 0.4, 1.0]
            if len(x_vals) * len(y_vals) != 12:
                raise SystemExit(
                    f"--grid-x * --grid-y must yield 12 endpoints, got {len(x_vals)}*{len(y_vals)}"
                )
            targets = [(float(x), float(y)) for x in x_vals for y in y_vals]
            # Choose unique real trajectories when possible.
            remaining: set[int] = set(range(int(endpoints.shape[0])))
            chosen: list[int] = []
            for tx, ty in targets:
                txy = np.asarray([float(tx), float(ty)], dtype=np.float64).reshape(1, 2)
                if remaining:
                    idxs = np.fromiter(remaining, dtype=np.int64)
                    d2 = np.sum((endpoints[idxs] - txy) ** 2, axis=1)
                    pick = int(idxs[int(np.argmin(d2))])
                    chosen.append(pick)
                    remaining.remove(pick)
                else:
                    d2 = np.sum((endpoints - txy) ** 2, axis=1)
                    chosen.append(int(np.argmin(d2)))
            for idx in chosen:
                pts = T[int(idx)].astype(np.float32)
                protos.append(
                    {"points_xy": [[float(x), float(y)] for x, y in pts.tolist()], "score": 0.0}
                )

        else:
            raise SystemExit(f"Unknown --method: {method}")

        payload = {
            "type": "task2_static_candidate_set",
            "method": str(method),
            "dataset": dataset_root.name,
            "takeover_clips": _portable_provenance_path(
                Path(args.takeover_clips),
                dataset_root=dataset_root,
            )
            if args.takeover_clips
            else None,
            "k": int(k),
            "n_points": int(n_pts),
            "num_samples_used": int(X.shape[0]) if X is not None else None,
            "seed": int(args.seed),
            "grid": (
                {"x": _parse_csv_floats(str(args.grid_x)), "y": _parse_csv_floats(str(args.grid_y))}
                if str(method).startswith("grid_")
                else None
            ),
            "candidates": protos,
        }

        if multi:
            dst = (out_dir / f"static_candidates_k{int(k)}.json").resolve()
        else:
            dst = out_path
        dst.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        samples_str = str(int(X.shape[0])) if X is not None else "n/a"
        print(f"[ok] wrote {dst} (k={k}, n_points={n_pts}, samples={samples_str})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
