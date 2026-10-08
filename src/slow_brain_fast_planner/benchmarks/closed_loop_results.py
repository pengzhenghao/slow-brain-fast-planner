from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiscoveredRun:
    """Metadata about a single closed-loop benchmark run directory."""

    run_dir: Path
    step: int
    step_name: str
    # Optional sweep keys inferred from folder structure
    delay_sweep: float | None = None
    vlm_query_hz_sweep: float | None = None
    vlm_query_interval_s_sweep: float | None = None
    # Extra context (e.g. step4 subfolder like baselines/score_fusion/...)
    group: str | None = None


def resolve_section5_root(
    candidates: Iterable[str | Path] | None = None,
) -> Path:
    """Resolve the `logs/section5_closed_loop` root, trying common repo-local locations."""
    fallback: list[Path] = [
        Path("logs/section5_closed_loop").resolve(),
    ]

    tried: list[Path] = []

    if candidates is not None:
        for c in candidates:
            p = (
                Path(c).expanduser().resolve()
                if not isinstance(c, Path)
                else c.expanduser().resolve()
            )
            tried.append(p)
            if p.exists():
                return p

    for p in fallback:
        p = p.expanduser().resolve()
        tried.append(p)
        if p.exists():
            return p

    # Return a helpful default for error messages.
    if tried:
        return tried[0]
    return fallback[0].expanduser().resolve()


def find_latest_run(parent_dir: Path, *, recursive: bool = True) -> Path | None:
    """Find the most recent `closed_loop_fusion_YYYYMMDD_HHMMSS` directory under parent."""
    parent_dir = Path(parent_dir)
    if not parent_dir.exists():
        return None
    if recursive:
        runs = sorted(
            [
                p
                for p in parent_dir.rglob("closed_loop_fusion_*")
                if p.is_dir() and p.name.startswith("closed_loop_fusion_")
            ]
        )
    else:
        runs = sorted(
            [
                p
                for p in parent_dir.iterdir()
                if p.is_dir() and p.name.startswith("closed_loop_fusion_")
            ]
        )
    return runs[-1] if runs else None


def _safe_float(x: object) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return float(v)
    except Exception:
        return None


def _get(d: object, path: list[str], default: object = None) -> object:
    cur: object = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def policy_is_stream_query(policy: str) -> bool:
    return str(policy) in (
        "vlm_stream",
        "vlm_stream_match",
        "score_fusion_stream",
        "prob_fusion_stream",
    )


def policy_is_max_rate(policy: str) -> bool:
    # "Max-rate" here means "attempt to request at every plan tick" (subject to inflight limits).
    # Stream-query policies use a fixed cadence instead.
    return str(policy) in ("vlm_hold", "vlm_hold_match", "score_fusion", "prob_fusion")


def load_closed_loop_run(
    run_dir: str | Path,
    *,
    prefer: Literal["results", "summary"] = "results",
    extra_cols: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load a run directory into a DataFrame and config dict.

    - Reads `config.json`
    - Reads `results_augmented.csv` or `results.csv`
    - Optionally reads `summary.csv`
    - Injects run-level columns:
      - `vlm_query_hz_config`, `dt_plan_s`, `dt_control_s`
      - `vlm_request_hz_effective` (per-row derived)
      - `run_dir` (string)
    """
    run_dir_p = Path(run_dir).expanduser().resolve()
    if not run_dir_p.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir_p}")

    cfg_path = run_dir_p / "config.json"
    cfg: dict = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Choose which table to return.
    df_results: pd.DataFrame | None = None
    df_summary: pd.DataFrame | None = None

    results_path = None
    for fn in ("results_augmented.csv", "results.csv"):
        p = run_dir_p / fn
        if p.exists():
            results_path = p
            break
    summary_path = run_dir_p / "summary.csv"
    if summary_path.exists():
        df_summary = pd.read_csv(summary_path)
    if results_path is not None:
        df_results = pd.read_csv(results_path)

    if prefer == "summary" and df_summary is not None:
        df = df_summary.copy()
    elif df_results is not None:
        df = df_results.copy()
    elif df_summary is not None:
        df = df_summary.copy()
    else:
        raise FileNotFoundError(
            f"No results.csv/results_augmented.csv/summary.csv found in {run_dir_p}"
        )

    # Inject run-level knobs.
    vlm_query_hz = _safe_float(_get(cfg, ["vlm_query_hz"], None))
    interval_cfg = _safe_float(_get(cfg, ["vlm_query_interval_s"], None))
    if interval_cfg is None and vlm_query_hz is not None and float(vlm_query_hz) > 1e-9:
        interval_cfg = 1.0 / float(vlm_query_hz)
    try:
        vlm_max_inflight = int(_get(cfg, ["vlm_max_inflight"], 0))  # 0/<=0 means "policy default"
    except Exception:
        vlm_max_inflight = 0
    dt_plan = _safe_float(_get(cfg, ["sim", "dt_plan"], None))
    dt_control = _safe_float(_get(cfg, ["sim", "dt_control"], None))

    df["run_dir"] = str(run_dir_p)
    df["vlm_query_hz_config"] = float(vlm_query_hz) if vlm_query_hz is not None else np.nan
    df["vlm_query_interval_s_config"] = float(interval_cfg) if interval_cfg is not None else np.nan
    df["vlm_max_inflight"] = int(vlm_max_inflight)
    df["dt_plan_s"] = float(dt_plan) if dt_plan is not None else np.nan
    df["dt_control_s"] = float(dt_control) if dt_control is not None else np.nan

    # Effective request rate (Hz) based on counters.
    if (
        "vlm_requests" in df.columns
        and "plan_ticks" in df.columns
        and np.isfinite(df["dt_plan_s"]).all()
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["vlm_request_hz_effective"] = df["vlm_requests"] / (
                df["plan_ticks"] * df["dt_plan_s"]
            )
    else:
        df["vlm_request_hz_effective"] = np.nan

    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v

    return df, cfg


_RE_DELAY = re.compile(r"^delay_(?P<delay>[-+]?\d+(?:\.\d+)?)$")
_RE_HZ = re.compile(r"^hz_(?P<hz>[-+]?\d+(?:\.\d+)?)$")
_RE_INTERVAL = re.compile(r"^interval_s_(?P<interval>[-+]?\d+(?:\.\d+)?)$")
_RE_DELAY_HZ = re.compile(r"^delay_(?P<delay>[-+]?\d+(?:\.\d+)?)_hz_(?P<hz>[-+]?\d+(?:\.\d+)?)$")
_RE_DELAY_INTERVAL = re.compile(
    r"^delay_(?P<delay>[-+]?\d+(?:\.\d+)?)_interval_s_(?P<interval>[-+]?\d+(?:\.\d+)?)$"
)


def discover_section5_runs(section5_root: str | Path) -> list[DiscoveredRun]:
    """Discover runs under `logs/section5_closed_loop` as laid out by
    `scripts/closed_loop_sim/section5_closed_loop_experiments.sh`."""
    root = Path(section5_root).expanduser().resolve()
    out: list[DiscoveredRun] = []

    # Step 1: step1_delay_basics/delay_<delay>/closed_loop_fusion_*
    step1 = root / "step1_delay_basics"
    if step1.exists():
        for ddir in sorted([p for p in step1.glob("delay_*") if p.is_dir()], key=lambda p: p.name):
            m = _RE_DELAY.match(ddir.name)
            delay = float(m.group("delay")) if m else None
            run_dir = find_latest_run(ddir, recursive=True)
            if run_dir is not None:
                out.append(
                    DiscoveredRun(
                        run_dir=run_dir,
                        step=1,
                        step_name="delay_basics",
                        delay_sweep=delay,
                    )
                )

    # Step 2:
    # - legacy: step2_vlm_query_rate_sweep/delay_<delay>/hz_<hz>/closed_loop_fusion_*
    # - preferred:
    # step2_vlm_query_interval_sweep/delay_<delay>/interval_s_<interval>/closed_loop_fusion_*
    step2 = root / "step2_vlm_query_interval_sweep"
    if not step2.exists():
        step2 = root / "step2_vlm_query_rate_sweep"
    if step2.exists():
        for ddir in sorted([p for p in step2.glob("delay_*") if p.is_dir()], key=lambda p: p.name):
            md = _RE_DELAY.match(ddir.name)
            delay = float(md.group("delay")) if md else None
            # Support both legacy hz_* and preferred interval_s_* folder naming.
            subdirs = sorted(
                [
                    p
                    for p in ddir.iterdir()
                    if p.is_dir() and (p.name.startswith("hz_") or p.name.startswith("interval_s_"))
                ],
                key=lambda p: p.name,
            )
            for sdir in subdirs:
                hz = None
                interval_s = None
                mhz = _RE_HZ.match(sdir.name)
                mint = _RE_INTERVAL.match(sdir.name)
                if mhz:
                    hz = float(mhz.group("hz"))
                    interval_s = (1.0 / hz) if hz > 1e-9 else None
                elif mint:
                    interval_s = float(mint.group("interval"))
                    hz = (1.0 / interval_s) if interval_s > 1e-9 else None
                run_dir = find_latest_run(sdir, recursive=True)
                if run_dir is not None:
                    out.append(
                        DiscoveredRun(
                            run_dir=run_dir,
                            step=2,
                            step_name="vlm_query_rate_sweep",
                            delay_sweep=delay,
                            vlm_query_hz_sweep=hz,
                            vlm_query_interval_s_sweep=interval_s,
                        )
                    )

    # Step 3:
    # - legacy: step3_lambda_sweep_fusions/delay_<delay>_hz_<hz>/closed_loop_fusion_*
    # - preferred (older):
    # step3_lambda_sweep_fusions/delay_<delay>_interval_s_<interval>/closed_loop_fusion_*
    # - current: step3_lambda_sweep_fusions/delay_<delay>/closed_loop_fusion_*
    step3 = root / "step3_lambda_sweep_fusions"
    if step3.exists():
        # Support legacy *_hz_*, preferred *_interval_s_*, and simple delay_*.
        dirs = sorted(
            [
                p
                for p in step3.iterdir()
                if p.is_dir()
                and (
                    ("_hz_" in p.name) or ("_interval_s_" in p.name) or p.name.startswith("delay_")
                )
            ],
            key=lambda p: p.name,
        )
        for dh in dirs:
            hz = None
            interval_s = None
            m_hz = _RE_DELAY_HZ.match(dh.name)
            m_int = _RE_DELAY_INTERVAL.match(dh.name)
            if m_hz:
                delay = float(m_hz.group("delay"))
                hz = float(m_hz.group("hz"))
                interval_s = (1.0 / hz) if hz > 1e-9 else None
            elif m_int:
                delay = float(m_int.group("delay"))
                interval_s = float(m_int.group("interval"))
                hz = (1.0 / interval_s) if interval_s > 1e-9 else None
            else:
                md = _RE_DELAY.match(dh.name)
                if not md:
                    continue
                delay = float(md.group("delay"))
            run_dir = find_latest_run(dh, recursive=True)
            if run_dir is not None:
                out.append(
                    DiscoveredRun(
                        run_dir=run_dir,
                        step=3,
                        step_name="lambda_sweep_fusions",
                        delay_sweep=delay,
                        vlm_query_hz_sweep=hz,
                        vlm_query_interval_s_sweep=interval_s,
                    )
                )

    # Step 4: step4_main_results/delay_<delay>/<policy_group>/closed_loop_fusion_*
    step4 = root / "step4_main_results"
    if step4.exists():
        for ddir in sorted([p for p in step4.glob("delay_*") if p.is_dir()], key=lambda p: p.name):
            md = _RE_DELAY.match(ddir.name)
            delay = float(md.group("delay")) if md else None
            for group in (
                "baselines",
                "score_fusion",
                "prob_fusion",
                "score_fusion_stream",
                "prob_fusion_stream",
            ):
                gdir = ddir / group
                run_dir = find_latest_run(gdir, recursive=True)
                if run_dir is not None:
                    out.append(
                        DiscoveredRun(
                            run_dir=run_dir,
                            step=4,
                            step_name="main_results",
                            delay_sweep=delay,
                            group=group,
                        )
                    )

    # -------------------------------------------------------------------------
    # Fallback discovery: "loose" runs
    # -------------------------------------------------------------------------
    # Only do this expensive recursive scan if we did not find any runs via the
    # standard step1/2/3/4 folder structure.
    if not out:
        # Some quick smoke scripts write:
        #   <root>/<name>/closed_loop_fusion_YYYYMMDD_HHMMSS/
        # without the step1/2/3/4 folder structure. We still want the notebook to
        # load them rather than erroring.
        #
        # We classify these as step=4 and use config.json to recover delay_s.
        for run_dir in sorted(
            [
                p
                for p in root.rglob("closed_loop_fusion_*")
                if p.is_dir() and p.name.startswith("closed_loop_fusion_")
            ]
        ):
            rd = run_dir.resolve()
            delay = None
            try:
                cfg_p = rd / "config.json"
                if cfg_p.exists():
                    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
                    delay = _safe_float(cfg.get("delay_s", None))
            except Exception:
                delay = None
            # Group name: relative parent folder under root (best-effort).
            try:
                rel_parent = rd.parent.relative_to(root)
                group = str(rel_parent) if str(rel_parent) not in (".", "") else None
            except Exception:
                group = rd.parent.name
            out.append(
                DiscoveredRun(
                    run_dir=rd,
                    step=4,
                    step_name="main_results",
                    delay_sweep=float(delay) if delay is not None else None,
                    group=group,
                )
            )

    return out


def load_section5_results_table(
    section5_root: str | Path,
    *,
    prefer: Literal["results", "summary"] = "results",
) -> pd.DataFrame:
    """Load all discovered section-5 runs into one tidy table (row-level).

    The returned dataframe includes:
    - `section5_step`, `section5_step_name`, `section5_delay_sweep`, `section5_vlm_query_hz_sweep`,
    `section5_group`
    - run-level injected columns from `load_closed_loop_run`
    """
    runs = discover_section5_runs(section5_root)
    if not runs:
        raise FileNotFoundError(
            f"No runs found under: {Path(section5_root).expanduser().resolve()}"
        )

    parts: list[pd.DataFrame] = []
    print(runs)
    for r in runs:
        try:
            df, _cfg = load_closed_loop_run(
                r.run_dir,
                prefer=prefer,
                extra_cols={
                    "section5_step": int(r.step),
                    "section5_step_name": str(r.step_name),
                    "section5_delay_sweep": float(r.delay_sweep)
                    if r.delay_sweep is not None
                    else np.nan,
                    "section5_vlm_query_hz_sweep": float(r.vlm_query_hz_sweep)
                    if r.vlm_query_hz_sweep is not None
                    else np.nan,
                    "section5_vlm_query_interval_s_sweep": float(r.vlm_query_interval_s_sweep)
                    if r.vlm_query_interval_s_sweep is not None
                    else np.nan,
                    "section5_group": (str(r.group) if r.group is not None else ""),
                },
            )
            parts.append(df)
        except Exception as e:
            print(f"Error loading run {r.run_dir}: {e}. skipped.")

    out = pd.concat(parts, ignore_index=True)

    # Convenience: a single "streaming frequency" column for downstream plots.
    if "policy" in out.columns:
        out["is_stream_query_policy"] = out["policy"].map(policy_is_stream_query).astype(bool)
    else:
        out["is_stream_query_policy"] = False

    # Prefer the sweep folder Hz when present, else config.
    hz = out.get("section5_vlm_query_hz_sweep", pd.Series(np.nan, index=out.index))
    interval_s = out.get("section5_vlm_query_interval_s_sweep", pd.Series(np.nan, index=out.index))
    cfg_hz = out.get("vlm_query_hz_config", pd.Series(np.nan, index=out.index))
    cfg_interval = out.get("vlm_query_interval_s_config", pd.Series(np.nan, index=out.index))

    # Preferred, human-friendly representation
    out["vlm_streaming_interval_s"] = np.where(np.isfinite(interval_s), interval_s, cfg_interval)
    # Hz is derived for convenience (prefer sweep Hz when present).
    out["vlm_streaming_hz"] = np.where(
        np.isfinite(hz),
        hz,
        np.where(
            np.isfinite(out["vlm_streaming_interval_s"]),
            1.0 / out["vlm_streaming_interval_s"],
            cfg_hz,
        ),
    )

    return out
