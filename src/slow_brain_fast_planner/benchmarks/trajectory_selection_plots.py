from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_PLT: Any = None  # None = not loaded, False = disabled, else = matplotlib.pyplot

# Colormap for candidate trajectories (distinct colors)
CANDIDATE_COLORS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#ffe119",  # yellow
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabed4",  # pink
]


def _get_plt():
    """Lazy matplotlib import (keeps CLI runnable in minimal envs)."""
    global _PLT
    if _PLT is False:
        return None
    if _PLT is not None:
        return _PLT
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        _PLT = plt
        return _PLT
    except Exception:
        _PLT = False
        return None


def write_local_frame_plot(
    *,
    out_path: Path,
    episode_id: str,
    t: float,
    selected_traj_xy: np.ndarray | None,
    gt_human_traj_xy: np.ndarray | None,
    gt_route_xy: np.ndarray | None,
    all_candidates_xy: list[np.ndarray] | None = None,
    candidate_scores: list[float] | None = None,
    selected_index: int | None = None,
    label_index: int | None = None,
    score_index: int | None = None,
    min_ade_index: int | None = None,
    min_ade_all_index: int | None = None,
    goal_xy: np.ndarray | None = None,
    goal_far_threshold_m: float = 30.0,
    title_suffix: str | None = None,
) -> None:
    """Write a per-snapshot top-down plot in the robot frame at time t.

    All trajectories are assumed to already be in the same robot frame:
    - x forward, y left, origin at robot base_link at time t.

    Args:
        out_path: Output PNG path
        episode_id: Episode identifier for title
        t: Timestamp
        selected_traj_xy: The selected (model-chosen) trajectory [N, 2]
        gt_human_traj_xy: Ground truth human future trajectory [N, 2]
        gt_route_xy: Ground truth route polyline [M, 2]
        all_candidates_xy: List of all candidate trajectories (each [N, 2])
        candidate_scores: Scores for each candidate (for legend)
        selected_index: Index of the selected candidate (highlighted)
        label_index: Index of the label/GT candidate (highlighted differently)
        score_index: Index of the highest-score candidate (for reference; legend entry)
        min_ade_index: Index of the min-ADE candidate (VISIBLE / post-NMS pool; legend entry)
        min_ade_all_index: Index of the global min-ADE candidate (ALL candidates; legend entry)
        goal_xy: Goal point in robot frame [2,] (optional)
        goal_far_threshold_m: If goal is farther than this, draw a clipped goal marker and annotate
        distance.
        title_suffix: Optional suffix for the title
    """
    plt = _get_plt()
    if plt is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))

    # Draw GT route first (background)
    if gt_route_xy is not None and len(gt_route_xy) >= 2:
        plt.plot(
            gt_route_xy[:, 0], gt_route_xy[:, 1], "m--", linewidth=2, alpha=0.35, label="_nolegend_"
        )

    # Draw all candidate trajectories
    if all_candidates_xy is not None:
        for _i, cand_xy in enumerate(all_candidates_xy):
            if cand_xy is None or len(cand_xy) < 2:
                continue
            # Ordinary candidates: grey + low alpha, no legend.
            plt.plot(
                cand_xy[:, 0],
                cand_xy[:, 1],
                color="#9a9a9a",
                linestyle="-",
                linewidth=1.2,
                alpha=0.18,
                zorder=2,
                label="_nolegend_",
            )
    elif selected_traj_xy is not None and len(selected_traj_xy) >= 2:
        # Fallback: just draw the selected trajectory if all_candidates not provided
        plt.plot(
            selected_traj_xy[:, 0],
            selected_traj_xy[:, 1],
            "g-",
            linewidth=3,
            alpha=0.95,
            label="Selected",
        )

    # Highlight reference candidates (legend entries only for: min ADE / highest score / selected)
    def _plot_cand_idx(
        idx: int,
        *,
        color: str,
        label: str,
        linestyle: str = "-",
        lw: float = 3.0,
        alpha: float = 0.95,
    ) -> None:
        if all_candidates_xy is None:
            return
        if idx < 0 or idx >= len(all_candidates_xy):
            return
        cand_xy = all_candidates_xy[idx]
        if cand_xy is None or len(cand_xy) < 2:
            return
        plt.plot(
            cand_xy[:, 0],
            cand_xy[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=lw,
            alpha=alpha,
            zorder=12,
            label=label,
        )

    # Highest-score (argmax score) and min-ADE (oracle) references
    if score_index is not None:
        _plot_cand_idx(
            int(score_index),
            color="#f58231",
            label="Highest score",
            linestyle="--",
            lw=2.8,
            alpha=0.9,
        )
    if min_ade_index is not None:
        _plot_cand_idx(
            int(min_ade_index),
            color="#4363d8",
            label="Min ADE (visible)",
            linestyle="--",
            lw=2.8,
            alpha=0.9,
        )
    if min_ade_all_index is not None:
        _plot_cand_idx(
            int(min_ade_all_index),
            color="#911eb4",
            label="Min ADE (all)",
            linestyle="--",
            lw=2.8,
            alpha=0.9,
        )

    # Selected trajectory highlight (prefer index -> candidate for consistent rendering)
    if selected_index is not None:
        _plot_cand_idx(
            int(selected_index),
            color="#3cb44b",
            label="Selected",
            linestyle="-",
            lw=3.6,
            alpha=0.98,
        )
    elif selected_traj_xy is not None and len(selected_traj_xy) >= 2:
        plt.plot(
            selected_traj_xy[:, 0],
            selected_traj_xy[:, 1],
            color="#3cb44b",
            linestyle="-",
            linewidth=3.6,
            alpha=0.98,
            zorder=12,
            label="Selected",
        )

    # Optional label index (debug-only): do not include in legend
    if label_index is not None and all_candidates_xy is not None:
        try:
            li = int(label_index)
            if 0 <= li < len(all_candidates_xy):
                c = all_candidates_xy[li]
                if c is not None and len(c) >= 2:
                    plt.plot(
                        c[:, 0],
                        c[:, 1],
                        color="#e6194b",
                        linestyle=":",
                        linewidth=2.2,
                        alpha=0.7,
                        zorder=11,
                        label="_nolegend_",
                    )
        except Exception:
            pass

    # Draw GT human trajectory (on top)
    if gt_human_traj_xy is not None and len(gt_human_traj_xy) >= 2:
        plt.plot(
            gt_human_traj_xy[:, 0],
            gt_human_traj_xy[:, 1],
            "c-",
            linewidth=3,
            alpha=0.9,
            zorder=15,
            label="GT future",
        )
        # Draw endpoint
        plt.scatter(
            [gt_human_traj_xy[-1, 0]],
            [gt_human_traj_xy[-1, 1]],
            color="cyan",
            s=80,
            marker="*",
            zorder=16,
            edgecolors="white",
            linewidths=1,
        )

    # Draw goal (robot frame). If far, clip and annotate.
    goal_plot_xy: tuple[float, float] | None = None
    if goal_xy is not None:
        try:
            g = np.asarray(goal_xy, dtype=np.float64).reshape(-1)
            if g.size >= 2:
                gx = float(g[0])
                gy = float(g[1])
                dist = float((gx * gx + gy * gy) ** 0.5)
                far_thr = float(goal_far_threshold_m)
                if far_thr > 0 and dist > far_thr:
                    scale = far_thr / dist if dist > 1e-9 else 1.0
                    px = gx * scale
                    py = gy * scale
                    goal_plot_xy = (float(px), float(py))
                    plt.scatter(
                        [px],
                        [py],
                        color="#ffe119",
                        s=120,
                        marker="*",
                        zorder=18,
                        edgecolors="k",
                        linewidths=0.8,
                        label="_nolegend_",
                    )
                    plt.annotate(
                        f"goal far ({dist:.1f} m)",
                        xy=(px, py),
                        xytext=(px + 1.0, py + 1.0),
                        arrowprops={"arrowstyle": "->", "color": "#ffe119", "alpha": 0.8},
                        fontsize=9,
                        color="#ffe119",
                        zorder=19,
                    )
                else:
                    goal_plot_xy = (float(gx), float(gy))
                    plt.scatter(
                        [gx],
                        [gy],
                        color="#ffe119",
                        s=120,
                        marker="*",
                        zorder=18,
                        edgecolors="k",
                        linewidths=0.8,
                        label="_nolegend_",
                    )
        except Exception:
            pass

    # Draw robot origin
    plt.plot([0.0], [0.0], "ko", markersize=8, label="_nolegend_", zorder=20)
    plt.axhline(0.0, color="k", linewidth=1, alpha=0.15)
    plt.axvline(0.0, color="k", linewidth=1, alpha=0.15)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    # Enforce a symmetric 1:1 view window around the origin so coco/real plots look comparable.
    try:
        max_abs = 0.0
        series: list[np.ndarray] = []
        for arr in (gt_route_xy, gt_human_traj_xy, selected_traj_xy):
            if arr is not None and isinstance(arr, np.ndarray) and arr.size >= 2:
                series.append(np.asarray(arr, dtype=np.float64))
        if all_candidates_xy is not None:
            for c in all_candidates_xy:
                if c is not None and isinstance(c, np.ndarray) and c.size >= 2:
                    series.append(np.asarray(c, dtype=np.float64))
        if series:
            pts = np.concatenate(
                [s[:, :2] for s in series if s.ndim == 2 and s.shape[1] >= 2], axis=0
            )
            if pts.size > 0:
                max_abs = float(np.max(np.abs(pts)))
        if goal_plot_xy is not None:
            max_abs = max(max_abs, float(max(abs(goal_plot_xy[0]), abs(goal_plot_xy[1]))))
        r = max(5.0, max_abs * 1.05)
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
    except Exception:
        pass
    plt.grid(True, alpha=0.25)
    plt.xlabel("x_forward (m)")
    plt.ylabel("y_left (m)")

    title = f"{episode_id}  t={t:.3f}s  (robot local frame)"
    if title_suffix:
        title = f"{title}  {title_suffix}"
    plt.title(title)

    # Legend: only show reference lines (min ADE / highest score / selected / GT)
    handles, labels = plt.gca().get_legend_handles_labels()
    keep: list[tuple[Any, str]] = []
    seen = set()
    for h, lab in zip(handles, labels, strict=False):
        if not lab or lab == "_nolegend_":
            continue
        if lab in seen:
            continue
        seen.add(lab)
        keep.append((h, lab))
    if keep:
        plt.legend(
            [h for h, _ in keep],
            [lab for _, lab in keep],
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
