#!/usr/bin/env python
"""
Measure Task 2 VLM token usage (Gemini/OpenAI-compatible adapters).

This is a lightweight harness that builds the *same* Task 2 VQA prompt template used by
`scripts/run_trajectory_selection.py`, but uses a generated overlay PNG instead of a real dataset.

Why:
- Gemini usage metadata can include cached input tokens and thinking tokens.
- Our benchmark aggregates token totals from provider metadata; this script helps sanity-check
  per-call averages and the breakdown (uncached prompt vs cached prompt vs output vs thoughts).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from slow_brain_fast_planner.benchmarks.model_adapters import GeminiGenAIAdapter, GeminiGenAIConfig
from slow_brain_fast_planner.benchmarks.vqa_trajectory import (
    VQATrajectoryPromptConfig,
    build_vqa_trajectory_selection_messages,
)


def _make_dummy_overlay_png(path: Path, *, size: int) -> None:
    img = Image.new("RGB", (int(size), int(size)), (18, 18, 20))
    draw = ImageDraw.Draw(img)

    # Draw a simple "scene": horizon + a few colored trajectory-like polylines.
    draw.rectangle([0, 0, size, int(size * 0.35)], fill=(30, 30, 35))
    draw.rectangle([0, int(size * 0.35), size, size], fill=(20, 20, 22))

    colors = [
        (255, 120, 120),
        (120, 255, 120),
        (120, 170, 255),
        (255, 210, 120),
        (220, 120, 255),
        (120, 255, 240),
    ]
    base_x = int(size * 0.5)
    base_y = int(size * 0.95)
    for i in range(18):
        c = colors[i % len(colors)]
        dx = int((i - 9) * (size * 0.02))
        pts = [
            (base_x, base_y),
            (base_x + dx, int(size * 0.75)),
            (base_x + int(dx * 1.3), int(size * 0.55)),
            (base_x + int(dx * 1.6), int(size * 0.40)),
        ]
        draw.line(pts, fill=c, width=max(2, size // 256))
        draw.ellipse(
            [pts[-1][0] - 3, pts[-1][1] - 3, pts[-1][0] + 3, pts[-1][1] + 3],
            fill=c,
            outline=None,
        )

    # Add a tiny label (helps mimic overlay text without relying on system fonts).
    try:
        font = ImageFont.load_default()
        draw.text((10, 10), "DUMMY OVERLAY", fill=(240, 240, 240), font=font)
    except Exception:
        draw.text((10, 10), "DUMMY OVERLAY", fill=(240, 240, 240))

    img.save(str(path), format="PNG")


def _fake_candidate_table(topk: int, *, include_goal_geom: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(int(topk)):
        end_x = 0.4 + 0.05 * float(i)
        end_y = (-1.0 if (i % 2) else 1.0) * (0.15 + 0.03 * float(i))
        traj_dist = 2.0 + 0.12 * float(i)
        row: dict[str, Any] = {
            "index": int(i),
            "score": float(topk - i),
            "raw_prob": float(1.0 / max(1, topk)),
            "nms_prob": float(1.0 / max(1, topk)),
            "traj_dist_m": float(traj_dist),
            "end_xy": [float(end_x), float(end_y)],
        }
        if include_goal_geom:
            # Arbitrary but consistent geometry.
            end_goal_dist = max(0.0, 5.0 - 0.15 * float(i))
            goal_ang = abs(30.0 - 1.1 * float(i))
            row.update(
                {
                    "end_goal_dist_m": float(end_goal_dist),
                    "goal_ang_diff_deg": float(goal_ang),
                    "progress_m": float(5.0 - float(end_goal_dist)),
                }
            )
        rows.append(row)
    return rows


def _as_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gemini-model", default="gemini-2.5-flash", help="Gemini model name.")
    p.add_argument("--n", type=int, default=10, help="Number of repeated calls.")
    p.add_argument("--topk", type=int, default=18, help="Number of candidates in the prompt table.")
    p.add_argument(
        "--image-size", type=int, default=768, help="Overlay PNG width/height in pixels."
    )
    p.add_argument(
        "--include-goal-geometry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include goal geometry columns in the candidate table.",
    )
    p.add_argument(
        "--include-goal-info",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include goal text/hint in the prompt (Task2 `--prompt-goal-info`).",
    )
    p.add_argument(
        "--prompt-show-scores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include planner confidence columns (score/probabilities) in prompt.",
    )
    p.add_argument(
        "--include-thoughts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request thought parts in the response (debugging only; may increase tokens).",
    )
    args = p.parse_args()

    # Keep streaming disabled so each call exposes complete usage metadata.
    os.environ["GEMINI_STREAMING"] = "0"
    os.environ["GEMINI_ENABLE_STREAMING"] = "0"

    prompt_cfg = VQATrajectoryPromptConfig(
        task_name="task2_trajectory_selection",
        task_description=None,
        include_candidate_score_table=True,
        include_planner_confidence=bool(args.prompt_show_scores),
        include_goal_hint=bool(args.include_goal_info),
        include_goal_geometry_columns=bool(args.include_goal_geometry),
        include_image_ref_part=True,
        require_strict_json=True,
    )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        overlay_path = td_path / "overlay.png"
        _make_dummy_overlay_png(overlay_path, size=int(args.image_size))

        adapter = GeminiGenAIAdapter(
            GeminiGenAIConfig(
                model=str(args.gemini_model),
                temperature=0.0,
                image_base_dir=td_path,
                include_thoughts=bool(args.include_thoughts),
            )
        )

        candidate_scores = _fake_candidate_table(
            int(args.topk), include_goal_geom=bool(args.include_goal_geometry)
        )

        messages = build_vqa_trajectory_selection_messages(
            cfg=prompt_cfg,
            overlay_frame_ref=str(overlay_path),
            history_frame_refs=None,
            num_candidates=int(args.topk),
            goal_text=("Go to the goal." if bool(args.include_goal_info) else None),
            goal_xy=([5.0, 0.0] if bool(args.include_goal_info) else None),
            goal_distance_m=(5.0 if bool(args.include_goal_info) else None),
            candidate_scores=candidate_scores,
            image_info={
                "overlay_traj_style": "polyline",
                "robot_width_m": 0.8,
                "goal_direction_arrow": False,
                "goal_projection_marker": "none",
            },
        )

        per_call: list[dict[str, Any]] = []
        for _i in range(int(args.n)):
            _ = adapter.call(messages=messages, obs={})
            u = adapter.last_usage or {}
            per_call.append(
                {
                    "prompt_tokens": _as_int(u.get("prompt_tokens")),
                    "cached_prompt_tokens": _as_int(u.get("cached_prompt_tokens")),
                    "output_tokens": _as_int(u.get("output_tokens")),
                    "thoughts_tokens": _as_int(u.get("thoughts_tokens")),
                    "total_tokens": _as_int(u.get("total_tokens")),
                }
            )

        def _mean(vals: list[int | None]) -> float | None:
            xs = [int(v) for v in vals if v is not None]
            return float(statistics.mean(xs)) if xs else None

        summary = {
            "gemini_model": str(args.gemini_model),
            "n": int(args.n),
            "topk": int(args.topk),
            "image_size": int(args.image_size),
            "prompt_cfg": asdict(prompt_cfg),
            "mean": {
                "prompt_tokens": _mean([r["prompt_tokens"] for r in per_call]),
                "cached_prompt_tokens": _mean([r["cached_prompt_tokens"] for r in per_call]),
                "output_tokens": _mean([r["output_tokens"] for r in per_call]),
                "thoughts_tokens": _mean([r["thoughts_tokens"] for r in per_call]),
                "total_tokens": _mean([r["total_tokens"] for r in per_call]),
            },
            "per_call": per_call,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
