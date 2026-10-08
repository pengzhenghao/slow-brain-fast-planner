"""Trajectory-selection VQA prompting/parsing helpers.

This is the **release** prompt/parse surface for the repo: given an overlay image of candidate
trajectories, a VLM returns either:
- {"action":"select_trajectory","selected_index": <int>, ...}
- {"action":"stop", ...}

The older escalation/request-intervention prompt variants were intentionally removed from the
open-source release to keep the repository scope focused.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

VQATrajectoryActionType = Literal["select_trajectory", "stop"]


CRITICAL_OBJECTS_HAZARDS = (
    "Common critical objects / hazards to consider (not exhaustive):\n"
    "- people / pedestrians (including children)\n"
    "- cars / trucks / buses / motorcycles\n"
    "- bicycles / scooters\n"
    "- intersections / crosswalks / merging traffic\n"
    "- construction cones / barriers\n"
    "- utility pole / signpost / bollard\n"
    "- curb edge / drop-off / fall risk\n"
    "- stairs / steep slope\n"
    "- grass / off-limits area\n"
    "- loose dirt / gravel / sand / heavy leaves (low traction / not a sidewalk)\n"
    "- parked vehicles opening doors\n"
    "- clutter / boxes / trash cans\n"
    "- animals\n"
    "- occlusion / blind corner / poor visibility"
)

DEFAULT_DRIVING_STYLE = (
    "Intention / driving style:\n"
    "- Do not drive/walk on grass.\n"
    "- Prefer paved sidewalks/paths; avoid loose dirt, gravel, sand, or heavy leaf piles.\n"
    "- Yield to humans and give them right-of-way.\n"
    "- Stop at intersections when uncertain or when crossing traffic/pedestrians may be present.\n"
    "- Avoid colliding with any object.\n"
    "- If anything looks unsafe or ambiguous, prefer the safest option; if none look safe, choose stop."
)


@dataclass(frozen=True)
class VQATrajectoryPromptConfig:
    """Prompt configuration for the VQA-style trajectory selection interface."""

    task_name: str
    task_description: str | None = None
    include_goal_hint: bool = True
    # If true, include goal-relative geometry columns in the candidate table (end_goal_dist_m, goal_ang_diff_deg, progress_m).
    # If false, the table remains purely trajectory-geometry + (optional) planner confidence.
    include_goal_geometry_columns: bool = True
    include_candidate_count_hint: bool = True
    include_candidate_score_table: bool = True
    # If true, include planner confidence columns (score/probabilities) in the candidate table.
    # If false, still keep geometry information (end_xy, traj_dist_m, goal-relative metrics) but
    # omit planner confidence to reduce bias.
    include_planner_confidence: bool = True
    require_strict_json: bool = True

    # If true, the user message includes an image_ref part (OpenAI-compatible adapters can convert it
    # into an image_url data URI at request time). If false, only text is emitted.
    include_image_ref_part: bool = True


def vqa_default_legend_text() -> str:
    # Keep it short and stable; this is intended to be the only “legend” text we give.
    return (
        "Legend:\n"
        "- Camera: front-facing fisheye (~100° FOV). Distortion near the image edges is normal.\n"
        "- Colored polylines are candidate trajectories.\n"
        "- Each trajectory endpoint is marked with a small colored dot.\n"
        "- Each trajectory has a number label near its endpoint: that number is the candidate index.\n"
        "- The number label background color matches the trajectory color (use this to match label ↔ line).\n"
        "- If a label is moved for readability, a thin leader line connects the label to the endpoint dot.\n"
        "- The magenta marker labeled 'GOAL' is the goal (may be absent if no goal is provided).\n"
        "- The robot is at the bottom of the image; forward is upward."
    )


def build_vqa_trajectory_selection_messages(
    *,
    cfg: VQATrajectoryPromptConfig,
    overlay_frame_ref: str | None,
    history_frame_refs: list[str] | None = None,
    num_candidates: int | None = None,
    goal_text: str | None = None,
    goal_xy: list[float] | None = None,
    goal_distance_m: float | None = None,
    candidate_scores: list[dict[str, Any]] | None = None,
    image_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a minimal VQA prompt: system + (legend + image + action schema)."""

    # Output schema with examples (critical_object first for CoT-style reasoning).
    actions = (
        "Return exactly ONE action:\n"
        'Schema: {"critical_object":[...], "reason":"...", "action":"select_trajectory|stop", "selected_index":<int|null>}\n'
        "\n"
        "Reason requirements:\n"
        '- Keep "reason" short\n'
        "- Use less than 50 words\n"
        "\n"
        "Examples (copy this key order exactly):\n"
        '{"critical_object": [], "reason": "Path 8 goes toward the goal and makes good progress while staying on sidewalk.", "action": "select_trajectory", "selected_index": 8}\n'
        '{"critical_object": ["pedestrian"], "reason": "Pedestrian crossing ahead. Path 3 curves around them safely.", "action": "select_trajectory", "selected_index": 3}\n'
        '{"critical_object": ["construction", "blocked_path"], "reason": "All trajectories lead into blocked area.", "action": "stop", "selected_index": null}\n'
        "\n"
        "Rules:\n"
        '- If action="select_trajectory", selected_index must be the trajectory index shown in the image.\n'
        '- If action is "stop", set selected_index to null.\n'
        '- If there are no critical objects, set "critical_object": [].'
    )

    # Static prompt block describing table columns.
    # Keep in system prompt so it doesn't compete with dynamic per-snapshot info.
    notes = None
    if bool(cfg.include_candidate_score_table):
        if bool(cfg.include_planner_confidence):
            if bool(cfg.include_goal_geometry_columns):
                notes = (
                    "Candidate table columns:\n"
                    "- index: the trajectory label shown in the overlay image (use this value for selected_index).\n"
                    "- score: raw planner score (higher = better according to the planner model).\n"
                    "- raw_prob: softmax probability over ALL candidates before filtering.\n"
                    "- nms_prob: normalized probability within the *displayed* candidate set (only candidates shown in image have non-zero values).\n"
                    "- traj_dist_m: total path length of the trajectory (meters).\n"
                    "- end_xy: trajectory endpoint in robot frame (x_forward, y_left) in meters.\n"
                    "- end_goal_dist_m: distance from trajectory endpoint to the goal (meters).\n"
                    "- goal_ang_diff_deg: absolute angle difference between endpoint direction and goal direction (degrees).\n"
                    "- progress_m: progress toward the goal = goal_distance - end_goal_dist_m (meters).\n"
                    "NOTE: end_goal_dist_m / goal_ang_diff_deg / progress_m are pure geometry and do NOT account for obstacles/off-limits areas."
                )
            else:
                notes = (
                    "Candidate table columns:\n"
                    "- index: the trajectory label shown in the overlay image (use this value for selected_index).\n"
                    "- score: raw planner score (higher = better according to the planner model).\n"
                    "- raw_prob: softmax probability over ALL candidates before filtering.\n"
                    "- nms_prob: normalized probability within the *displayed* candidate set (only candidates shown in image have non-zero values).\n"
                    "- traj_dist_m: total path length of the trajectory (meters).\n"
                    "- end_xy: trajectory endpoint in robot frame (x_forward, y_left) in meters."
                )
        else:
            if bool(cfg.include_goal_geometry_columns):
                notes = (
                    "Candidate table columns (planner confidence intentionally hidden):\n"
                    "- index: the trajectory label shown in the overlay image (use this value for selected_index).\n"
                    "- traj_dist_m: total path length of the trajectory (meters).\n"
                    "- end_xy: trajectory endpoint in robot frame (x_forward, y_left) in meters.\n"
                    "- end_goal_dist_m: distance from trajectory endpoint to the goal (meters).\n"
                    "- goal_ang_diff_deg: absolute angle difference between endpoint direction and goal direction (degrees).\n"
                    "- progress_m: progress toward the goal = goal_distance - end_goal_dist_m (meters).\n"
                    "NOTE: end_goal_dist_m / goal_ang_diff_deg / progress_m are pure geometry and do NOT account for obstacles/off-limits areas."
                )
            else:
                notes = (
                    "Candidate table columns (planner confidence intentionally hidden):\n"
                    "- index: the trajectory label shown in the overlay image (use this value for selected_index).\n"
                    "- traj_dist_m: total path length of the trajectory (meters).\n"
                    "- end_xy: trajectory endpoint in robot frame (x_forward, y_left) in meters."
                )

    # Visual prompting: robot footprint (global assumption unless overridden by image_info).
    robot_width_m = 0.8
    overlay_traj_style = None
    try:
        if isinstance(image_info, dict):
            if image_info.get("robot_width_m") is not None:
                robot_width_m = float(image_info.get("robot_width_m"))
            if isinstance(image_info.get("overlay_traj_style"), str):
                overlay_traj_style = str(image_info.get("overlay_traj_style"))
    except Exception:
        robot_width_m = 0.8
        overlay_traj_style = None

    system_lines = [
        "You are a navigation assistant controlling a ground robot.",
        f"Robot footprint: assume the robot is {robot_width_m:.2f} m wide (use this as the clearance envelope).",
        "At each step, a local planner proposes multiple candidate trajectories (shown in an overlay image).",
        "Each candidate trajectory is a polyline of 20 waypoints sampled at 5 Hz (Δt=0.2 s), representing ~4 seconds of future motion.",
        "Your job is to choose ONE action for the next step.",
    ]
    if str(overlay_traj_style) == "corridor":
        system_lines.append(
            "Overlay note: each candidate may appear as a wide translucent corridor/bar. "
            "This corridor represents the swept robot footprint (width above) along that trajectory. "
            "If the corridor overlaps obstacles/people/off-walkway areas, treat it as unsafe."
        )
    # Extra overlay conventions (goal cues).
    try:
        if isinstance(image_info, dict) and bool(image_info.get("goal_direction_arrow")):
            system_lines.append(
                "Overlay note (goal): the goal may be shown as a floating magenta arrow above the scene with a dashed vertical line down to the ground-projected goal location."
            )
            if str(image_info.get("goal_projection_marker") or "none") == "none":
                system_lines.append(
                    "Overlay note (goal): there may be NO dot on the ground; use the dashed line as the anchor."
                )
    except Exception:
        pass
    if history_frame_refs:
        try:
            n_hist = int(len(history_frame_refs))
        except Exception:
            n_hist = 0
        if n_hist > 0:
            hist_span_s = max(0.0, float(n_hist) * 0.2)
            system_lines.append(
                f"Additional context: {n_hist} history camera frames are provided BEFORE the current overlay frame. "
                f"These frames are recorded at 5 Hz (Δt=0.2 s), so the total history span is ~{hist_span_s:.1f} s."
            )
    if cfg.task_description:
        system_lines.append(str(cfg.task_description).strip())
    else:
        system_lines.append(
            "Task: select a candidate trajectory index that is SAFE and makes progress toward the goal."
        )
        system_lines.append(
            "IMPORTANT: candidates are short-horizon (~4 seconds). You do NOT need to reach the goal in one step; "
            "instead choose a trajectory that moves toward the goal direction while staying safe."
        )
        system_lines.append(
            "Decision procedure:\n"
            "1) Scene understanding first: identify sidewalk/path vs grass/dirt/planters, and any pedestrians/obstacles.\n"
            "2) HARD REJECT any trajectory that goes onto grass/off-limits surface or too close to a pedestrian/obstacle.\n"
            "3) Only among the remaining safe options, use goal geometry (progress/angle) as a tie-breaker.\n"
            "4) If you are unsure whether a trajectory stays on sidewalk (ambiguous), choose a safer option or stop."
        )
        if bool(cfg.include_planner_confidence):
            system_lines.append(
                "Planner confidence (score/probabilities) may be provided in the table. Treat it as a weak hint only; do NOT trust it."
            )
        else:
            system_lines.append(
                "Planner confidence (score/probabilities) is intentionally hidden. Use the image + goal geometry to decide."
            )

    system_lines.extend(
        [
            "If none look safe, choose stop. If the image is missing/unclear or you cannot decide, choose stop.",
            (
                "Sometimes a very short trajectory indicates a 'stop-like' action. "
                'In that case, prefer returning {"action":"select_trajectory","selected_index": <that short trajectory index>} '
                "rather than returning stop."
            ),
        ]
    )
    system_lines.append("")
    system_lines.append(CRITICAL_OBJECTS_HAZARDS)
    system_lines.append("")
    system_lines.append(DEFAULT_DRIVING_STYLE)
    if notes:
        system_lines.append("")
        system_lines.append(str(notes))
    system_lines.append("")
    system_lines.append(
        "Goal semantics:\n"
        "- The magenta marker labeled 'GOAL' indicates the desired navigation target in the robot frame.\n"
        "- The goal can be far beyond the 4-second candidate horizon and may be off-screen.\n"
        "- Choose a candidate that moves TOWARD the goal (good progress/angle) while remaining safe.\n"
        "- Do NOT treat the goal marker as a physical object you must reach immediately.\n"
        "- IMPORTANT: do NOT choose a trajectory just because its index label is closest to the GOAL marker. Use the table + image.\n"
        "- Overlay convention (raised goal): the GOAL may be drawn as a floating arrow above the scene with a dashed vertical line.\n"
        "  The dashed vertical line indicates the ground-projected goal location (z=0). There may be no dot on the ground (or a tiny dot)."
    )
    system_lines.append("")
    system_lines.append(vqa_default_legend_text())
    system_lines.append("")
    system_lines.append(actions)
    if cfg.require_strict_json:
        system_lines.append("Return ONLY valid JSON. Do not include any extra text.")
    system = "\n".join(system_lines)

    user_parts: list[dict[str, Any]] = []
    user_text_lines: list[str] = []

    if bool(cfg.include_goal_hint):
        goal_lines: list[str] = []
        if goal_text:
            goal_lines.append(f"- Text: {goal_text.strip()}")
        if goal_xy is not None and len(goal_xy) >= 2:
            try:
                gx = float(goal_xy[0])
                gy = float(goal_xy[1])
                goal_lines.append(
                    f"- Vector (robot frame): x_forward={gx:.2f} m, y_left={gy:.2f} m"
                )
                ang = math.degrees(math.atan2(gy, gx)) if (gx != 0.0 or gy != 0.0) else 0.0
                goal_lines.append(f"- Bearing: {ang:+.1f} deg (0=forward, +left)")
            except Exception:
                pass
        if goal_distance_m is not None:
            try:
                goal_lines.append(f"- Distance: {float(goal_distance_m):.2f} m")
            except Exception:
                pass

        if goal_lines:
            user_text_lines.append("Goal:\n" + "\n".join(goal_lines))
        else:
            user_text_lines.append("Goal: (not provided in this snapshot)")

    if cfg.include_candidate_count_hint and num_candidates is not None:
        try:
            n = int(len(candidate_scores)) if candidate_scores else int(num_candidates)
        except Exception:
            n = int(num_candidates)
        user_text_lines.append(f"Number of candidates shown: {n}")

    if cfg.include_candidate_score_table and candidate_scores:
        user_text_lines.append(
            _format_candidate_confidence_table(
                candidate_scores,
                include_notes=False,
                include_planner_confidence=bool(cfg.include_planner_confidence),
                include_goal_geometry_columns=bool(cfg.include_goal_geometry_columns),
                goal_xy=(goal_xy if bool(cfg.include_goal_geometry_columns) else None),
            )
        )
    user_parts.append({"type": "text", "text": "\n\n".join(user_text_lines)})

    if cfg.include_image_ref_part:
        if history_frame_refs:
            for ref in history_frame_refs:
                if isinstance(ref, str) and ref.strip():
                    user_parts.append({"type": "image_ref", "image_ref": str(ref)})
        if overlay_frame_ref is not None:
            user_parts.append({"type": "image_ref", "image_ref": str(overlay_frame_ref)})

    return [{"role": "system", "content": system}, {"role": "user", "content": user_parts}]


def _extract_json_like(s: str) -> str | None:
    """Best-effort extraction of a JSON object or integer from a model response."""

    s = str(s).strip()
    if not s:
        return None

    # If there are code fences, prefer the first fenced block.
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner

    # Otherwise try to decode the first JSON value embedded in the string.
    try:
        dec = json.JSONDecoder()
        start_idxs: list[int] = []
        for ch in ("{", "["):
            i = s.find(ch)
            if i >= 0:
                start_idxs.append(i)
        m_int = re.search(r"-?\d+", s)
        if m_int:
            start_idxs.append(int(m_int.start()))
        start_idxs = sorted(set([i for i in start_idxs if i >= 0]))
        for i in start_idxs:
            try:
                _obj, end = dec.raw_decode(s[i:])
                if end is not None and int(end) > 0:
                    return s[i : i + int(end)].strip()
            except Exception:
                continue
    except Exception:
        pass

    m2 = re.search(r"\{[\s\S]*?\}", s)
    if m2:
        return m2.group(0).strip()

    m3 = re.search(r"-?\d+", s)
    if m3:
        return m3.group(0)

    return None


def parse_vqa_trajectory_action(
    raw: str,
    *,
    num_candidates: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a model response into a trajectory action dict."""

    s = _extract_json_like(str(raw))
    if not s:
        return None, "empty_response"

    try:
        obj = json.loads(s)
    except Exception:
        try:
            idx = int(s)
            if 0 <= idx < int(num_candidates):
                return {"action": "select_trajectory", "selected_index": idx}, "fallback_int"
            return None, f"index_out_of_range:{idx}"
        except Exception:
            return None, "invalid_json"

    if isinstance(obj, int):
        idx2 = int(obj)
        if 0 <= idx2 < int(num_candidates):
            return {
                "action": "select_trajectory",
                "selected_index": idx2,
                "critical_object": [],
                "risks": [],
            }, None
        return None, f"index_out_of_range:{idx2}"

    if not isinstance(obj, dict):
        return None, "json_not_object"

    rationale: str | None = None
    try:
        rat = obj.get("rationale")
        if isinstance(rat, str) and rat.strip():
            rationale = rat.strip()
        else:
            rsn = obj.get("reason")
            if isinstance(rsn, str) and rsn.strip():
                rationale = rsn.strip()
    except Exception:
        rationale = None

    if "action" not in obj and "selected_index" in obj:
        try:
            idx3 = int(obj["selected_index"])
        except Exception:
            return None, "selected_index_not_int"
        if 0 <= idx3 < int(num_candidates):
            out = {"action": "select_trajectory", "selected_index": idx3}
            if rationale is not None:
                out["rationale"] = rationale
            out["critical_object"], out["risks"] = _extract_critical_and_risks(obj)
            return out, "legacy_selected_index"
        return None, f"index_out_of_range:{idx3}"

    action = obj.get("action") or obj.get("type")
    if action in ("select", "select_trajectory", "trajectory", "choose", "choose_trajectory"):
        if "selected_index" not in obj and "index" in obj:
            obj["selected_index"] = obj.get("index")
        if "selected_index" not in obj:
            return None, "missing_selected_index"
        try:
            idx4 = int(obj["selected_index"])
        except Exception:
            return None, "selected_index_not_int"
        if 0 <= idx4 < int(num_candidates):
            out2 = {"action": "select_trajectory", "selected_index": idx4}
            if rationale is not None:
                out2["rationale"] = rationale
            out2["critical_object"], out2["risks"] = _extract_critical_and_risks(obj)
            return out2, None
        return None, f"index_out_of_range:{idx4}"

    if action in ("stop", "STOP"):
        out3 = {"action": "stop"}
        if rationale is not None:
            out3["rationale"] = rationale
        out3["critical_object"], out3["risks"] = _extract_critical_and_risks(obj)
        return out3, None

    if action in ("ask_for_help", "ask", "help"):
        out4 = {"action": "stop"}
        if rationale is not None:
            out4["rationale"] = rationale
        out4["critical_object"], out4["risks"] = _extract_critical_and_risks(obj)
        return out4, "normalized_action:ask_for_help->stop"

    return None, f"unknown_action:{action!r}"


def _extract_critical_and_risks(obj: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Extract optional `critical_object`/`risks` fields (best-effort, robust)."""

    critical: list[str] = []
    try:
        co = obj.get("critical_object") if "critical_object" in obj else obj.get("critical_objects")
        if isinstance(co, str) and co.strip():
            critical = [co.strip()]
        elif isinstance(co, list):
            critical = [str(x).strip() for x in co if str(x).strip()][:8]
    except Exception:
        critical = []

    risks: list[str] = []
    try:
        r = obj.get("risks") or obj.get("risk_factors")
        if isinstance(r, list):
            risks = [str(x).strip() for x in r if str(x).strip()][:8]
    except Exception:
        risks = []

    return critical, risks


def _format_candidate_confidence_table(
    rows: list[dict[str, Any]],
    *,
    include_notes: bool = True,
    include_planner_confidence: bool = True,
    include_goal_geometry_columns: bool = True,
    goal_xy: list[float] | None = None,
) -> str:
    """Format candidate confidence rows as an aligned monospace table.

    Only shows candidates that are displayed in the overlay image.
    """

    _ = bool(include_notes)  # preserved for API compatibility; notes live in the system prompt

    lines: list[str] = []
    lines.append("Candidate trajectories shown in image:")
    if not bool(include_planner_confidence):
        lines.append("(Ordering: index ascending; planner confidence columns hidden)")

    # Parse goal info once (best-effort).
    gx = gy = None
    goal_dist_m = None
    goal_ang_deg = None
    try:
        if goal_xy is not None and len(goal_xy) >= 2:
            gx = float(goal_xy[0])
            gy = float(goal_xy[1])
            if math.isfinite(gx) and math.isfinite(gy):
                goal_dist_m = float(math.hypot(gx, gy))
                goal_ang_deg = (
                    float(math.degrees(math.atan2(gy, gx))) if (gx != 0.0 or gy != 0.0) else 0.0
                )
    except Exception:
        gx = gy = goal_dist_m = goal_ang_deg = None

    rows2: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        rr = dict(r)
        if not bool(include_planner_confidence):
            rr.pop("score", None)
            rr.pop("raw_prob", None)
            rr.pop("nms_prob", None)

        # Add goal-relative metrics if enabled and end_xy and goal are available.
        try:
            end_xy = rr.get("end_xy")
            if (
                bool(include_goal_geometry_columns)
                and gx is not None
                and gy is not None
                and goal_dist_m is not None
                and goal_ang_deg is not None
                and isinstance(end_xy, (list, tuple))
                and len(end_xy) >= 2
            ):
                ex = float(end_xy[0])
                ey = float(end_xy[1])
                if math.isfinite(ex) and math.isfinite(ey):
                    end_goal_dist = float(math.hypot(float(gx) - ex, float(gy) - ey))
                    rr["end_goal_dist_m"] = end_goal_dist
                    end_ang = (
                        float(math.degrees(math.atan2(ey, ex))) if (ex != 0.0 or ey != 0.0) else 0.0
                    )
                    diff = (end_ang - float(goal_ang_deg) + 180.0) % 360.0 - 180.0
                    rr["goal_ang_diff_deg"] = abs(float(diff))
                    rr["progress_m"] = float(goal_dist_m) - float(end_goal_dist)
        except Exception:
            pass
        rows2.append(rr)

    if not bool(include_planner_confidence):
        try:
            rows2.sort(key=lambda d: int(d.get("index", 0)))
        except Exception:
            pass

    preferred_cols: list[str] = ["index"]
    if bool(include_planner_confidence):
        preferred_cols += ["score", "raw_prob", "nms_prob"]
    preferred_cols += ["traj_dist_m", "end_xy"]
    if bool(include_goal_geometry_columns):
        preferred_cols += ["end_goal_dist_m", "goal_ang_diff_deg", "progress_m"]

    cols = [c for c in preferred_cols if any((c in r) for r in rows2 if isinstance(r, dict))]
    if not cols:
        cols = ["index"]

    def fmt(col: str, v: Any) -> str:
        if v is None:
            return "-"
        try:
            if col == "index":
                return str(int(v))
            if col == "score":
                return f"{float(v):.2f}"
            if col == "traj_dist_m":
                return f"{float(v):.1f}m"
            if col == "end_goal_dist_m":
                return f"{float(v):.1f}m"
            if col == "progress_m":
                return f"{float(v):+.1f}m"
            if col == "goal_ang_diff_deg":
                return f"{float(v):.1f}deg"
            if "prob" in col:
                return f"{float(v):.3f}"
            if col == "end_xy":
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    return f"({float(v[0]):.1f}, {float(v[1]):.1f})"
                return str(v)
        except Exception:
            return str(v)
        return str(v)

    formatted_rows: list[list[str]] = []
    widths = {c: len(c) for c in cols}
    for r in rows2:
        fr: list[str] = []
        for c in cols:
            s = fmt(c, r.get(c))
            widths[c] = max(widths[c], len(s))
            fr.append(s)
        formatted_rows.append(fr)

    header = " | ".join([c.rjust(widths[c]) for c in cols])
    sep = "-+-".join(["-" * widths[c] for c in cols])
    lines.append(header)
    lines.append(sep)

    for fr in formatted_rows:
        lines.append(" | ".join([fr[i].rjust(widths[cols[i]]) for i in range(len(cols))]))

    return "\n".join(lines)
