from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from slow_brain_fast_planner import constants
from slow_brain_fast_planner.benchmarks.planner_postprocessing import (
    _trajectory_nms_endpoints,
    planner_postprocess_v2,
)
from slow_brain_fast_planner.benchmarks.rgb_frame_loader import RGBFrameLoader
from slow_brain_fast_planner.schema.canonical_episode import PlannerCandidatesRecord, RGBRecord


@dataclass(frozen=True)
class OverlayConfig:
    top_k: int = constants.NMS_MAX_TRAJECTORIES
    min_score: float | None = None

    # Projection model.
    projection: str = constants.OVERLAY_PROJECTION  # "fisheye_v1" | "simple_xy"

    # Candidate set selection.
    candidate_set: str = (
        "planner_v2"  # "planner_v2" | "raw_topk" | "kcenter_endpoints" | "nms_only"
    )
    nms_max_trajectories: int = constants.NMS_MAX_TRAJECTORIES
    nms_distance_threshold: float = constants.NMS_DISTANCE_THRESHOLD
    prob_threshold: float = constants.PROB_THRESHOLD

    # --- Fisheye projection (calibrated parameters for the front fisheye camera) ---
    camera_height_m: float = constants.CAMERA_HEIGHT_M
    fisheye_k: float = constants.FISHEYE_K
    # If fx/fy/cx/cy are None, they are derived from image size at render time.
    fisheye_fx: float | None = constants.FISHEYE_FX
    fisheye_fy: float | None = constants.FISHEYE_FY
    fisheye_cx: float | None = constants.FISHEYE_CX
    fisheye_cy: float | None = constants.FISHEYE_CY

    # --- Simple projection from robot (x forward, y left) to image pixels ---
    px_per_meter: float = 25.0
    origin_u: float = 0.5  # normalized x in [0,1]
    origin_v: float = 0.9  # normalized y in [0,1]

    # Rendering.
    line_width: int = constants.LINE_WIDTH
    # Default to the same thickness as the regular line overlays.
    # Keep highlights non-discriminative by using the same width unless explicitly overridden.
    highlight_line_width: int = constants.LINE_WIDTH
    alpha: float = constants.ALPHA
    draw_points: bool = True
    point_radius: int = 2
    point_every: int = 2

    # Trajectory visualization style (for visual prompting).
    # - "line": draw a centerline polyline (legacy/default)
    # - "corridor": draw a filled corridor representing robot footprint along the trajectory
    traj_style: str = "line"  # "line" | "corridor"
    # Assumed robot width in meters for corridor rendering (half width is applied to left/right).
    robot_width_m: float = 0.8
    # Fill alpha for the corridor polygon (separate from `alpha` used for line/points).
    corridor_alpha: float = 0.18
    # If true, draw left/right corridor boundary lines on top of the filled polygon.
    corridor_outline: bool = True

    highlight_pred: bool = True
    highlight_label: bool = True

    # VLM-friendly labeling.
    label_indices: bool = True
    label_font_size: int = constants.LABEL_FONT_SIZE
    label_bg_alpha: float = 0.75
    label_text_rgb: tuple[int, int, int] = (255, 255, 255)
    label_bg_rgb: tuple[int, int, int] = (0, 0, 0)

    # Sector fan overlay (for anchor/sector masking prompts).
    # This is a *fixed* robot-frame angular partition; it does not depend on candidate ranking.
    #
    # Semantics:
    # - bearing is computed in robot frame: 0°=forward (+x), +left (+y), -right (-y)
    # - sector_id increases from right->left as bearing increases
    draw_sectors: bool = False
    sector_count: int = 8
    sector_min_bearing_deg: float = -90.0
    sector_max_bearing_deg: float = 90.0
    sector_ray_min_m: float = 0.6
    sector_ray_max_m: float = 10.0
    sector_ray_steps: int = 9
    sector_line_width: int = 2
    sector_line_rgb: tuple[int, int, int] = (255, 255, 255)
    sector_line_alpha: float = 0.18
    sector_label: bool = True
    sector_label_margin_px: int = 22

    # Goal visualization.
    # We support:
    # - goal_text: drawn as a top banner (e.g., `episode.static_context.goal_description`)
    # - goal_xy: drawn as a projected marker in the image (robot frame x-forward, y-left)
    draw_goal_text: bool = True
    goal_text_font_size: int = 16
    goal_text_bg_alpha: float = 0.85
    goal_text_rgb: tuple[int, int, int] = (255, 255, 255)
    goal_text_bg_rgb: tuple[int, int, int] = (0, 0, 0)
    goal_text_max_chars: int = 140

    draw_goal_marker: bool = True
    goal_marker_rgb: tuple[int, int, int] = (255, 0, 255)  # magenta (high-contrast)
    goal_marker_radius: int = 8
    goal_marker_line_width: int = 3
    # VLM-friendly: draw an additional "hanging" arrow that indicates the goal direction
    # (so the model doesn't confuse the goal marker with a required endpoint to reach within the
    # short horizon).
    draw_goal_direction_arrow: bool = False
    # When using draw_goal_direction_arrow, optionally draw a small marker at the ground-projected
    # anchor.
    # Default: none (avoid giving the VLM a "pick the label closest to the dot" shortcut).
    goal_projection_marker: str = "none"  # "none" | "dot"
    goal_projection_dot_radius: int = 4
    goal_projection_dot_alpha: float = 0.55

    # Canvas padding: extend the image so short/out-of-frame trajectories can still be shown.
    pad_left: int = 0
    pad_right: int = 0
    pad_top: int = 0
    pad_bottom: int = constants.PAD_BOTTOM
    pad_rgb: tuple[int, int, int] = (0, 0, 0)

    # If true, extend each drawn trajectory to the bottom edge of the overlay image.
    # This is useful when pad_bottom=0, where the first projected waypoint can be above the bottom
    # and the trajectory may look like it "starts in mid-air".
    extend_start_to_bottom: bool = True

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.projection not in ("fisheye_v1", "simple_xy"):
            raise ValueError("projection must be one of: fisheye_v1, simple_xy")
        if self.candidate_set not in ("planner_v2", "raw_topk", "kcenter_endpoints", "nms_only"):
            raise ValueError(
                "candidate_set must be one of: planner_v2, raw_topk, kcenter_endpoints, nms_only"
            )
        if self.nms_max_trajectories <= 0:
            raise ValueError("nms_max_trajectories must be > 0")
        if self.nms_distance_threshold < 0:
            raise ValueError("nms_distance_threshold must be >= 0")
        if not (0.0 <= self.prob_threshold <= 1.0):
            raise ValueError("prob_threshold must be in [0,1]")
        if self.camera_height_m <= 0:
            raise ValueError("camera_height_m must be > 0")
        if self.px_per_meter <= 0:
            raise ValueError("px_per_meter must be > 0")
        if not (0.0 <= self.origin_u <= 1.0 and 0.0 <= self.origin_v <= 1.0):
            raise ValueError("origin_u/origin_v must be in [0,1]")
        if self.line_width <= 0:
            raise ValueError("line_width must be > 0")
        if self.highlight_line_width <= 0:
            raise ValueError("highlight_line_width must be > 0")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")
        if self.point_radius < 0:
            raise ValueError("point_radius must be >= 0")
        if self.point_every <= 0:
            raise ValueError("point_every must be > 0")
        if self.traj_style not in ("line", "corridor"):
            raise ValueError("traj_style must be one of: line, corridor")
        if not (self.robot_width_m > 0.0) or (not math.isfinite(float(self.robot_width_m))):
            raise ValueError("robot_width_m must be finite and > 0")
        if not (0.0 <= float(self.corridor_alpha) <= 1.0):
            raise ValueError("corridor_alpha must be in [0,1]")
        if self.label_font_size <= 0:
            raise ValueError("label_font_size must be > 0")
        if not (0.0 <= self.label_bg_alpha <= 1.0):
            raise ValueError("label_bg_alpha must be in [0,1]")
        if self.sector_count <= 0:
            raise ValueError("sector_count must be > 0")
        if not (float(self.sector_min_bearing_deg) < float(self.sector_max_bearing_deg)):
            raise ValueError("sector_min_bearing_deg must be < sector_max_bearing_deg")
        if self.sector_ray_min_m <= 0:
            raise ValueError("sector_ray_min_m must be > 0")
        if self.sector_ray_max_m <= 0:
            raise ValueError("sector_ray_max_m must be > 0")
        if float(self.sector_ray_min_m) >= float(self.sector_ray_max_m):
            raise ValueError("sector_ray_min_m must be < sector_ray_max_m")
        if self.sector_ray_steps < 2:
            raise ValueError("sector_ray_steps must be >= 2")
        if self.sector_line_width <= 0:
            raise ValueError("sector_line_width must be > 0")
        if not (0.0 <= float(self.sector_line_alpha) <= 1.0):
            raise ValueError("sector_line_alpha must be in [0,1]")
        if int(self.sector_label_margin_px) < 0:
            raise ValueError("sector_label_margin_px must be >= 0")
        if self.goal_text_font_size <= 0:
            raise ValueError("goal_text_font_size must be > 0")
        if not (0.0 <= self.goal_text_bg_alpha <= 1.0):
            raise ValueError("goal_text_bg_alpha must be in [0,1]")
        if self.goal_text_max_chars <= 0:
            raise ValueError("goal_text_max_chars must be > 0")
        if self.goal_marker_radius < 0:
            raise ValueError("goal_marker_radius must be >= 0")
        if self.goal_marker_line_width <= 0:
            raise ValueError("goal_marker_line_width must be > 0")
        if str(self.goal_projection_marker) not in ("none", "dot"):
            raise ValueError("goal_projection_marker must be one of: none, dot")
        if int(self.goal_projection_dot_radius) < 0:
            raise ValueError("goal_projection_dot_radius must be >= 0")
        if not (0.0 <= float(self.goal_projection_dot_alpha) <= 1.0):
            raise ValueError("goal_projection_dot_alpha must be in [0,1]")
        for name, v in [
            ("pad_left", self.pad_left),
            ("pad_right", self.pad_right),
            ("pad_top", self.pad_top),
            ("pad_bottom", self.pad_bottom),
        ]:
            if int(v) < 0:
                raise ValueError(f"{name} must be >= 0")


def sector_boundaries_deg(cfg: OverlayConfig) -> list[float]:
    """Return sector boundary angles in degrees (length = sector_count + 1)."""
    n = int(cfg.sector_count)
    a0 = float(cfg.sector_min_bearing_deg)
    a1 = float(cfg.sector_max_bearing_deg)
    step = (a1 - a0) / float(n)
    return [a0 + step * float(i) for i in range(n + 1)]


def sector_id_for_bearing_deg(bearing_deg: float, cfg: OverlayConfig) -> int:
    """Map a bearing (deg) to a sector id in [0, sector_count-1] by fixed angle bins."""
    n = int(cfg.sector_count)
    a0 = float(cfg.sector_min_bearing_deg)
    a1 = float(cfg.sector_max_bearing_deg)
    if n <= 1:
        return 0
    # Clamp to the covered range; keep the upper edge inclusive by nudging with epsilon.
    b = float(bearing_deg)
    if b < a0:
        b = a0
    if b > a1:
        b = a1
    # If b==a1, ensure it lands in the last bin.
    if b >= a1:
        return n - 1
    step = (a1 - a0) / float(n)
    sid = int(math.floor((b - a0) / step))
    return max(0, min(n - 1, sid))


def sector_map_for_record(
    record: PlannerCandidatesRecord, cfg: OverlayConfig
) -> dict[int, list[int]]:
    """Group candidate indices by sector id, using candidate endpoint bearing."""
    out: dict[int, list[int]] = {i: [] for i in range(int(cfg.sector_count))}
    for idx, c in enumerate(record.candidates):
        try:
            end_xy = c.points_xy[-1]
            x = float(end_xy[0])
            y = float(end_xy[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            bearing = math.degrees(math.atan2(y, x)) if (x != 0.0 or y != 0.0) else 0.0
            sid = sector_id_for_bearing_deg(bearing, cfg)
            out.setdefault(int(sid), []).append(int(idx))
        except Exception:
            continue
    return out


def resolve_frame_ref(frame_ref: str, *, episode_dir: Path, dataset_root: Path) -> Path:
    ref_path = Path(frame_ref)
    if ref_path.is_absolute():
        return ref_path

    cand1 = (episode_dir / ref_path).resolve()
    if cand1.exists():
        return cand1

    cand2 = (dataset_root / ref_path).resolve()
    if cand2.exists():
        return cand2

    return cand1


def find_nearest_rgb_record(
    rgb_records: list[RGBRecord], t: float, tol_s: float
) -> RGBRecord | None:
    """Return nearest RGBRecord to time t within tolerance (records should be sorted by t)."""

    if not rgb_records:
        return None
    times = [float(r.t) for r in rgb_records]
    i = bisect_left(times, t)
    best_idx: int | None = None
    best_dist = float("inf")
    for j in (i - 1, i):
        if 0 <= j < len(times):
            dist = abs(times[j] - t)
            if dist < best_dist:
                best_dist = dist
                best_idx = j
    if best_idx is None or best_dist > tol_s:
        return None
    return rgb_records[best_idx]


def _rank_color(rank: int, n: int) -> tuple[int, int, int]:
    """Color for a ranked trajectory.

    VLM-friendly palette: high-contrast, colorblind-friendly (Okabe-Ito-ish),
    then fall back to a larger categorical palette (for up to ~24 trajectories),
    then fall back to HSV spacing if n exceeds that.
    """

    # Small palette (Okabe-Ito-ish).
    palette8 = [
        (230, 159, 0),  # orange
        (86, 180, 233),  # sky blue
        (0, 158, 115),  # bluish green
        (240, 228, 66),  # yellow
        (0, 114, 178),  # blue
        (213, 94, 0),  # vermillion
        (204, 121, 167),  # reddish purple
        (0, 0, 0),  # black (high contrast)
    ]

    # Larger categorical palette (24 distinct-ish colors) tuned for overlays.
    # Source inspiration: Tableau/Matplotlib-style categorical palettes, hand-picked for contrast.
    palette24 = [
        (230, 159, 0),  # orange
        (86, 180, 233),  # sky blue
        (0, 158, 115),  # bluish green
        (240, 228, 66),  # yellow
        (0, 114, 178),  # blue
        (213, 94, 0),  # vermillion
        (204, 121, 167),  # purple
        (0, 0, 0),  # black
        (166, 206, 227),  # light blue
        (31, 120, 180),  # deep blue
        (178, 223, 138),  # light green
        (51, 160, 44),  # green
        (251, 154, 153),  # light red
        (227, 26, 28),  # red
        (253, 191, 111),  # light orange
        (255, 127, 0),  # orange
        (202, 178, 214),  # lavender
        (106, 61, 154),  # purple
        (255, 255, 153),  # pale yellow
        (177, 89, 40),  # brown
        (141, 211, 199),  # teal
        (255, 255, 179),  # cream
        (190, 186, 218),  # periwinkle
        (251, 128, 114),  # salmon
    ]

    if rank < len(palette8) and int(n) <= 8:
        return palette8[rank]
    if rank < len(palette24):
        return palette24[rank]

    # HSV fallback: evenly spaced hues (golden ratio) with high saturation/value.
    # This avoids collapsing into "many blues" when n is large.
    try:
        import colorsys

        phi = 0.618033988749895  # golden ratio conjugate
        h = (float(rank) * phi) % 1.0
        s = 0.78
        v = 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0))
    except Exception:
        # Last resort: keep deterministic grayscale.
        g = int(64 + (191 * (rank % 8)) / 7)
        return g, g, g


def _load_font(size: int) -> ImageFont.ImageFont:
    # Avoid hard deps on system fonts; DejaVuSans is commonly present, but fall back.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=int(size))
    except Exception:
        return ImageFont.load_default()


def _xy_to_px(
    x: float,
    y: float,
    *,
    width: int,
    height: int,
    cfg: OverlayConfig,
) -> tuple[float, float]:
    if cfg.projection == "simple_xy":
        # Robot frame: x forward, y left. Image: x right, y down.
        origin_x = float(cfg.origin_u) * float(width)
        origin_y = float(cfg.origin_v) * float(height)
        px = origin_x - float(y) * float(cfg.px_per_meter)
        py = origin_y - float(x) * float(cfg.px_per_meter)
        return px, py

    # Fisheye projection model matching the front camera calibration.
    # Points are on ground plane: (x, y, z=0), with camera at height camera_height_m.
    # Camera transform uses a fixed rotation matrix R:
    #   cam_x = -y, cam_y = h, cam_z = x
    # then fisheye distortion:
    #   theta = atan(r), theta_d = theta*(1 + k*theta^2), scaling = theta_d/r
    h = float(cfg.camera_height_m)
    cam_z = float(x)
    if cam_z <= 1e-8:
        return float("nan"), float("nan")

    cam_x = -float(y)
    cam_y = h

    xn = cam_x / cam_z
    yn = cam_y / cam_z
    r = math.sqrt(xn * xn + yn * yn)
    theta = math.atan(r)
    k = float(cfg.fisheye_k)
    theta_d = theta * (1.0 + k * theta * theta)
    if r <= 1e-8:
        scaling = 1.0
    else:
        scaling = theta_d / r
    x_dist = xn * scaling
    y_dist = yn * scaling

    # Derive intrinsics if not provided. The planner code used for 960x540:
    #   f = 790/2=395, cx=480, cy=270. We keep the same f/width ratio as a default.
    # IMPORTANT: the defaults must scale with the actual input image size.
    # Reference defaults correspond to width=960, height=540:
    #   fx=395, cx=480, cy=270  =>  fx/width ≈ 0.4114583, cx=0.5*width, cy=0.5*height
    fx_default = (790.0 / 2.0) * (float(width) / 960.0)
    fy_default = fx_default
    cx_default = 0.5 * float(width)
    cy_default = 0.5 * float(height)

    fx = float(cfg.fisheye_fx) if cfg.fisheye_fx is not None else float(fx_default)
    fy = float(cfg.fisheye_fy) if cfg.fisheye_fy is not None else float(fy_default)
    cx = float(cfg.fisheye_cx) if cfg.fisheye_cx is not None else float(cx_default)
    cy = float(cfg.fisheye_cy) if cfg.fisheye_cy is not None else float(cy_default)

    u = fx * x_dist + cx
    v = fy * y_dist + cy
    return u, v


def _safe_int(v: float) -> int:
    if not math.isfinite(v):
        return 0
    return int(round(v))


def _softmax_stable_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    if s <= 0 or (not math.isfinite(float(s))):
        return np.full_like(ex, 1.0 / float(ex.size))
    return ex / s


def _kcenter_endpoints_indices(
    endpoints_xy: np.ndarray, *, start_idx: int, out_k: int
) -> list[int]:
    """Greedy k-center on endpoints to preserve geometric diversity (VLM-friendly)."""
    K = int(endpoints_xy.shape[0])
    if K <= 0:
        return []
    k = int(min(int(out_k), int(K)))
    start = int(max(0, min(int(start_idx), int(K - 1))))

    selected: list[int] = [start]
    selected_set: set[int] = {start}

    for _ in range(1, k):
        best_i: int | None = None
        best_d2 = -1.0
        for i in range(K):
            if i in selected_set:
                continue
            # distance to closest selected endpoint
            min_d2 = float("inf")
            for j in selected:
                dx = float(endpoints_xy[i, 0] - endpoints_xy[j, 0])
                dy = float(endpoints_xy[i, 1] - endpoints_xy[j, 1])
                d2 = dx * dx + dy * dy
                if d2 < min_d2:
                    min_d2 = d2
            if min_d2 > best_d2:
                best_d2 = min_d2
                best_i = int(i)
        if best_i is None:
            best_i = int(selected[-1])
        selected.append(int(best_i))
        selected_set.add(int(best_i))

    return selected


def select_overlay_candidate_indices_and_probs(
    record: PlannerCandidatesRecord,
    cfg: OverlayConfig,
) -> tuple[list[int], list[float] | None]:
    """Return the candidate indices displayed in the overlay, plus a per-candidate normalized prob.

    Notes:
    - Indices are in the original `record.candidates` index space.
    - The returned prob is selection-set-specific (e.g., post-NMS for planner_v2, or softmax over
    the selected set).
      It is intended for the prompt table / sampling baselines, not as a calibrated probability.
    """
    K = len(record.candidates)
    if K <= 0:
        return [], None

    scores = np.asarray([float(c.score) for c in record.candidates], dtype=np.float64)
    ranked = sorted(range(K), key=lambda i: (-float(scores[i]), int(i)))

    selected: list[int] = []
    sel_probs: list[float] | None = None

    if cfg.candidate_set == "planner_v2":
        kept, probs, _err = planner_postprocess_v2(
            record,
            max_trajectories=int(cfg.nms_max_trajectories),
            distance_threshold=float(cfg.nms_distance_threshold),
            prob_threshold=float(cfg.prob_threshold),
        )
        if kept and probs:
            # kept is already sorted by prob desc; apply top_k cap.
            selected = [int(i) for i in kept[: int(cfg.top_k)]]
            sel_probs = [float(p) for p in probs[: len(selected)]]
        else:
            # Fall back to raw ranking if filtering removed everything.
            selected = [int(i) for i in ranked[: int(cfg.top_k)]]
            if selected:
                sel_probs_arr = _softmax_stable_np(scores[np.asarray(selected, dtype=np.int64)])
                sel_probs = [float(x) for x in sel_probs_arr.tolist()]

    elif cfg.candidate_set == "raw_topk":
        for i in ranked:
            if cfg.min_score is not None and float(scores[i]) < float(cfg.min_score):
                continue
            selected.append(int(i))
            if len(selected) >= int(cfg.top_k):
                break
        if selected:
            sel_probs_arr = _softmax_stable_np(scores[np.asarray(selected, dtype=np.int64)])
            sel_probs = [float(x) for x in sel_probs_arr.tolist()]

    elif cfg.candidate_set == "nms_only":
        # Endpoint-distance NMS without probability thresholding. This keeps geometric diversity
        # while still producing a bounded set size for VLM feasibility.
        endpoints = np.asarray([c.points_xy[-1] for c in record.candidates], dtype=np.float64)
        if endpoints.ndim != 2 or endpoints.shape[1] != 2:
            endpoints = np.zeros((int(K), 2), dtype=np.float64)

        keep = _trajectory_nms_endpoints(
            scores=scores,
            endpoints_xy=endpoints,
            max_trajectories=int(cfg.nms_max_trajectories),
            distance_threshold=float(cfg.nms_distance_threshold),
        )
        # Keep is padded to nms_max_trajectories; apply top_k cap for display.
        selected = [int(i) for i in keep[: int(cfg.top_k)]] if keep.size else [int(ranked[0])]
        if selected:
            sel_probs_arr = _softmax_stable_np(scores[np.asarray(selected, dtype=np.int64)])
            sel_probs = [float(x) for x in sel_probs_arr.tolist()]

    elif cfg.candidate_set == "kcenter_endpoints":
        # Use endpoint-only diversity; start from the raw score argmax for stability.
        endpoints = np.asarray([c.points_xy[-1] for c in record.candidates], dtype=np.float64)
        if endpoints.ndim != 2 or endpoints.shape[1] != 2:
            endpoints = np.zeros((int(K), 2), dtype=np.float64)
        start_idx = int(ranked[0]) if ranked else 0
        selected = _kcenter_endpoints_indices(endpoints, start_idx=start_idx, out_k=int(cfg.top_k))
        if selected:
            sel_probs_arr = _softmax_stable_np(scores[np.asarray(selected, dtype=np.int64)])
            sel_probs = [float(x) for x in sel_probs_arr.tolist()]

    else:
        # Should be unreachable due to OverlayConfig validation; keep a safe fallback.
        selected = [int(i) for i in ranked[: int(cfg.top_k)]]
        if selected:
            sel_probs_arr = _softmax_stable_np(scores[np.asarray(selected, dtype=np.int64)])
            sel_probs = [float(x) for x in sel_probs_arr.tolist()]

    return selected, sel_probs


def render_overlay_pil(
    *,
    base_image: Image.Image,
    record: PlannerCandidatesRecord,
    cfg: OverlayConfig,
    selected_indices: list[int] | None = None,
    cluster_groups: dict[int, list[int]] | None = None,
    label_indices_subset: list[int] | None = None,
    cluster_member_style: str | None = None,  # None|"lines"|"faint_lines"|"endpoints"
    pred_index: int | None = None,
    label_index: int | None = None,
    goal_text: str | None = None,
    goal_xy: tuple[float, float] | None = None,
) -> Image.Image:
    """Render planner candidate trajectories onto an image (PIL backend)."""

    base = base_image.convert("RGBA")
    base_w, base_h = base.size

    out_w = int(base_w) + int(cfg.pad_left) + int(cfg.pad_right)
    out_h = int(base_h) + int(cfg.pad_top) + int(cfg.pad_bottom)

    # Pad with a solid background so projected points outside the original frame are still visible.
    img = Image.new("RGBA", (out_w, out_h), (cfg.pad_rgb[0], cfg.pad_rgb[1], cfg.pad_rgb[2], 255))
    img.alpha_composite(base, (int(cfg.pad_left), int(cfg.pad_top)))

    overlay = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = _load_font(int(cfg.label_font_size))
    goal_font = _load_font(int(cfg.goal_text_font_size))

    K = len(record.candidates)
    if K == 0:
        return img

    if selected_indices is None:
        selected, _sel_probs = select_overlay_candidate_indices_and_probs(record, cfg)
    else:
        # Allow callers (e.g., hierarchical prompting) to render a custom subset of candidates
        # while keeping labels in the ORIGINAL planner index space.
        selected = [int(i) for i in selected_indices]
        _sel_probs = None
        # Filter invalid indices (best-effort).
        selected = [int(i) for i in selected if 0 <= int(i) < int(K)]

    # Ensure highlighted indices are included (even if not top-k).
    for idx in (label_index, pred_index):
        if idx is None:
            continue
        if 0 <= int(idx) < K and int(idx) not in selected:
            selected.append(int(idx))

    alpha = int(round(float(cfg.alpha) * 255.0))
    label_bg_alpha = int(round(float(cfg.label_bg_alpha) * 255.0))

    # --- Geometry helpers (pixel space, after padding) ---
    def _inside(u: float, v: float) -> bool:
        return 0.0 <= float(u) < float(out_w) and 0.0 <= float(v) < float(out_h)

    def _clamp_to_rect(u: float, v: float) -> tuple[float, float]:
        uu = float(min(max(float(u), 0.0), float(out_w - 1)))
        vv = float(min(max(float(v), 0.0), float(out_h - 1)))
        return uu, vv

    def _segment_rect_intersection(
        p0: tuple[float, float],
        p1: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Return intersection of segment p0->p1 with the image bounds.

        Uses a simple parametric intersection with the 4 rectangle edges.
        Returns the intersection point closest to p1 along the segment.
        """

        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        dx = x1 - x0
        dy = y1 - y0
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return None

        xmin = 0.0
        ymin = 0.0
        xmax = float(out_w - 1)
        ymax = float(out_h - 1)

        hits: list[tuple[float, float, float]] = []  # (t, x, y)

        def _try_add(t: float) -> None:
            if not (0.0 <= t <= 1.0):
                return
            x = x0 + t * dx
            y = y0 + t * dy
            if xmin - 1e-6 <= x <= xmax + 1e-6 and ymin - 1e-6 <= y <= ymax + 1e-6:
                hits.append((float(t), float(x), float(y)))

        # Intersect with vertical edges x=xmin/xmax
        if abs(dx) > 1e-12:
            _try_add((xmin - x0) / dx)
            _try_add((xmax - x0) / dx)
        # Intersect with horizontal edges y=ymin/ymax
        if abs(dy) > 1e-12:
            _try_add((ymin - y0) / dy)
            _try_add((ymax - y0) / dy)

        if not hits:
            return None
        # Choose the hit closest to p1 => largest t.
        hits.sort(key=lambda txy: txy[0])
        _t, hx, hy = hits[-1]
        return float(hx), float(hy)

    def _label_anchor_for_polyline(px_all: list[tuple[float, float]]) -> tuple[int, int]:
        """Choose a visible label anchor for a polyline.

        - If endpoint is in-bounds, label it.
        - Else, place label at the intersection of the endpoint segment with the image rectangle,
          so labels remain visible even when trajectories go out-of-frame.
        - Else fall back to clamping the endpoint.
        """

        if not px_all:
            return 0, 0
        end = px_all[-1]
        if _inside(end[0], end[1]):
            return _safe_int(end[0]), _safe_int(end[1])

        # Walk backwards to find a segment that crosses into/out of the rectangle.
        for j in range(len(px_all) - 2, -1, -1):
            prev = px_all[j]
            if not (math.isfinite(prev[0]) and math.isfinite(prev[1])):
                continue
            inter = _segment_rect_intersection(prev, end)
            if inter is not None:
                u, v = inter
                u2, v2 = _clamp_to_rect(u, v)
                return _safe_int(u2), _safe_int(v2)

        # Fallback: clamp the endpoint.
        u2, v2 = _clamp_to_rect(end[0], end[1])
        return _safe_int(u2), _safe_int(v2)

    # Collect label placements while drawing lines; render labels in a second pass so they
    # always appear on top of trajectories (better for VLM input & readability).
    label_positions: list[tuple[int, int, str]] = []  # (u_px, v_px, text)
    # For each displayed candidate index, store the *endpoint anchor* pixel (in-bounds).
    # This lets us draw endpoint markers and leader lines when labels are nudged for readability.
    label_anchor_by_idx: dict[int, tuple[int, int]] = {}
    # Candidates that ended up with no drawable points (either out of bounds or unprojectable).
    # Kept for backward compatibility; normally empty because we clamp labels into-bounds.
    missing_label_indices: list[tuple[int, float | None]] = []
    label_subset_set: set[int] | None = None
    if isinstance(label_indices_subset, list):
        try:
            label_subset_set = {int(x) for x in label_indices_subset}
        except Exception:
            label_subset_set = None

    reps_set: set[int] = set()
    if isinstance(cluster_groups, dict) and cluster_groups:
        try:
            reps_set = {int(r) for r in cluster_groups.keys()}
        except Exception:
            reps_set = set()

    # Default cluster-member rendering:
    # - if cluster_groups is used (hierarchical rep stage), make the visualization readable by
    # default.
    #   Most useful: show rep centerlines and member endpoints (no spaghetti).
    member_style = str(cluster_member_style) if isinstance(cluster_member_style, str) else None
    if member_style is None and isinstance(cluster_groups, dict) and cluster_groups:
        member_style = "endpoints"

    def _draw_index_label_at(draw_ctx: ImageDraw.ImageDraw, u: int, v: int, text: str) -> None:
        # Center the label at (u,v).
        #
        # IMPORTANT: PIL's textbbox can return negative offsets (depending on font and glyph),
        # so we must account for bbox[0]/bbox[1] when placing text, otherwise the text will
        # look "misaligned" inside the background box.
        pad = 3
        try:
            bx0, by0, bx1, by1 = draw_ctx.textbbox((0, 0), text, font=font)
            tw = float(bx1 - bx0)
            th = float(by1 - by0)
        except Exception:
            # Fallback: approximate bbox with textlength/fontsize (no negative offsets available).
            bx0, by0 = 0.0, 0.0
            tw = float(draw_ctx.textlength(text, font=font))
            th = float(cfg.label_font_size)

        # Compute box corners around the *glyph bbox* (not the anchor point).
        x0 = float(u) - tw / 2.0 - float(pad)
        y0 = float(v) - th / 2.0 - float(pad)
        x1 = float(u) + tw / 2.0 + float(pad)
        y1 = float(v) + th / 2.0 + float(pad)

        # Background box.
        draw_ctx.rectangle(
            (_safe_int(x0), _safe_int(y0), _safe_int(x1), _safe_int(y1)),
            fill=(cfg.label_bg_rgb[0], cfg.label_bg_rgb[1], cfg.label_bg_rgb[2], label_bg_alpha),
            outline=(0, 0, 0, min(255, label_bg_alpha + 60)),
            width=1,
        )

        # Place text so that its bbox is centered at (u,v).
        text_x = float(u) - tw / 2.0 - float(bx0)
        text_y = float(v) - th / 2.0 - float(by0)
        draw_ctx.text(
            (_safe_int(text_x), _safe_int(text_y)),
            text,
            fill=(cfg.label_text_rgb[0], cfg.label_text_rgb[1], cfg.label_text_rgb[2], 255),
            font=font,
        )

    # Optional: draw a sector "fan" overlay to make sector-based preconditions human/VLM-readable.
    # This draws fixed-angle boundary rays and labels S0..S(N-1) along the bottom of the image.
    if bool(cfg.draw_sectors):
        try:
            # Bottom edge of the *camera image* within the padded canvas (not including pad_bottom).
            v_img_bottom = int(cfg.pad_top) + int(base_h) - 1
            v_img_bottom = max(0, min(int(v_img_bottom), int(out_h - 1)))

            bounds_deg = sector_boundaries_deg(cfg)
            line_a = int(round(float(cfg.sector_line_alpha) * 255.0))
            line_col = (
                cfg.sector_line_rgb[0],
                cfg.sector_line_rgb[1],
                cfg.sector_line_rgb[2],
                line_a,
            )

            d0 = float(cfg.sector_ray_min_m)
            d1 = float(cfg.sector_ray_max_m)
            steps = int(cfg.sector_ray_steps)
            distances = [
                d0 + (d1 - d0) * (float(i) / float(max(1, steps - 1))) for i in range(steps)
            ]

            def _ray_points_px(angle_deg: float) -> list[tuple[float, float]]:
                th = math.radians(float(angle_deg))
                pts: list[tuple[float, float]] = []
                for d in distances:
                    x = float(d) * math.cos(th)
                    y = float(d) * math.sin(th)
                    u0, v0 = _xy_to_px(x, y, width=base_w, height=base_h, cfg=cfg)
                    if not (math.isfinite(u0) and math.isfinite(v0)):
                        continue
                    u2 = float(u0) + float(cfg.pad_left)
                    v2 = float(v0) + float(cfg.pad_top)
                    if not (math.isfinite(u2) and math.isfinite(v2)):
                        continue
                    pts.append((u2, v2))
                if pts:
                    # Extend the ray to the bottom edge of the camera image for a clear origin.
                    u_start, v_start = pts[0]
                    u_start = float(min(max(u_start, 0.0), float(out_w - 1)))
                    if abs(float(v_img_bottom) - float(v_start)) > 3.0:
                        pts.insert(0, (u_start, float(v_img_bottom)))
                return pts

            # Boundary rays.
            for a in bounds_deg:
                pts = _ray_points_px(float(a))
                if len(pts) >= 2:
                    draw.line(
                        [(_safe_int(u), _safe_int(v)) for u, v in pts],
                        fill=line_col,
                        width=int(cfg.sector_line_width),
                    )

            # Sector labels along the bottom, centered on each sector.
            if bool(cfg.sector_label):
                for sid in range(int(cfg.sector_count)):
                    a0 = float(bounds_deg[sid])
                    a1 = float(bounds_deg[sid + 1])
                    amid = 0.5 * (a0 + a1)
                    pts_mid = _ray_points_px(amid)
                    if pts_mid:
                        u_mid = float(pts_mid[0][0])
                    else:
                        # Fallback: simple monotonic placement across the width.
                        frac = float(sid + 0.5) / float(max(1, int(cfg.sector_count)))
                        u_mid = float(frac) * float(out_w)

                    u_mid = float(min(max(u_mid, 0.0), float(out_w - 1)))
                    if int(cfg.pad_bottom) > 0:
                        # Prefer placing labels in the padded bottom strip (avoid occluding the
                        # camera image).
                        v_strip_top = int(cfg.pad_top) + int(base_h)
                        v_off = min(int(cfg.pad_bottom) - 1, int(cfg.sector_label_margin_px))
                        v_lab = v_strip_top + max(0, int(v_off))
                    else:
                        v_lab = int(v_img_bottom) - int(cfg.sector_label_margin_px)
                    v_lab = max(0, min(int(v_lab), int(out_h - 1)))
                    _draw_index_label_at(draw, _safe_int(u_mid), int(v_lab), f"S{sid}")
        except Exception:
            # Never fail overlay rendering due to the sector visualization.
            pass

    def draw_points_list(
        pts: list[tuple[float, float]],
        *,
        rgb: tuple[int, int, int],
        width_px: int,
        label: str | None = None,
        line_stipple: bool = False,
    ) -> None:
        if not pts:
            return

        px_pts = []
        for j, (x, y) in enumerate(pts):
            if cfg.draw_points and (j % int(cfg.point_every) == 0):
                px, py = _xy_to_px(x, y, width=base_w, height=base_h, cfg=cfg)
                if not (math.isfinite(px) and math.isfinite(py)):
                    continue
                px2 = px + float(cfg.pad_left)
                py2 = py + float(cfg.pad_top)
                if not (0 <= px2 < out_w and 0 <= py2 < out_h):
                    continue
                px_pts.append((px2, py2))
                if cfg.point_radius > 0:
                    r = int(cfg.point_radius)
                    draw.ellipse(
                        (
                            _safe_int(px2) - r,
                            _safe_int(py2) - r,
                            _safe_int(px2) + r,
                            _safe_int(py2) + r,
                        ),
                        fill=(rgb[0], rgb[1], rgb[2], alpha),
                        outline=None,
                    )
            else:
                px, py = _xy_to_px(x, y, width=base_w, height=base_h, cfg=cfg)
                if not (math.isfinite(px) and math.isfinite(py)):
                    continue
                px2 = px + float(cfg.pad_left)
                py2 = py + float(cfg.pad_top)
                if not (0 <= px2 < out_w and 0 <= py2 < out_h):
                    continue
                px_pts.append((px2, py2))

        if len(px_pts) >= 2:
            # Note: PIL draw.line doesn't support stippling directly.
            # We can simulate dot/dash if needed, but solid is fine for now.
            draw.line(
                [(_safe_int(x), _safe_int(y)) for x, y in px_pts],
                fill=(rgb[0], rgb[1], rgb[2], alpha),
                width=int(width_px),
                joint="curve",
            )

        if label and px_pts:
            endx, endy = px_pts[-1]
            # Draw label immediately for auxiliary trajectories
            _draw_index_label_at(draw, _safe_int(endx), _safe_int(endy), label)

    def _traj_length_m(pts_xy: list[tuple[float, float]] | list[list[float]]) -> float:
        """Path length in meters (robot frame), best-effort."""
        try:
            if not pts_xy or len(pts_xy) < 2:
                return 0.0
            s = 0.0
            for j in range(1, len(pts_xy)):
                x0, y0 = float(pts_xy[j - 1][0]), float(pts_xy[j - 1][1])
                x1, y1 = float(pts_xy[j][0]), float(pts_xy[j][1])
                dx = x1 - x0
                dy = y1 - y0
                d = math.hypot(dx, dy)
                if math.isfinite(d):
                    s += float(d)
            return float(s)
        except Exception:
            return 0.0

    def _offset_corridor_xy(
        xy: list[list[float]] | list[tuple[float, float]],
        *,
        half_width_m: float,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Compute left/right offset polylines in robot frame (x forward, y left)."""
        if half_width_m <= 0 or not math.isfinite(float(half_width_m)):
            return [], []
        if len(xy) < 2:
            return [], []
        left: list[tuple[float, float]] = []
        right: list[tuple[float, float]] = []
        last_nx = last_ny = None
        for i in range(len(xy) - 1):
            x0, y0 = float(xy[i][0]), float(xy[i][1])
            x1, y1 = float(xy[i + 1][0]), float(xy[i + 1][1])
            dx = x1 - x0
            dy = y1 - y0
            norm = math.hypot(dx, dy)
            if norm < 1e-6 or (not math.isfinite(norm)):
                continue
            # Left normal of the segment direction: normal = [-dy, dx] / ||dir||
            nx = -dy / norm
            ny = dx / norm
            last_nx, last_ny = float(nx), float(ny)
            left.append((x0 + half_width_m * nx, y0 + half_width_m * ny))
            right.append((x0 - half_width_m * nx, y0 - half_width_m * ny))
        if left and last_nx is not None and last_ny is not None:
            xl, yl = float(xy[-1][0]), float(xy[-1][1])
            left.append((xl + half_width_m * float(last_nx), yl + half_width_m * float(last_ny)))
            right.append((xl - half_width_m * float(last_nx), yl - half_width_m * float(last_ny)))
        return left, right

    def _extend_to_bottom(px_pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Extend the start of a polyline to the bottom edge (visual anchoring)."""
        if not px_pts or len(px_pts) < 1:
            return px_pts
        if not bool(cfg.extend_start_to_bottom):
            return px_pts
        try:
            v_edge = float(out_h - 1)
            u_edge: float | None = None
            if len(px_pts) >= 2:
                u0, v0 = float(px_pts[0][0]), float(px_pts[0][1])
                u1, v1 = float(px_pts[1][0]), float(px_pts[1][1])
                denom = v0 - v1
                if abs(denom) > 1e-6:
                    t = (v_edge - v0) / denom  # along (p0 - p1)
                    if t >= 0.0 and math.isfinite(t):
                        u_edge = u0 + t * (u0 - u1)
            if u_edge is None:
                u_edge = float(px_pts[0][0])
            u_edge = float(min(max(u_edge, 0.0), float(out_w - 1)))
            if abs(v_edge - float(px_pts[0][1])) > 3.0:
                return [(float(u_edge), v_edge)] + list(px_pts)
            return px_pts
        except Exception:
            return px_pts

    # Precompute per-trajectory geometry so we can render in multiple passes:
    # 1) corridors (background), 2) centerlines, 3) dots, 4) labels.
    traj_geom: dict[int, dict[str, object]] = {}
    for idx in selected:
        try:
            pts_xy = record.candidates[int(idx)].points_xy
        except Exception:
            pts_xy = None
        if not pts_xy:
            traj_geom[int(idx)] = {
                "len_m": 0.0,
                "px_line": [],
                "px_all": [],
                "markers": [],
                "corridor": None,
            }
            continue

        px_line: list[tuple[float, float]] = []
        px_all: list[tuple[float, float]] = []
        markers: list[tuple[float, float]] = []

        for j, (x, y) in enumerate(pts_xy):
            u, v = _xy_to_px(float(x), float(y), width=base_w, height=base_h, cfg=cfg)
            if not (math.isfinite(u) and math.isfinite(v)):
                continue
            u2 = float(u) + float(cfg.pad_left)
            v2 = float(v) + float(cfg.pad_top)
            px_all.append((u2, v2))
            if 0 <= u2 < out_w and 0 <= v2 < out_h:
                px_line.append((u2, v2))
                draw_pts = bool(cfg.draw_points)
                if member_style == "endpoints" and int(idx) not in reps_set:
                    draw_pts = False
                if draw_pts and (j % int(cfg.point_every) == 0):
                    markers.append((u2, v2))

        px_line = _extend_to_bottom(px_line)

        # Corridor geometry (projected left/right boundaries + polygon).
        corridor = None
        if str(cfg.traj_style) == "corridor" and len(pts_xy) >= 2:
            half_w = 0.5 * float(cfg.robot_width_m)
            left_xy, right_xy = _offset_corridor_xy(pts_xy, half_width_m=half_w)
            left_px: list[tuple[float, float]] = []
            right_px: list[tuple[float, float]] = []
            for k in range(min(len(left_xy), len(right_xy))):
                lx, ly = left_xy[k]
                rx, ry = right_xy[k]
                lu, lv = _xy_to_px(float(lx), float(ly), width=base_w, height=base_h, cfg=cfg)
                ru, rv = _xy_to_px(float(rx), float(ry), width=base_w, height=base_h, cfg=cfg)
                if not (
                    math.isfinite(lu)
                    and math.isfinite(lv)
                    and math.isfinite(ru)
                    and math.isfinite(rv)
                ):
                    continue
                left_px.append((float(lu) + float(cfg.pad_left), float(lv) + float(cfg.pad_top)))
                right_px.append((float(ru) + float(cfg.pad_left), float(rv) + float(cfg.pad_top)))
            # Match the "line" style behavior: visually anchor the trajectory to the bottom edge
            # even when the first projected point begins above the bottom of the image.
            #
            # For corridor rendering, extend BOTH boundary polylines so the filled corridor also
            # reaches the bottom edge (otherwise the corridor can appear to "start in mid-air").
            left_px = _extend_to_bottom(left_px)
            right_px = _extend_to_bottom(right_px)
            if len(left_px) >= 2 and len(right_px) >= 2:
                corridor = {"left_px": left_px, "right_px": right_px}

        traj_geom[int(idx)] = {
            "len_m": _traj_length_m(pts_xy),
            "px_line": px_line,
            "px_all": px_all,
            "markers": markers,
            "corridor": corridor,
        }

        # Always compute an endpoint anchor if possible (used for endpoint dots).
        if px_all:
            ex, ey = _label_anchor_for_polyline(px_all)
            label_anchor_by_idx[int(idx)] = (int(ex), int(ey))
            # Labels are drawn in a second pass on top of all lines.
            # Use original candidate index for the label (matches the candidate_confidence table).
            # IMPORTANT: when label_indices_subset is provided, we intentionally suppress labels
            # for non-subset trajectories (e.g., hierarchical rep-stage overlays). Do NOT treat
            # them as "missing"; otherwise they can show up in the fallback bottom-strip and
            # clutter the image.
            if cfg.label_indices and (label_subset_set is None or int(idx) in label_subset_set):
                label_positions.append((int(ex), int(ey), str(idx)))

    # Draw ordering: longest first => shorter ends up on top (less occlusion).
    selected_by_len = sorted(
        [int(i) for i in selected],
        key=lambda i: (-float(traj_geom.get(int(i), {}).get("len_m", 0.0) or 0.0), int(i)),
    )

    # Color assignment should remain rank-based (planner ordering), not length-based.
    idx_to_color: dict[int, tuple[int, int, int]] = {}
    if isinstance(cluster_groups, dict) and cluster_groups:
        # Hierarchical rep stage: color ALL trajectories within the same cluster with the same
        # color.
        # cluster_groups: rep_idx -> [member_idx...]
        try:
            reps = [int(r) for r in cluster_groups.keys()]
            reps.sort()
            rep_to_color: dict[int, tuple[int, int, int]] = {}
            for rank, r in enumerate(reps):
                rep_to_color[int(r)] = _rank_color(int(rank), max(1, len(reps)))
            member_to_rep: dict[int, int] = {}
            for r, members in cluster_groups.items():
                rr = int(r)
                if not isinstance(members, list):
                    continue
                for m in members:
                    try:
                        mi = int(m)
                    except Exception:
                        continue
                    member_to_rep[int(mi)] = rr
                member_to_rep.setdefault(rr, rr)
            for idx in selected:
                rep = member_to_rep.get(int(idx), int(idx))
                idx_to_color[int(idx)] = rep_to_color.get(int(rep), _rank_color(0, 1))
        except Exception:
            # Fallback to rank-based coloring.
            for rank, idx in enumerate(selected):
                idx_to_color[int(idx)] = _rank_color(int(rank), len(selected))
    else:
        for rank, idx in enumerate(selected):
            idx_to_color[int(idx)] = _rank_color(int(rank), len(selected))

    # --- Pass 1: corridor backgrounds (if enabled) ---
    if str(cfg.traj_style) == "corridor" and member_style != "endpoints":
        fill_a = int(round(float(cfg.corridor_alpha) * 255.0))
        edge_a = min(255, fill_a + 60)
        for idx in selected_by_len:
            if member_style == "endpoints" and int(idx) not in reps_set:
                continue
            color = idx_to_color.get(int(idx), (255, 255, 255))
            g = traj_geom.get(int(idx), {})
            cor = g.get("corridor") if isinstance(g, dict) else None
            if not isinstance(cor, dict):
                continue
            left_px = cor.get("left_px")
            right_px = cor.get("right_px")
            if not (isinstance(left_px, list) and isinstance(right_px, list)):
                continue
            if len(left_px) < 2 or len(right_px) < 2:
                continue
            poly = list(left_px) + list(reversed(right_px))
            draw.polygon(
                [(_safe_int(u), _safe_int(v)) for (u, v) in poly],
                fill=(color[0], color[1], color[2], fill_a),
            )
            if bool(cfg.corridor_outline):
                edge_w = max(1, int(max(1, int(cfg.line_width) // 2)))
                draw.line(
                    [(_safe_int(u), _safe_int(v)) for (u, v) in left_px],
                    fill=(color[0], color[1], color[2], edge_a),
                    width=int(edge_w),
                    joint="curve",
                )
                draw.line(
                    [(_safe_int(u), _safe_int(v)) for (u, v) in right_px],
                    fill=(color[0], color[1], color[2], edge_a),
                    width=int(edge_w),
                    joint="curve",
                )

    # --- Pass 2: colored centerlines (always draw, independent of corridor) ---
    for idx in selected_by_len:
        if member_style == "endpoints" and int(idx) not in reps_set:
            continue
        color = idx_to_color.get(int(idx), (255, 255, 255))
        g = traj_geom.get(int(idx), {})
        px_line = g.get("px_line") if isinstance(g, dict) else None
        if not isinstance(px_line, list) or len(px_line) < 2:
            continue
        this_alpha = int(alpha)
        if member_style == "faint_lines" and int(idx) not in reps_set:
            # Hierarchical rep-stage visualization often downsamples (e.g., grid mode),
            # so very faint lines can disappear. Keep members clearly visible while
            # still de-emphasizing them relative to reps.
            this_alpha = max(90, int(round(0.62 * float(alpha))))

        # Use a thicker stroke for cluster representatives so they stand out.
        w_main = int(cfg.highlight_line_width) if int(idx) in reps_set else int(cfg.line_width)
        w_main = max(1, int(w_main))

        # Add a thin black outline behind each trajectory for contrast against clutter/background.
        try:
            w_outline = max(1, int(w_main) + (2 if int(idx) in reps_set else 1))
            a_outline = min(255, int(this_alpha) + (30 if member_style == "faint_lines" else 20))
            draw.line(
                [(_safe_int(x), _safe_int(y)) for x, y in px_line],
                fill=(0, 0, 0, int(a_outline)),
                width=int(w_outline),
                joint="curve",
            )
        except Exception:
            pass
        draw.line(
            [(_safe_int(x), _safe_int(y)) for x, y in px_line],
            fill=(color[0], color[1], color[2], int(this_alpha)),
            width=int(w_main),
            joint="curve",
        )

    # --- Pass 3: dots (always on top of corridors/lines) ---
    if cfg.point_radius > 0:
        r = int(cfg.point_radius)
        for idx in selected_by_len:
            if member_style == "endpoints" and int(idx) not in reps_set:
                continue
            color = idx_to_color.get(int(idx), (255, 255, 255))
            g = traj_geom.get(int(idx), {})
            markers = g.get("markers") if isinstance(g, dict) else None
            if not isinstance(markers, list) or not markers:
                continue
            for px2, py2 in markers:
                draw.ellipse(
                    (
                        _safe_int(px2) - r,
                        _safe_int(py2) - r,
                        _safe_int(px2) + r,
                        _safe_int(py2) + r,
                    ),
                    fill=(color[0], color[1], color[2], alpha),
                    outline=None,
                )

    # --- Pass 3.2: endpoint markers (make endpoints unambiguous) ---
    # Draw a small filled marker at each trajectory endpoint anchor (same color as the trajectory).
    try:
        if label_anchor_by_idx:
            rr = max(4, int(round(float(cfg.label_font_size) * 0.25)))
            rr = int(min(max(rr, 4), 9))
            for idx in selected_by_len:
                if int(idx) not in label_anchor_by_idx:
                    continue
                u_end, v_end = label_anchor_by_idx[int(idx)]
                color = idx_to_color.get(int(idx), (255, 255, 255))
                # Slightly more opaque than line alpha for visibility.
                rr2 = rr
                a_fill = min(255, int(alpha) + 80)
                if member_style == "faint_lines" and int(idx) not in reps_set:
                    # In rep-stage grid overlays, endpoints can dominate after downsampling.
                    # Keep them smaller and less opaque so the polylines remain the primary cue.
                    rr2 = max(3, int(rr) - 2)
                    a_fill = min(200, int(alpha) + 30)
                elif member_style == "endpoints" and int(idx) not in reps_set:
                    a_fill = min(255, 150)
                draw.ellipse(
                    (
                        int(u_end) - int(rr2),
                        int(v_end) - int(rr2),
                        int(u_end) + int(rr2),
                        int(v_end) + int(rr2),
                    ),
                    fill=(color[0], color[1], color[2], a_fill),
                    outline=(0, 0, 0, min(255, a_fill + 60)),
                    width=2,
                )
    except Exception:
        pass

    # Highlight label and prediction (draw on top of everything else).
    def _draw_highlight(idx: int, *, rgb: tuple[int, int, int]) -> None:
        g = traj_geom.get(int(idx), {})
        if not isinstance(g, dict):
            return
        px_line = g.get("px_line")
        if isinstance(px_line, list) and len(px_line) >= 2:
            draw.line(
                [(_safe_int(x), _safe_int(y)) for x, y in px_line],
                fill=(rgb[0], rgb[1], rgb[2], min(255, alpha + 40)),
                width=int(cfg.highlight_line_width),
                joint="curve",
            )
        markers = g.get("markers")
        if cfg.point_radius > 0 and isinstance(markers, list) and markers:
            rr = max(2, int(cfg.point_radius) + 1)
            for px2, py2 in markers:
                draw.ellipse(
                    (
                        _safe_int(px2) - rr,
                        _safe_int(py2) - rr,
                        _safe_int(px2) + rr,
                        _safe_int(py2) + rr,
                    ),
                    fill=(rgb[0], rgb[1], rgb[2], min(255, alpha + 40)),
                    outline=None,
                )

    if cfg.highlight_label and label_index is not None and 0 <= int(label_index) < K:
        _draw_highlight(int(label_index), rgb=(255, 255, 0))
    if cfg.highlight_pred and pred_index is not None and 0 <= int(pred_index) < K:
        _draw_highlight(int(pred_index), rgb=(0, 255, 0))

    out = Image.alpha_composite(img, overlay)

    # Draw goal banner early (top). The goal marker/direction arrow is drawn later (on top of
    # labels)
    # so it remains visually salient for VLM prompting.
    goal_draw = ImageDraw.Draw(out, "RGBA")

    # --- Goal banner text (e.g., episode.static_context.goal_description) ---
    if cfg.draw_goal_text:
        # If no explicit goal text is provided, fall back to displaying the numeric goal vector
        # (this is the most important "what goal did the planner receive?" signal for debugging).
        text_src: str | None = None
        if goal_text:
            text_src = str(goal_text)
        elif goal_xy is not None:
            try:
                gx = float(goal_xy[0])
                gy = float(goal_xy[1])
                dist = math.hypot(gx, gy)
                text_src = f"goal_xy=({gx:.2f},{gy:.2f})  dist={dist:.2f}m"
            except Exception:
                text_src = None

        text = (text_src or "").strip().replace("\n", " ")
        if text:
            if len(text) > int(cfg.goal_text_max_chars):
                text = text[: int(cfg.goal_text_max_chars) - 1] + "…"
            pad_x = 8
            pad_y = 6
            try:
                bbox = goal_draw.textbbox((0, 0), text, font=goal_font)
                tw = int(bbox[2] - bbox[0])
                th = int(bbox[3] - bbox[1])
            except Exception:
                tw = int(goal_draw.textlength(text, font=goal_font))
                th = int(cfg.goal_text_font_size) + 6
            x0 = 0
            y0 = 0
            x1 = min(out_w, tw + pad_x * 2)
            y1 = min(out_h, th + pad_y * 2)
            bg_a = int(round(float(cfg.goal_text_bg_alpha) * 255.0))
            goal_draw.rectangle(
                (x0, y0, x1, y1),
                fill=(
                    cfg.goal_text_bg_rgb[0],
                    cfg.goal_text_bg_rgb[1],
                    cfg.goal_text_bg_rgb[2],
                    bg_a,
                ),
            )
            goal_draw.text(
                (pad_x, pad_y),
                text,
                fill=(cfg.goal_text_rgb[0], cfg.goal_text_rgb[1], cfg.goal_text_rgb[2], 255),
                font=goal_font,
            )

    # --- Goal marker (robot-frame xy projected to pixels) ---
    # If not explicitly provided, try to read from extra fields in the planner_candidates record.
    if cfg.draw_goal_marker and goal_xy is None:
        # Support a few common names.
        for key in ("goal_xy", "goal_point_xy", "goal_point"):
            v = getattr(record, key, None)
            if v is None:
                continue
            try:
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    goal_xy = (float(v[0]), float(v[1]))
                    break
            except Exception:
                continue

    # Pre-pass: when using the raised "goal arrow" visualization, draw the dashed "hanging line"
    # (and an optional tiny ground anchor marker) BEFORE candidate endpoint labels, so labels
    # remain readable.
    # The arrowhead + GOAL label are drawn later (on top).
    if goal_xy is not None and bool(getattr(cfg, "draw_goal_direction_arrow", False)):
        try:
            gx = float(goal_xy[0])
            gy = float(goal_xy[1])

            col = (cfg.goal_marker_rgb[0], cfg.goal_marker_rgb[1], cfg.goal_marker_rgb[2], 255)

            def _ray_to_border(
                x0: float, y0: float, x1: float, y1: float, *, width: int, height: int
            ) -> tuple[int, int]:
                """Intersect ray (x0,y0) -> (x1,y1) with the image rectangle border."""
                dx = float(x1 - x0)
                dy = float(y1 - y0)
                if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                    return int(min(max(round(x0), 0), width - 1)), int(
                        min(max(round(y0), 0), height - 1)
                    )

                candidates: list[tuple[float, float, float]] = []  # (t, x, y)
                x_min = 0.0
                x_max = float(width - 1)
                y_min = 0.0
                y_max = float(height - 1)

                # Intersections with vertical borders.
                if abs(dx) >= 1e-9:
                    for xb in (x_min, x_max):
                        t = (xb - float(x0)) / dx
                        if t > 0:
                            y = float(y0) + t * dy
                            if y_min - 1e-6 <= y <= y_max + 1e-6:
                                candidates.append((t, xb, y))
                # Intersections with horizontal borders.
                if abs(dy) >= 1e-9:
                    for yb in (y_min, y_max):
                        t = (yb - float(y0)) / dy
                        if t > 0:
                            x = float(x0) + t * dx
                            if x_min - 1e-6 <= x <= x_max + 1e-6:
                                candidates.append((t, x, yb))

                if not candidates:
                    # Fallback: clamp the destination point.
                    bx = int(min(max(round(x1), 0), width - 1))
                    by = int(min(max(round(y1), 0), height - 1))
                    return bx, by

                _t, xb_best, yb_best = min(candidates, key=lambda a: a[0])
                bx = int(min(max(round(xb_best), 0), width - 1))
                by = int(min(max(round(yb_best), 0), height - 1))
                return bx, by

            # Compute goal anchor (projection onto image plane; if off-screen, clamp to border).
            anchor_x: float | None = None
            anchor_y: float | None = None
            gu, gv = _xy_to_px(gx, gy, width=base_w, height=base_h, cfg=cfg)
            if math.isfinite(gu) and math.isfinite(gv):
                gu2 = float(gu) + float(cfg.pad_left)
                gv2 = float(gv) + float(cfg.pad_top)
                if 0 <= gu2 < out_w and 0 <= gv2 < out_h:
                    anchor_x, anchor_y = float(gu2), float(gv2)
                else:
                    cx0 = float(out_w) / 2.0
                    cy0 = float(out_h) / 2.0
                    bx, by = _ray_to_border(
                        cx0, cy0, float(gu2), float(gv2), width=out_w, height=out_h
                    )
                    anchor_x, anchor_y = float(bx), float(by)
            else:
                # Projection failed (e.g., goal behind camera in fisheye model): show direction on
                # border.
                cx0 = float(out_w) / 2.0
                cy0 = float(out_h) / 2.0
                dir_u = -float(gy)
                dir_v = -float(gx)
                if abs(dir_u) < 1e-9 and abs(dir_v) < 1e-9:
                    dir_v = 1.0
                scale = float(max(out_w, out_h)) * 2.0
                tx = cx0 + dir_u * scale
                ty = cy0 + dir_v * scale
                bx, by = _ray_to_border(cx0, cy0, tx, ty, width=out_w, height=out_h)
                anchor_x, anchor_y = float(bx), float(by)

            if anchor_x is None or anchor_y is None:
                raise ValueError("goal_anchor_unavailable")

            # Optional tiny ground anchor marker (kept deliberately small to avoid "nearest label
            # to dot" bias).
            if (
                str(getattr(cfg, "goal_projection_marker", "none")) == "dot"
                and int(cfg.goal_projection_dot_radius) > 0
            ):
                try:
                    rr = int(cfg.goal_projection_dot_radius)
                    a = int(round(float(cfg.goal_projection_dot_alpha) * 255.0))
                    goal_draw.ellipse(
                        (
                            _safe_int(float(anchor_x) - rr),
                            _safe_int(float(anchor_y) - rr),
                            _safe_int(float(anchor_x) + rr),
                            _safe_int(float(anchor_y) + rr),
                        ),
                        fill=(
                            cfg.goal_marker_rgb[0],
                            cfg.goal_marker_rgb[1],
                            cfg.goal_marker_rgb[2],
                            a,
                        ),
                        outline=(0, 0, 0, min(255, a + 60)),
                        width=1,
                    )
                except Exception:
                    pass

            # Draw the raised goal arrow + dashed hanging line behind labels.
            try:
                fs = float(max(10, int(cfg.label_font_size)))
                head_w = float(max(24.0, min(48.0, fs * 3.0)))
                head_h = float(max(18.0, min(38.0, fs * 2.4)))
                y_top = float(max(68.0, float(cfg.goal_text_font_size) + 44.0))
                y_top = float(min(max(y_top, 10.0), float(out_h - 1) - 60.0))
                tip_y = float(y_top + head_h)
                xh = float(min(max(float(anchor_x), 0.0), float(out_w - 1)))
                y0 = float(anchor_y)

                # Arrowhead (downward triangle) with a small shadow.
                base_y = float(y_top)
                base_l = float(xh - head_w * 0.5)
                base_r = float(xh + head_w * 0.5)
                base_l = float(min(max(base_l, 2.0), float(out_w - 3)))
                base_r = float(min(max(base_r, 2.0), float(out_w - 3)))
                tip_x = float(min(max(xh, 1.0), float(out_w - 2)))
                tri = [(base_l, base_y), (base_r, base_y), (tip_x, tip_y)]

                def _poly(shift: tuple[float, float], fill: tuple[int, int, int, int]) -> None:
                    dx2, dy2 = shift
                    pts2 = [(_safe_int(px + dx2), _safe_int(py + dy2)) for (px, py) in tri]
                    goal_draw.polygon(pts2, fill=fill)
                    goal_draw.line(pts2 + [pts2[0]], fill=(0, 0, 0, 255), width=2)

                # Shadow + main arrowhead.
                _poly((3.0, 3.0), (0, 0, 0, 150))
                _poly(
                    (0.0, 0.0),
                    (cfg.goal_marker_rgb[0], cfg.goal_marker_rgb[1], cfg.goal_marker_rgb[2], 235),
                )

                # GOAL tag placed near the arrowhead (kept behind candidate labels).
                try:
                    side = 1 if xh < float(out_w) * 0.5 else -1
                    tx0 = float(xh) + float(side) * float(head_w * 0.75 + 18.0)
                    ty0 = float(max(18.0, base_y - 10.0))
                    tx0 = float(min(max(tx0, 24.0), float(out_w - 24.0)))

                    text = "GOAL"
                    pad = 4
                    try:
                        bx0, by0, bx1, by1 = goal_draw.textbbox((0, 0), text, font=font)
                        tw = float(bx1 - bx0)
                        th = float(by1 - by0)
                    except Exception:
                        bx0, by0 = 0.0, 0.0
                        tw = float(goal_draw.textlength(text, font=font))
                        th = float(cfg.label_font_size)
                    x0 = float(tx0) - tw / 2.0 - float(pad)
                    y0b = float(ty0) - th / 2.0 - float(pad)
                    x1 = float(tx0) + tw / 2.0 + float(pad)
                    y1b = float(ty0) + th / 2.0 + float(pad)
                    goal_draw.rectangle(
                        (_safe_int(x0), _safe_int(y0b), _safe_int(x1), _safe_int(y1b)),
                        fill=(
                            cfg.goal_marker_rgb[0],
                            cfg.goal_marker_rgb[1],
                            cfg.goal_marker_rgb[2],
                            235,
                        ),
                        outline=(0, 0, 0, 255),
                        width=2,
                    )
                    goal_draw.text(
                        (
                            _safe_int(float(tx0) - tw / 2.0 - float(bx0)),
                            _safe_int(float(ty0) - th / 2.0 - float(by0)),
                        ),
                        text,
                        fill=(0, 0, 0, 255),
                        font=font,
                    )
                except Exception:
                    pass

                y_line0 = float(tip_y + 6.0)
                y_line1 = float(y0 - 6.0)
                if y_line1 < y_line0:
                    y_line0, y_line1 = y_line1, y_line0

                line_w = max(2, int(round(float(max(1, int(cfg.goal_marker_line_width))) * 0.8)))
                line_col = (
                    cfg.goal_marker_rgb[0],
                    cfg.goal_marker_rgb[1],
                    cfg.goal_marker_rgb[2],
                    170,
                )
                dash = 12.0
                gap = 7.0
                yy = float(y_line0)
                while yy < float(y_line1) - 1.0:
                    y2 = min(float(yy + dash), float(y_line1))
                    goal_draw.line(
                        (
                            _safe_int(xh + 3.0),
                            _safe_int(yy + 3.0),
                            _safe_int(xh + 3.0),
                            _safe_int(y2 + 3.0),
                        ),
                        fill=(0, 0, 0, 110),
                        width=line_w,
                    )
                    goal_draw.line(
                        (_safe_int(xh), _safe_int(yy), _safe_int(xh), _safe_int(y2)),
                        fill=line_col,
                        width=line_w,
                    )
                    yy += dash + gap
            except Exception:
                pass
        except Exception:
            pass

    # NOTE: the goal marker/direction arrow arrowhead + label are drawn after candidate labels
    # (later in this function).

    # Second pass: draw endpoint labels on top of the composited image.
    if cfg.label_indices and label_positions:
        draw_out_labels = ImageDraw.Draw(out, "RGBA")
        # Collision-avoid label placement (keeps indices readable when endpoints cluster).
        placed_rects: list[tuple[float, float, float, float]] = []  # (x0,y0,x1,y1) in px

        def _measure_label(text: str) -> tuple[float, float, float, float, float, float]:
            """Return (bx0,by0,tw,th,pad,outline_pad) in px for background box sizing.

            bx0/by0 are the glyph bbox offsets (can be negative).
            tw/th are glyph bbox size (without padding).
            """
            pad = 3.0
            try:
                bx0, by0, bx1, by1 = draw_out_labels.textbbox((0, 0), text, font=font)
                tw = float(bx1 - bx0)
                th = float(by1 - by0)
                return float(bx0), float(by0), float(tw), float(th), float(pad), 0.0
            except Exception:
                # Fallback: no negative bbox offsets available.
                tw = float(draw_out_labels.textlength(text, font=font))
                th = float(cfg.label_font_size)
                return 0.0, 0.0, float(tw), float(th), float(pad), 0.0

        def _rect_for_center(
            cx: float, cy: float, *, tw: float, th: float, pad: float
        ) -> tuple[float, float, float, float]:
            x0 = float(cx) - float(tw) / 2.0 - float(pad)
            y0 = float(cy) - float(th) / 2.0 - float(pad)
            x1 = float(cx) + float(tw) / 2.0 + float(pad)
            y1 = float(cy) + float(th) / 2.0 + float(pad)
            return x0, y0, x1, y1

        def _clamp_center_to_fit(
            cx: float, cy: float, *, tw: float, th: float, pad: float
        ) -> tuple[float, float]:
            x0, y0, x1, y1 = _rect_for_center(cx, cy, tw=tw, th=th, pad=pad)
            dx = dy = 0.0
            if x0 < 0.0:
                dx = -x0
            if x1 > float(out_w - 1):
                dx = float(out_w - 1) - x1 if abs(dx) < 1e-9 else dx
            if y0 < 0.0:
                dy = -y0
            if y1 > float(out_h - 1):
                dy = float(out_h - 1) - y1 if abs(dy) < 1e-9 else dy
            return float(cx + dx), float(cy + dy)

        def _intersects(
            a: tuple[float, float, float, float],
            b: tuple[float, float, float, float],
            *,
            margin: float,
        ) -> bool:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            m = float(max(0.0, margin))
            return not (ax1 + m < bx0 or bx1 + m < ax0 or ay1 + m < by0 or by1 + m < ay0)

        def _intersection_area(
            a: tuple[float, float, float, float], b: tuple[float, float, float, float]
        ) -> float:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            ix0 = max(ax0, bx0)
            iy0 = max(ay0, by0)
            ix1 = min(ax1, bx1)
            iy1 = min(ay1, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                return 0.0
            return float((ix1 - ix0) * (iy1 - iy0))

        def _draw_candidate_label_at(
            draw_ctx: ImageDraw.ImageDraw,
            u: int,
            v: int,
            text: str,
            *,
            bg_rgb: tuple[int, int, int],
        ) -> None:
            """Candidate label with background matching its trajectory color."""
            pad = 3
            try:
                bx0, by0, bx1, by1 = draw_ctx.textbbox((0, 0), text, font=font)
                tw = float(bx1 - bx0)
                th = float(by1 - by0)
            except Exception:
                bx0, by0 = 0.0, 0.0
                tw = float(draw_ctx.textlength(text, font=font))
                th = float(cfg.label_font_size)

            # Choose text color for contrast.
            lum = 0.2126 * float(bg_rgb[0]) + 0.7152 * float(bg_rgb[1]) + 0.0722 * float(bg_rgb[2])
            txt_rgb = (0, 0, 0) if lum > 150.0 else (255, 255, 255)

            x0 = float(u) - tw / 2.0 - float(pad)
            y0 = float(v) - th / 2.0 - float(pad)
            x1 = float(u) + tw / 2.0 + float(pad)
            y1 = float(v) + th / 2.0 + float(pad)

            draw_ctx.rectangle(
                (_safe_int(x0), _safe_int(y0), _safe_int(x1), _safe_int(y1)),
                fill=(int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), label_bg_alpha),
                outline=(0, 0, 0, min(255, label_bg_alpha + 70)),
                width=2,
            )

            text_x = float(u) - tw / 2.0 - float(bx0)
            text_y = float(v) - th / 2.0 - float(by0)
            draw_ctx.text(
                (_safe_int(text_x), _safe_int(text_y)),
                text,
                fill=(txt_rgb[0], txt_rgb[1], txt_rgb[2], 255),
                font=font,
            )

        # Try a small deterministic offset set (spiral-ish) around the endpoint.
        def _candidate_centers(
            cx0: float, cy0: float, *, step: float, max_r: float
        ) -> list[tuple[float, float]]:
            centers: list[tuple[float, float]] = [(cx0, cy0)]
            r = float(step)
            while r <= float(max_r) + 1e-6:
                # Cardinals + diagonals first (most readable).
                for dx, dy in [
                    (0.0, -r),
                    (0.0, r),
                    (-r, 0.0),
                    (r, 0.0),
                    (-r, -r),
                    (r, -r),
                    (-r, r),
                    (r, r),
                ]:
                    centers.append((float(cx0 + dx), float(cy0 + dy)))
                # Extra offsets to break symmetry (helps when many labels land on same endpoint).
                for dx, dy in [
                    (-2.0 * r, 0.0),
                    (2.0 * r, 0.0),
                    (0.0, -2.0 * r),
                    (0.0, 2.0 * r),
                ]:
                    centers.append((float(cx0 + dx), float(cy0 + dy)))
                r += float(step)
            return centers

        # Keep a modest separation so labels don't visually merge.
        min_sep = 2.0
        step = float(max(6.0, float(cfg.label_font_size) * 0.6))
        max_r = float(max(24.0, float(cfg.label_font_size) * 2.5))

        for u, v, text in label_positions:
            cx0 = float(u)
            cy0 = float(v)
            bx0, by0, tw, th, pad, _ = _measure_label(str(text))
            _ = bx0, by0  # bbox offsets handled inside _draw_index_label_at

            best_center = (cx0, cy0)
            best_penalty = float("inf")

            for cx_try, cy_try in _candidate_centers(cx0, cy0, step=step, max_r=max_r):
                cx_fit, cy_fit = _clamp_center_to_fit(cx_try, cy_try, tw=tw, th=th, pad=pad)
                rect = _rect_for_center(cx_fit, cy_fit, tw=tw, th=th, pad=pad)

                if not any(_intersects(rect, r, margin=min_sep) for r in placed_rects):
                    best_center = (cx_fit, cy_fit)
                    best_penalty = 0.0
                    break

                # Otherwise, compute a soft penalty and keep the best.
                pen = 0.0
                for r in placed_rects:
                    pen += _intersection_area(rect, r)
                # Prefer staying close to the endpoint when ties happen.
                pen += 0.01 * float((cx_fit - cx0) ** 2 + (cy_fit - cy0) ** 2)
                if pen < best_penalty:
                    best_penalty = float(pen)
                    best_center = (cx_fit, cy_fit)

            # Record the chosen rectangle so later labels avoid it.
            cx_best, cy_best = best_center
            rect_best = _rect_for_center(cx_best, cy_best, tw=tw, th=th, pad=pad)
            placed_rects.append(rect_best)

            # Leader line from endpoint anchor -> label center (when label is nudged away).
            idx_int: int | None = None
            try:
                idx_int = int(str(text))
            except Exception:
                idx_int = None
            anchor = label_anchor_by_idx.get(int(idx_int)) if idx_int is not None else None
            col = (
                idx_to_color.get(int(idx_int), cfg.label_bg_rgb)
                if idx_int is not None
                else cfg.label_bg_rgb
            )
            if anchor is not None:
                try:
                    ax, ay = float(anchor[0]), float(anchor[1])
                    dx = float(cx_best) - ax
                    dy = float(cy_best) - ay
                    dist = math.hypot(dx, dy)
                    if dist > 8.0:
                        draw_out_labels.line(
                            (
                                _safe_int(ax + 2.0),
                                _safe_int(ay + 2.0),
                                _safe_int(cx_best + 2.0),
                                _safe_int(cy_best + 2.0),
                            ),
                            fill=(0, 0, 0, 110),
                            width=3,
                        )
                        draw_out_labels.line(
                            (_safe_int(ax), _safe_int(ay), _safe_int(cx_best), _safe_int(cy_best)),
                            fill=(col[0], col[1], col[2], min(255, label_bg_alpha + 40)),
                            width=2,
                        )
                except Exception:
                    pass

            _draw_candidate_label_at(
                draw_out_labels, _safe_int(cx_best), _safe_int(cy_best), str(text), bg_rgb=col
            )

    # Fallback: if some candidates have no visible points, still render their indices in a bottom
    # strip.
    # NOTE: these labels must be drawn onto `out` (post-composite), otherwise they won't appear.
    if cfg.label_indices and missing_label_indices and int(cfg.pad_bottom) > 0:
        draw_out = ImageDraw.Draw(out, "RGBA")

        def draw_label_center(u: int, v: int, text: str) -> int:
            pad = 3
            try:
                bx0, by0, bx1, by1 = draw_out.textbbox((0, 0), text, font=font)
                tw = float(bx1 - bx0)
                th = float(by1 - by0)
            except Exception:
                bx0, by0 = 0.0, 0.0
                tw = float(draw_out.textlength(text, font=font))
                th = float(cfg.label_font_size)

            x0 = float(u) - tw / 2.0 - float(pad)
            y0 = float(v) - th / 2.0 - float(pad)
            x1 = float(u) + tw / 2.0 + float(pad)
            y1 = float(v) + th / 2.0 + float(pad)
            draw_out.rectangle(
                (_safe_int(x0), _safe_int(y0), _safe_int(x1), _safe_int(y1)),
                fill=(
                    cfg.label_bg_rgb[0],
                    cfg.label_bg_rgb[1],
                    cfg.label_bg_rgb[2],
                    label_bg_alpha,
                ),
                outline=(0, 0, 0, min(255, label_bg_alpha + 60)),
                width=1,
            )
            text_x = float(u) - tw / 2.0 - float(bx0)
            text_y = float(v) - th / 2.0 - float(by0)
            draw_out.text(
                (_safe_int(text_x), _safe_int(text_y)),
                text,
                fill=(cfg.label_text_rgb[0], cfg.label_text_rgb[1], cfg.label_text_rgb[2], 255),
                font=font,
            )
            return int(round(tw + 2 * float(pad))) + 18

        strip_y = out_h - max(24, int(cfg.label_font_size) + 10)

        # Sort by anchor u when available; keep None anchors last.
        missing_sorted = sorted(
            missing_label_indices, key=lambda t: (t[1] is None, t[1] if t[1] is not None else 0.0)
        )

        # Place labels near their u anchors, with light collision avoidance.
        placed: list[tuple[int, int]] = []  # (x_center, approx_width)
        for idx, u_anchor in missing_sorted:
            text = str(idx)
            # Estimate width for collision avoidance.
            try:
                bbox = draw_out.textbbox((0, 0), text, font=font)
                tw = int(bbox[2] - bbox[0]) + 18
            except Exception:
                tw = max(32, int(cfg.label_font_size) + 18)

            if u_anchor is None or not math.isfinite(u_anchor):
                x_center = 16 + sum(w for _, w in placed)  # simple left-to-right fallback
            else:
                x_center = int(round(u_anchor))

            # Clamp inside bounds.
            x_center = max(16, min(int(x_center), out_w - 16))

            # Simple collision avoidance: if too close to an existing label, nudge right.
            min_gap = 8
            for px, pw in placed:
                if abs(x_center - px) < (tw // 2 + pw // 2 + min_gap):
                    x_center = px + (pw // 2 + tw // 2 + min_gap)
            x_center = max(16, min(int(x_center), out_w - 16))

            draw_label_center(int(x_center), int(strip_y), text)
            placed.append((int(x_center), int(tw)))

    # --- Goal marker (draw last so it can't be occluded by labels) ---
    # We support two visualizations:
    # - ground marker (legacy): a crosshair + label at the projected goal point
    # - raised/hanging marker (VLM-friendly): a floating arrow+label connected by a vertical dashed
    # line
    #   to the projected goal point. This helps communicate that the goal is a *target direction*,
    #   not an immediate 4-second endpoint to reach.
    if goal_xy is not None:
        try:
            goal_draw2 = ImageDraw.Draw(out, "RGBA")

            gx = float(goal_xy[0])
            gy = float(goal_xy[1])

            r = int(cfg.goal_marker_radius)
            w = int(cfg.goal_marker_line_width)
            col = (cfg.goal_marker_rgb[0], cfg.goal_marker_rgb[1], cfg.goal_marker_rgb[2], 255)

            def _draw_goal_label_at(
                draw_ctx: ImageDraw.ImageDraw, x: int, y: int, text: str
            ) -> None:
                """Goal label with a distinct style (magenta background) to avoid confusion
                with candidate indices."""
                pad = 4
                try:
                    bx0, by0, bx1, by1 = draw_ctx.textbbox((0, 0), text, font=font)
                    tw = float(bx1 - bx0)
                    th = float(by1 - by0)
                except Exception:
                    bx0, by0 = 0.0, 0.0
                    tw = float(draw_ctx.textlength(text, font=font))
                    th = float(cfg.label_font_size)

                x0 = float(x) - tw / 2.0 - float(pad)
                y0 = float(y) - th / 2.0 - float(pad)
                x1 = float(x) + tw / 2.0 + float(pad)
                y1 = float(y) + th / 2.0 + float(pad)

                goal_draw2.rectangle(
                    (_safe_int(x0), _safe_int(y0), _safe_int(x1), _safe_int(y1)),
                    fill=(
                        cfg.goal_marker_rgb[0],
                        cfg.goal_marker_rgb[1],
                        cfg.goal_marker_rgb[2],
                        255,
                    ),
                    outline=(0, 0, 0, 255),
                    width=2,
                )
                text_x = float(x) - tw / 2.0 - float(bx0)
                text_y = float(y) - th / 2.0 - float(by0)
                goal_draw2.text(
                    (_safe_int(text_x), _safe_int(text_y)),
                    text,
                    fill=(0, 0, 0, 255),
                    font=font,
                )

            def _draw_goal_crosshair_at(x: int, y: int) -> None:
                """Legacy goal marker on the ground plane (crosshair + label)."""
                if r > 0:
                    goal_draw2.ellipse((x - r, y - r, x + r, y + r), outline=col, width=w)
                    goal_draw2.line((x - r, y, x + r, y), fill=col, width=w)
                    goal_draw2.line((x, y - r, x, y + r), fill=col, width=w)
                _draw_goal_label_at(goal_draw2, x, y - (max(0, r) + 14), "GOAL")

            def _ray_to_border(
                x0: float, y0: float, x1: float, y1: float, *, width: int, height: int
            ) -> tuple[int, int]:
                """Intersect ray (x0,y0) -> (x1,y1) with the image rectangle border."""
                dx = float(x1 - x0)
                dy = float(y1 - y0)
                if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                    return int(min(max(round(x0), 0), width - 1)), int(
                        min(max(round(y0), 0), height - 1)
                    )

                candidates: list[tuple[float, float, float]] = []  # (t, x, y)
                x_min = 0.0
                x_max = float(width - 1)
                y_min = 0.0
                y_max = float(height - 1)

                # Intersections with vertical borders.
                if abs(dx) >= 1e-9:
                    for xb in (x_min, x_max):
                        t = (xb - float(x0)) / dx
                        if t > 0:
                            y = float(y0) + t * dy
                            if y_min - 1e-6 <= y <= y_max + 1e-6:
                                candidates.append((t, xb, y))
                # Intersections with horizontal borders.
                if abs(dy) >= 1e-9:
                    for yb in (y_min, y_max):
                        t = (yb - float(y0)) / dy
                        if t > 0:
                            x = float(x0) + t * dx
                            if x_min - 1e-6 <= x <= x_max + 1e-6:
                                candidates.append((t, x, yb))

                if not candidates:
                    # Fallback: clamp the destination point.
                    bx = int(min(max(round(x1), 0), width - 1))
                    by = int(min(max(round(y1), 0), height - 1))
                    return bx, by

                _t, xb_best, yb_best = min(candidates, key=lambda a: a[0])
                bx = int(min(max(round(xb_best), 0), width - 1))
                by = int(min(max(round(yb_best), 0), height - 1))
                return bx, by

            # Goal marker position (project goal_xy to pixels). If off-screen, place on border.
            anchor_x: float | None = None
            anchor_y: float | None = None
            gu, gv = _xy_to_px(gx, gy, width=base_w, height=base_h, cfg=cfg)
            if math.isfinite(gu) and math.isfinite(gv):
                gu2 = float(gu) + float(cfg.pad_left)
                gv2 = float(gv) + float(cfg.pad_top)
                if 0 <= gu2 < out_w and 0 <= gv2 < out_h:
                    anchor_x, anchor_y = float(gu2), float(gv2)
                else:
                    cx0 = float(out_w) / 2.0
                    cy0 = float(out_h) / 2.0
                    bx, by = _ray_to_border(
                        cx0, cy0, float(gu2), float(gv2), width=out_w, height=out_h
                    )
                    anchor_x, anchor_y = float(bx), float(by)
            else:
                # Projection failed (e.g., fisheye model when goal is behind camera); show
                # direction on border.
                cx0 = float(out_w) / 2.0
                cy0 = float(out_h) / 2.0
                dir_u = -float(gy)
                dir_v = -float(gx)
                if abs(dir_u) < 1e-9 and abs(dir_v) < 1e-9:
                    dir_v = 1.0
                scale = float(max(out_w, out_h)) * 2.0
                tx = cx0 + dir_u * scale
                ty = cy0 + dir_v * scale
                bx, by = _ray_to_border(cx0, cy0, tx, ty, width=out_w, height=out_h)
                anchor_x, anchor_y = float(bx), float(by)

            if anchor_x is None or anchor_y is None:
                raise ValueError("goal_anchor_unavailable")

            # Raised / hanging marker (preferred for VLM): drawn BEFORE candidate labels so it
            # doesn't occlude indices.
            if bool(getattr(cfg, "draw_goal_direction_arrow", False)):
                pass

            # Legacy ground crosshair marker.
            elif bool(cfg.draw_goal_marker) and r > 0:
                _draw_goal_crosshair_at(_safe_int(anchor_x), _safe_int(anchor_y))
        except Exception:
            pass

    return out


def write_overlay_image(
    *,
    frame_path: Path,
    out_path: Path,
    record: PlannerCandidatesRecord,
    cfg: OverlayConfig,
    pred_index: int | None = None,
    label_index: int | None = None,
    goal_text: str | None = None,
    goal_xy: tuple[float, float] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(frame_path) as im:
        overlay = render_overlay_pil(
            base_image=im,
            record=record,
            cfg=cfg,
            pred_index=pred_index,
            label_index=label_index,
            goal_text=goal_text,
            goal_xy=goal_xy,
        )
        overlay.convert("RGB").save(out_path, format="PNG")


def write_overlay_for_rgb_record(
    *,
    rgb_record: RGBRecord,
    episode_dir: Path,
    dataset_root: Path,
    out_path: Path,
    record: PlannerCandidatesRecord,
    cfg: OverlayConfig,
    pred_index: int | None = None,
    label_index: int | None = None,
    goal_text: str | None = None,
    goal_xy: tuple[float, float] | None = None,
    rgb_loader: RGBFrameLoader | None = None,
    base_image_width: int | None = None,
) -> None:
    """Write an overlay image using an `RGBRecord`.

    This supports:
    - image-backed `frame_ref` (png/jpg/etc)
    - video-backed `frame_ref` (mp4) with `rgb_record.frame_index` extension field
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    loader = rgb_loader or RGBFrameLoader()
    base = loader.load(rgb_record, episode_dir=episode_dir, dataset_root=dataset_root)
    # Keep overlays visually consistent across datasets:
    # - COCO frames can be very high-res, making labels/lines look tiny.
    # - Real logs are often smaller.
    # If the caller does not specify a width, we auto-cap very large frames to a reasonable size
    # (never upsample).
    if base_image_width is None and base.size[0] > 960:
        base_image_width = 960
    if base_image_width is not None:
        w = int(base_image_width)
        if w > 0 and base.size[0] != w:
            h = max(1, int(round(float(base.size[1]) * (float(w) / float(base.size[0])))))
            base = base.resize((w, h), resample=Image.BILINEAR)
    overlay = render_overlay_pil(
        base_image=base,
        record=record,
        cfg=cfg,
        pred_index=pred_index,
        label_index=label_index,
        goal_text=goal_text,
        goal_xy=goal_xy,
    )
    overlay.convert("RGB").save(out_path, format="PNG")


def write_overlay_from_base_image(
    *,
    base_image: Image.Image,
    out_path: Path,
    record: PlannerCandidatesRecord,
    cfg: OverlayConfig,
    pred_index: int | None = None,
    label_index: int | None = None,
    goal_text: str | None = None,
    goal_xy: tuple[float, float] | None = None,
) -> None:
    """Same as `write_overlay_image`, but uses an in-memory base image (e.g. a placeholder)."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = render_overlay_pil(
        base_image=base_image,
        record=record,
        cfg=cfg,
        pred_index=pred_index,
        label_index=label_index,
        goal_text=goal_text,
        goal_xy=goal_xy,
    )
    overlay.convert("RGB").save(out_path, format="PNG")


def write_overlay_with_placeholder(
    *,
    width: int,
    height: int,
    out_path: Path,
    record: PlannerCandidatesRecord,
    cfg: OverlayConfig,
    pred_index: int | None = None,
    label_index: int | None = None,
    goal_text: str | None = None,
    goal_xy: tuple[float, float] | None = None,
    background_rgb: tuple[int, int, int] = (240, 240, 240),
) -> None:
    """Write an overlay using a blank placeholder base image."""

    base = Image.new("RGB", (int(width), int(height)), color=background_rgb)
    write_overlay_from_base_image(
        base_image=base,
        out_path=out_path,
        record=record,
        cfg=cfg,
        pred_index=pred_index,
        label_index=label_index,
        goal_text=goal_text,
        goal_xy=goal_xy,
    )


def project_xy_to_uv(
    x: float, y: float, *, width: int, height: int, cfg: OverlayConfig
) -> tuple[float, float]:
    """Project a robot-frame ground point (x forward, y left) to image (u, v) pixels."""

    return _xy_to_px(x, y, width=width, height=height, cfg=cfg)


def candidate_endpoint_uv(
    record: PlannerCandidatesRecord, idx: int, *, width: int, height: int, cfg: OverlayConfig
) -> tuple[float, float] | None:
    """Return the projected endpoint pixel for candidate idx, or None if invalid/out-of-model."""

    if idx < 0 or idx >= len(record.candidates):
        return None
    pts = record.candidates[idx].points_xy
    if not pts:
        return None
    x, y = pts[-1]
    u, v = project_xy_to_uv(float(x), float(y), width=width, height=height, cfg=cfg)
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return u, v
