from __future__ import annotations

import base64
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slow_brain_fast_planner.utils.io import read_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Invalid JSONL record (expected object) at {path}:{line_no}")
            out.append(obj)
    return out


def escape_pre(text: str) -> str:
    return html.escape(text, quote=False)


def escape(text: str) -> str:
    return html.escape(text, quote=False)


def _fmt_metric_value(v: Any, *, digits: int = 6) -> str:
    """Human-friendly formatting for metrics values in HTML.

    - None -> "-"
    - floats -> rounded, trimmed (matches metrics_report.txt style)
    - dict/list -> compact JSON
    """
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        s = f"{v:.{int(digits)}f}"
        # Trim trailing zeros.
        s = s.rstrip("0").rstrip(".")
        return s
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, sort_keys=True)
        except Exception:
            return str(v)
    return str(v)


def _flatten_metrics(obj: Any, *, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested dict/list metrics into (key_path, value) pairs."""
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            kk = str(k)
            p2 = kk if not prefix else f"{prefix}.{kk}"
            out.extend(_flatten_metrics(obj.get(k), prefix=p2))
        return out
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            p2 = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.extend(_flatten_metrics(v, prefix=p2))
        return out
    out.append((prefix or "value", obj))
    return out


def render_messages_human(
    messages: Any, *, run_dir: Path | None = None, embed_images: bool = False
) -> str:
    """Render messages in a human-readable way (avoids JSON-escaped \\n).

    Supports our internal format:
    - {"role": "system"|"user", "content": "string"} OR
    - {"role": "user", "content": [{"type":"text","text":...}, {"type":"image_ref","image_ref":...}]}

    Each role is rendered as a collapsible <details> block.
    System prompts are collapsed by default; user prompts are expanded.
    """

    if not isinstance(messages, list):
        return "<div class='muted'>No messages.</div>"

    def _resolve_ref_path(ref: str) -> Path | None:
        """Resolve an image_ref to an on-disk path (handles merged shard runs)."""
        if run_dir is None:
            return None
        try:
            p = Path(str(ref))
            if p.is_absolute():
                return p if p.exists() else None
            # Try directly under run_dir first.
            # NOTE: avoid .resolve() so we don't follow shard symlinks (keeps paths within run_dir).
            cand = Path(run_dir) / p
            if cand.exists():
                return cand
            # If this is a merged-shards run, artifacts may live under RUN_DIR/shards/<shardX>/...
            shards_dir = (Path(run_dir) / "shards").resolve()
            if shards_dir.is_dir():
                for child in sorted([x for x in shards_dir.iterdir() if x.is_dir()]):
                    cand2 = child / p
                    if cand2.exists():
                        return cand2
            return None
        except Exception:
            return None

    out: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", ""))
        content = m.get("content")

        # System messages are collapsed by default; others are open
        is_system = role.lower() == "system"
        open_attr = "" if is_system else " open"

        out.append(f"<details class='msg-collapsible'{open_attr}>")
        out.append(f"<summary><strong>{escape(role)}</strong> (click to expand/collapse)</summary>")
        out.append("<div class='msg-content'>")

        # Multi-part message (OpenAI-style).
        if isinstance(content, list):
            text_chunks: list[str] = []
            image_refs: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text_chunks.append(str(part.get("text", "")))
                elif ptype == "image_ref":
                    image_refs.append(str(part.get("image_ref", "")))
                else:
                    # Unknown part; best-effort stringify.
                    text_chunks.append(str(part))

            if text_chunks:
                out.append("<pre>" + escape_pre("\n\n".join(text_chunks)) + "</pre>")
            for ref in image_refs:
                out.append(f"<div class='muted'>image_ref: <code>{escape(ref)}</code></div>")
                rp = _resolve_ref_path(ref)
                if rp is not None and rp.exists():
                    try:
                        img_src = _image_src(rp, embed=bool(embed_images), run_dir=Path(run_dir))
                        out.append("<div class='imgwrap' style='margin-top:6px'>")
                        out.append(f'<img src="{img_src}" alt="image_ref">')
                        out.append("</div>")
                    except Exception:
                        pass
        else:
            out.append("<pre>" + escape_pre("" if content is None else str(content)) + "</pre>")

        out.append("</div>")
        out.append("</details>")

    return "\n".join(out)


def _extract_rationale_from_raw(raw: Any) -> str | None:
    """Best-effort parse for rationale from a raw model output string (JSON)."""

    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            rat = obj.get("rationale")
            if isinstance(rat, str) and rat.strip():
                return rat.strip()
    except Exception:
        return None
    return None


def pretty_json_from_raw(raw: Any) -> str | None:
    """Best-effort pretty JSON rendering from a raw model output string."""

    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    # Try direct JSON.
    try:
        obj = json.loads(s)
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, indent=2, sort_keys=True)
    except Exception:
        pass
    # Try fenced JSON block.
    try:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
        if m:
            inner = m.group(1).strip()
            obj2 = json.loads(inner)
            if isinstance(obj2, (dict, list)):
                return json.dumps(obj2, indent=2, sort_keys=True)
    except Exception:
        pass
    # Try a JSON object substring.
    try:
        m2 = re.search(r"\{[\s\S]*\}", s)
        if m2:
            obj3 = json.loads(m2.group(0))
            if isinstance(obj3, (dict, list)):
                return json.dumps(obj3, indent=2, sort_keys=True)
    except Exception:
        pass
    return None


def _image_to_data_uri(path: Path) -> str:
    b = path.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _image_src(path: Path, *, embed: bool, run_dir: Path) -> str:
    """Return an image src attribute value.

    If embed=True, returns a base64 data URI (self-contained but large).
    If embed=False, returns a relative path from run_dir (small, requires images on disk).
    """
    if embed:
        return _image_to_data_uri(path)
    # Compute relative path from run_dir to the image.
    #
    # IMPORTANT: do NOT call path.resolve() before relative_to(). For merged runs we often
    # symlink RUN_DIR/shards/<shardX> -> <original_shard_dir>. Resolving the image path
    # would follow the symlink and make the path appear outside RUN_DIR, which breaks
    # relative linking and causes webviews / HTTP servers to fail loading images.
    try:
        return str(path.relative_to(run_dir))
    except Exception:
        pass
    try:
        rel = path.resolve().relative_to(run_dir.resolve())
        return str(rel)
    except Exception:
        # Image is outside run_dir; fall back to absolute path.
        return str(path.resolve())


def _snapshot_key(rec: dict[str, Any]) -> tuple[str, float, int]:
    return (str(rec.get("episode_id")), float(rec.get("t")), int(rec.get("snapshot_index")))


@dataclass(frozen=True)
class RunSampleView:
    original_index: int
    episode_id: str
    t: float
    snapshot_index: int
    prediction: Any
    label: Any
    correct: Any
    skipped: bool
    skip_reason: Any
    goal_distance_m: Any
    selected_end_dist_to_goal_m: Any
    selected_goal_ang_diff_deg: Any
    selected_traj_avg_dist_to_goal_m: Any
    selected_goal_progress_m: Any
    # Open-loop social metrics (optional; present in newer Task2 outputs).
    traj_len_m: Any
    maoe_deg: Any
    dcr: Any
    tcr: Any
    compliance_source: Any
    overlay_frame_ref: str | None
    overlay_error: str | None
    local_plot_frame_ref: str | None
    local_plot_error: str | None
    auto_enabled: Any
    ade: Any
    rationale: Any
    prompt_messages: Any
    raw_model_response: Any
    parsed_advice: Any
    parse_note: Any
    provider_meta: Any
    thoughts: Any


def generate_run_report_html(
    *,
    run_dir: Path,
    out_path: Path | None = None,
    predictions_path: Path | None = None,
    embed_images: bool = False,
) -> Path:
    """Generate a single self-contained HTML report for browsing a run directory.

    Args:
        run_dir: Path to the run directory containing predictions and artifacts.
        out_path: Output path for the HTML report. Defaults to run_dir/report.html.
        predictions_path: Path to predictions.jsonl. Defaults to run_dir/predictions.jsonl.
        embed_images: If True, embed images as base64 data URIs (creates large self-contained files).
                      If False (default), use relative file paths (much smaller, requires images on disk).

    Expected inputs (best-effort):
    - RUN_DIR/predictions.jsonl (or predictions_path)
    - RUN_DIR/traces/*.jsonl (events with types: obs, model_call, action, metric)
    - RUN_DIR/artifacts/overlays/... (referenced by overlay_frame_ref in predictions.jsonl)
    """

    run_dir = Path(run_dir).resolve()
    if out_path is None:
        out_path = run_dir / "report.html"
    out_path = Path(out_path).resolve()

    preds_path = (
        predictions_path if predictions_path is not None else (run_dir / "predictions.jsonl")
    )
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {preds_path}")
    preds = _read_jsonl(preds_path)

    # Optional: config/metrics.
    cfg: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg_obj = read_json(cfg_path)
        if isinstance(cfg_obj, dict):
            cfg = cfg_obj
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics_obj = read_json(metrics_path)
        if isinstance(metrics_obj, dict):
            metrics = metrics_obj

    # Optional UI behavior (per-run config).
    default_hide_auto_true = False
    try:
        if isinstance(cfg, dict):
            rep = cfg.get("report") if isinstance(cfg.get("report"), dict) else None
            if isinstance(rep, dict):
                default_hide_auto_true = bool(rep.get("default_hide_auto_true", False))
    except Exception:
        default_hide_auto_true = False

    # Parse traces to pull out prompt + raw model output per snapshot.
    prompt_by_key: dict[tuple[str, float, int], Any] = {}
    raw_by_key: dict[tuple[str, float, int], Any] = {}
    thoughts_by_key: dict[tuple[str, float, int], Any] = {}
    parsed_by_key: dict[tuple[str, float, int], Any] = {}
    parse_note_by_key: dict[tuple[str, float, int], str] = {}
    provider_meta_by_key: dict[tuple[str, float, int], Any] = {}
    traces_dir = run_dir / "traces"
    if traces_dir.is_dir():
        for trace_path in sorted(traces_dir.glob("*.jsonl")):
            for ev in _read_jsonl(trace_path):
                if ev.get("event_type") != "model_call":
                    continue
                try:
                    k = _snapshot_key(ev)
                except Exception:
                    continue
                if "messages" in ev:
                    prompt_by_key[k] = ev.get("messages")
                if "raw_response" in ev:
                    raw_by_key[k] = ev.get("raw_response")
                if "parsed_advice" in ev:
                    parsed_by_key[k] = ev.get("parsed_advice")
                    try:
                        pa = ev.get("parsed_advice")
                        if isinstance(pa, dict):
                            dbg = pa.get("debug")
                            if isinstance(dbg, dict):
                                pn = dbg.get("parse_note")
                                if isinstance(pn, str) and pn.strip():
                                    parse_note_by_key[k] = pn.strip()
                    except Exception:
                        pass
                if "provider_meta" in ev:
                    provider_meta_by_key[k] = ev.get("provider_meta")
                if "thoughts" in ev:
                    thoughts_by_key[k] = ev.get("thoughts")

    # If traces/ doesn't exist or yields nothing (e.g. older runs), try a best-effort scan
    # of traces/events.jsonl which might be in the same directory (common for simple runs).
    if not prompt_by_key:
        fallback_trace = run_dir / "traces" / "events.jsonl"
        if fallback_trace.exists():
            for ev in _read_jsonl(fallback_trace):
                if ev.get("event_type") != "model_call":
                    continue
                try:
                    k = _snapshot_key(ev)
                except Exception:
                    continue
                if "messages" in ev:
                    prompt_by_key[k] = ev.get("messages")
                if "raw_response" in ev:
                    raw_by_key[k] = ev.get("raw_response")
                if "parsed_advice" in ev:
                    parsed_by_key[k] = ev.get("parsed_advice")
                    try:
                        pa = ev.get("parsed_advice")
                        if isinstance(pa, dict):
                            dbg = pa.get("debug")
                            if isinstance(dbg, dict):
                                pn = dbg.get("parse_note")
                                if isinstance(pn, str) and pn.strip():
                                    parse_note_by_key[k] = pn.strip()
                    except Exception:
                        pass
                if "provider_meta" in ev:
                    provider_meta_by_key[k] = ev.get("provider_meta")
                if "thoughts" in ev:
                    thoughts_by_key[k] = ev.get("thoughts")

    samples: list[RunSampleView] = []
    for i, rec in enumerate(preds):
        k = _snapshot_key(rec)
        parse_note = rec.get("parse_note")
        if not (isinstance(parse_note, str) and parse_note.strip()):
            parse_note = parse_note_by_key.get(k)
        samples.append(
            RunSampleView(
                original_index=i,
                episode_id=str(rec.get("episode_id")),
                t=float(rec.get("t")),
                snapshot_index=int(rec.get("snapshot_index")),
                prediction=rec.get("prediction"),
                label=rec.get("label"),
                correct=rec.get("correct"),
                skipped=bool(rec.get("skipped")),
                skip_reason=rec.get("skip_reason"),
                goal_distance_m=rec.get("goal_distance_m"),
                selected_end_dist_to_goal_m=rec.get("selected_end_dist_to_goal_m"),
                selected_goal_ang_diff_deg=rec.get("selected_goal_ang_diff_deg"),
                selected_traj_avg_dist_to_goal_m=rec.get("selected_traj_avg_dist_to_goal_m"),
                selected_goal_progress_m=rec.get("selected_goal_progress_m"),
                traj_len_m=rec.get("traj_len_m"),
                maoe_deg=rec.get("maoe_deg"),
                dcr=rec.get("dcr"),
                tcr=rec.get("tcr"),
                compliance_source=rec.get("compliance_source"),
                overlay_frame_ref=rec.get("overlay_frame_ref"),
                overlay_error=rec.get("overlay_error"),
                local_plot_frame_ref=rec.get("local_plot_frame_ref"),
                local_plot_error=rec.get("local_plot_error"),
                auto_enabled=rec.get("auto_enabled"),
                ade=rec.get("ade"),
                rationale=rec.get("rationale"),
                prompt_messages=prompt_by_key.get(k),
                raw_model_response=raw_by_key.get(k),
                parsed_advice=parsed_by_key.get(k),
                parse_note=parse_note,
                provider_meta=provider_meta_by_key.get(k),
                thoughts=thoughts_by_key.get(k),
            )
        )

    # Default to sorting by highest ADE (Task 2) descending.
    def _get_ade_for_sort(s: RunSampleView) -> float:
        if isinstance(s.ade, dict):
            val = s.ade.get("selected")
            if isinstance(val, (int, float)):
                return float(val)
        return -1.0

    samples.sort(key=_get_ade_for_sort, reverse=True)

    # Render HTML.
    def _resolve_ref_in_run(ref: str) -> Path | None:
        """Resolve a relative artifact ref under run_dir (handles merged shards)."""
        try:
            p = Path(str(ref))
            if p.is_absolute():
                return p if p.exists() else None
            # NOTE: avoid .resolve() so we don't follow shard symlinks (keeps paths within run_dir).
            cand = run_dir / p
            if cand.exists():
                return cand
            shards_dir = (run_dir / "shards").resolve()
            if shards_dir.is_dir():
                for child in sorted([x for x in shards_dir.iterdir() if x.is_dir()]):
                    cand2 = child / p
                    if cand2.exists():
                        return cand2
            return None
        except Exception:
            return None

    def _collect_image_refs_from_messages(messages: Any) -> list[str]:
        refs: list[str] = []
        if not isinstance(messages, list):
            return refs
        for m in messages:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_ref":
                        r = part.get("image_ref")
                        if isinstance(r, str) and r.strip():
                            refs.append(r.strip())
        return refs

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append("<html>")
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append("<title>Slow Brain, Fast Planner Run Report</title>")
    parts.append(
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;"
        "margin:16px;color:#111;background:#fff;}"
        "code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;}"
        ".muted{color:#666;}"
        ".kv{display:flex;gap:12px;flex-wrap:wrap;}"
        ".kv .item{background:#f6f6f6;border:1px solid #e7e7e7;border-radius:8px;padding:8px 10px;}"
        ".sample{border:1px solid #e7e7e7;border-radius:10px;padding:10px;margin:12px 0;}"
        ".row{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap;}"
        ".col{flex:1;min-width:320px;}"
        ".imgwrap{background:#111;border-radius:10px;padding:8px;}"
        "img{max-width:100%;height:auto;border-radius:6px;display:block;}"
        "pre{white-space:pre-wrap;word-break:break-word;background:#0b1020;color:#f5f5f5;"
        "padding:10px;border-radius:10px;overflow:auto;}"
        ".badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;"
        "border:1px solid #ddd;background:#fafafa;margin-left:8px;}"
        ".ok{border-color:#b7e3b7;background:#edfff0;}"
        ".bad{border-color:#f2b8b8;background:#fff0f0;}"
        ".msg{border:1px solid #e7e7e7;border-radius:10px;padding:10px;margin:10px 0;background:#fff;}"
        ".controls{border:1px solid #e7e7e7;border-radius:10px;padding:10px;margin:12px 0;background:#fafafa;}"
        ".msg-collapsible{border:1px solid #e7e7e7;border-radius:10px;margin:10px 0;background:#fff;}"
        ".msg-collapsible summary{cursor:pointer;padding:10px;background:#fafafa;border-radius:10px;}"
        ".msg-collapsible summary:hover{background:#f0f0f0;}"
        ".msg-collapsible[open] summary{border-bottom:1px solid #e7e7e7;border-radius:10px 10px 0 0;}"
        ".msg-content{padding:10px;}"
        ".controls button{padding:6px 12px;cursor:pointer;border-radius:6px;border:1px solid #ddd;background:#fff;}"
        ".controls button.active{background:#007bff;color:#fff;border-color:#0056b3;}"
        "</style>"
    )
    parts.append("</head>")
    parts.append("<body>")

    parts.append("<h2>Slow Brain, Fast Planner run report</h2>")
    parts.append(f'<div class="muted">Run dir: <code>{html.escape(str(run_dir))}</code></div>')

    # Compact summary at the top (human-friendly).
    if metrics is not None:
        parts.append("<h3>Metrics summary</h3>")
        items: list[tuple[str, Any]] = []
        for k in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "takeover_rate",
            "accuracy_vs_min_ade",
            "accuracy_vs_score",
            "ade_model_avg",
            "ade_score_avg",
            "ade_min_avg",
            "ade_count",
            "goal_distance_m_avg",
            "goal_distance_m_count",
            "selected_end_dist_to_goal_m_avg",
            "selected_goal_ang_diff_deg_avg",
            "selected_traj_avg_dist_to_goal_m_avg",
            "selected_goal_progress_m_avg",
            "maoe_deg_avg",
            "dcr_avg",
            "tcr_avg",
            "selected_end_dist_to_goal_m_count",
            "selected_goal_ang_diff_deg_count",
            "selected_traj_avg_dist_to_goal_m_count",
            "selected_goal_progress_m_count",
            "route_dev_model_avg",
            "route_dev_score_avg",
            "route_dev_min_avg",
            "route_dev_count",
            "accuracy_vs_route_min",
            "accuracy_vs_route_score",
            "auto_enabled_true_frac",
            "auto_enabled_true_count",
            "auto_enabled_false_count",
            "auto_enabled_missing_count",
            "top3_accuracy",
            "top5_accuracy",
            "snapshots_evaluated",
            "snapshots_total",
            "snapshots_skipped",
            "episodes_total",
            "episodes_schema_invalid",
        ]:
            if k in metrics:
                items.append((k, metrics.get(k)))
        if items:
            parts.append('<div class="kv">')
            for k, v in items:
                parts.append(
                    f'<div class="item"><div class="muted">{escape(str(k))}</div><div><code>{escape(_fmt_metric_value(v))}</code></div></div>'
                )
            parts.append("</div>")

        # Full metrics table (flattened) for easy scanning / copy.
        parts.append("<h3>All metrics</h3>")
        parts.append(
            '<div class="muted">Flattened view of <code>metrics.json</code> (rounded floats).</div>'
        )
        flat = _flatten_metrics(metrics)
        if flat:
            parts.append(
                "<style>"
                "table.metrics{border-collapse:collapse;width:100%;max-width:1100px;}"
                "table.metrics th,table.metrics td{border:1px solid #e7e7e7;padding:6px 8px;text-align:left;vertical-align:top;}"
                "table.metrics th{background:#fafafa;position:sticky;top:0;}"
                "table.metrics td.key{white-space:nowrap;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;}"
                "</style>"
            )
            parts.append("<details open>")
            parts.append("<summary><strong>Show/Hide all metrics table</strong></summary>")
            parts.append(
                "<div style='overflow:auto;max-height:520px;border:1px solid #eee;border-radius:10px;padding:6px;margin-top:8px;'>"
            )
            parts.append("<table class='metrics'>")
            parts.append("<thead><tr><th>key</th><th>value</th></tr></thead>")
            parts.append("<tbody>")
            for kk, vv in flat:
                parts.append(
                    "<tr>"
                    f"<td class='key'>{escape(str(kk))}</td>"
                    f"<td><code>{escape(_fmt_metric_value(vv))}</code></td>"
                    "</tr>"
                )
            parts.append("</tbody></table></div></details>")

    # Episode-level maps (written by evaluation scripts under artifacts/maps).
    maps_dir = run_dir / "artifacts" / "maps"
    if maps_dir.is_dir():
        map_paths = sorted([p for p in maps_dir.glob("*.png") if p.is_file()])
        if map_paths:
            parts.append("<h3>Episode maps</h3>")
            parts.append(
                '<div class="muted">These are episode-level artifacts (shown once, not per snapshot).</div>'
            )
            for mp in map_paths:
                try:
                    img_src = _image_src(mp, embed=embed_images, run_dir=run_dir)
                    parts.append('<div class="sample">')
                    parts.append(f"<div><strong>{html.escape(mp.name)}</strong></div>")
                    parts.append('<div class="imgwrap">')
                    parts.append(
                        f'<img src="{img_src}" alt="episode_map" style="max-height:520px;">'
                    )
                    parts.append("</div>")
                    parts.append("</div>")
                except Exception:
                    parts.append(f"<div class='muted'><code>{html.escape(str(mp))}</code></div>")

    # Keep full config/metrics available, but behind <details> so the top of the report stays clean.
    if cfg is not None:
        parts.append("<details>")
        parts.append("<summary><strong>Config (raw JSON)</strong></summary>")
        parts.append("<pre>" + escape_pre(json.dumps(cfg, indent=2, sort_keys=True)) + "</pre>")
        parts.append("</details>")
    if metrics is not None:
        parts.append("<details>")
        parts.append("<summary><strong>Metrics (raw JSON)</strong></summary>")
        parts.append("<pre>" + escape_pre(json.dumps(metrics, indent=2, sort_keys=True)) + "</pre>")
        parts.append("</details>")

    parts.append("<h3>Samples</h3>")
    parts.append(f'<div class="muted">Total samples: {len(samples)}</div>')
    parts.append('<div id="samples-container">')

    # Report controls (filters/sorting/search). Historically these were only shown when
    # `auto_enabled` existed, but Task2 runs often omit that field; we still want ADE-improvement
    # browsing in those reports.
    has_auto_true = any(s.auto_enabled is True for s in samples)
    has_auto_false = any(s.auto_enabled is False for s in samples)
    if samples:
        # Default behavior:
        # - For takeover reports: hide auto_enabled=True (negatives) unless toggled on.
        # - For other reports: show all by default.
        checked_attr = "" if default_hide_auto_true else " checked"
        parts.append('<div class="controls">')
        parts.append("<div><strong>Controls</strong></div>")
        parts.append("<div style='margin-top:8px;'>")
        if has_auto_true or has_auto_false:
            parts.append(
                "<label>"
                f"<input type='checkbox' id='showAutoTrue'{checked_attr}> "
                "Show auto_enabled=True samples (auto)"
                "</label>"
            )
        else:
            parts.append(
                "<div class='muted'>auto_enabled not present in this run (no auto filter).</div>"
            )
        parts.append("</div>")
        parts.append("<div style='margin-top:10px;'>")
        parts.append("<strong>Sort order:</strong> ")
        parts.append(
            "<button id='sortADE' class='active' onclick='_sortByADE()'>Highest ADE first</button> "
        )
        parts.append(
            "<button id='sortOriginal' onclick='_sortByOriginal()'>Original order</button> "
        )
        parts.append(
            "<button id='sortBestImprovement' onclick='_sortByBestImprovement()'>Highest ADE improvement</button> "
        )
        parts.append(
            "<button id='sortWorstImprovement' onclick='_sortByWorstImprovement()'>Lowest ADE improvement</button>"
        )
        parts.append("</div>")
        parts.append("<div style='margin-top:10px;'>")
        parts.append("<strong>Quick search:</strong> ")
        parts.append(
            "<input type='text' id='quickSearch' placeholder='Type to filter by title/stats...' "
            "style='padding:6px;border-radius:6px;border:1px solid #ddd;width:300px;' "
            "oninput='_applySearch()'>"
        )
        parts.append("</div>")
        parts.append("</div>")
        parts.append(
            "<script>"
            "function _scrollToSamplesTop(){"
            "  const container=document.getElementById('samples-container');"
            "  if(!container){return;}"
            "  const y=container.getBoundingClientRect().top + window.scrollY - 8;"
            "  window.scrollTo({top:y,behavior:'smooth'});"
            "}"
            "function _applyAutoFilter(){"
            "  const cb=document.getElementById('showAutoTrue');"
            "  if(!cb){return;}"
            "  const show=cb.checked;"
            "  document.querySelectorAll('.sample[data-auto-enabled=\"true\"]').forEach((el)=>{"
            "    el.style.display = show ? '' : 'none';"
            "  });"
            "}"
            "function _clearSortButtons(){"
            "  ['sortADE','sortOriginal','sortBestImprovement','sortWorstImprovement'].forEach(id=>{"
            "    const el=document.getElementById(id);"
            "    if(el)el.classList.remove('active');"
            "  });"
            "}"
            "function _sortByADE(){"
            "  const container=document.getElementById('samples-container');"
            "  if(!container){return;}"
            "  const samples=Array.from(container.querySelectorAll('.sample'));"
            "  samples.sort((a,b)=>parseFloat(b.getAttribute('data-ade')||-1)-parseFloat(a.getAttribute('data-ade')||-1));"
            "  samples.forEach(s=>container.appendChild(s));"
            "  _clearSortButtons();"
            "  const b=document.getElementById('sortADE'); if(b)b.classList.add('active');"
            "  _scrollToSamplesTop();"
            "}"
            "function _sortByOriginal(){"
            "  const container=document.getElementById('samples-container');"
            "  if(!container){return;}"
            "  const samples=Array.from(container.querySelectorAll('.sample'));"
            "  samples.sort((a,b)=>parseInt(a.getAttribute('data-original-index')||0)-parseInt(b.getAttribute('data-original-index')||0));"
            "  samples.forEach(s=>container.appendChild(s));"
            "  _clearSortButtons();"
            "  const b=document.getElementById('sortOriginal'); if(b)b.classList.add('active');"
            "  _scrollToSamplesTop();"
            "}"
            "function _sortByBestImprovement(){"
            "  const container=document.getElementById('samples-container');"
            "  if(!container){return;}"
            "  const samples=Array.from(container.querySelectorAll('.sample'));"
            "  samples.sort((a,b)=>parseFloat(b.getAttribute('data-ade-improvement')||0)-parseFloat(a.getAttribute('data-ade-improvement')||0));"
            "  samples.forEach(s=>container.appendChild(s));"
            "  _clearSortButtons();"
            "  const b=document.getElementById('sortBestImprovement'); if(b)b.classList.add('active');"
            "  _scrollToSamplesTop();"
            "}"
            "function _sortByWorstImprovement(){"
            "  const container=document.getElementById('samples-container');"
            "  if(!container){return;}"
            "  const samples=Array.from(container.querySelectorAll('.sample'));"
            "  samples.sort((a,b)=>parseFloat(a.getAttribute('data-ade-improvement')||0)-parseFloat(b.getAttribute('data-ade-improvement')||0));"
            "  samples.forEach(s=>container.appendChild(s));"
            "  _clearSortButtons();"
            "  const b=document.getElementById('sortWorstImprovement'); if(b)b.classList.add('active');"
            "  _scrollToSamplesTop();"
            "}"
            "function _applySearch(){"
            "  const q=document.getElementById('quickSearch');"
            "  const term=(q && q.value ? q.value : '').toLowerCase();"
            "  const cb=document.getElementById('showAutoTrue');"
            "  const showAuto=cb?cb.checked:true;"
            "  document.querySelectorAll('.sample').forEach((el)=>{"
            "    const isAutoTrue = el.getAttribute('data-auto-enabled') === 'true';"
            "    if(isAutoTrue && !showAuto){"
            "      el.style.display='none';"
            "      return;"
            "    }"
            "    const text=(el.innerText||'').toLowerCase();"
            "    el.style.display = text.includes(term) ? '' : 'none';"
            "  });"
            "}"
            "document.addEventListener('DOMContentLoaded', ()=>{"
            "  const cb=document.getElementById('showAutoTrue');"
            "  if(cb){cb.addEventListener('change', ()=>{_applyAutoFilter(); _applySearch();});}"
            "  _applyAutoFilter();"
            "  _applySearch();"
            "});"
            "</script>"
        )

    for i, s in enumerate(samples):
        badge = ""
        if s.skipped:
            badge = '<span class="badge">skipped</span>'
        elif s.correct is True:
            badge = '<span class="badge ok">correct</span>'
        elif s.correct is False:
            badge = '<span class="badge bad">wrong</span>'
        else:
            badge = '<span class="badge">unknown</span>'

        ade_text = ""
        try:
            if isinstance(s.ade, dict):
                amin = s.ade.get("min")
                ascore = s.ade.get("score")
                asel = s.ade.get("selected")
                if amin is not None or ascore is not None or asel is not None:

                    def _f(v):
                        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

                    ade_text = f"  ade(sel/min/score)={_f(asel)}/{_f(amin)}/{_f(ascore)}"
                    # Add improvement (positive = VLM better)
                    if isinstance(ascore, (int, float)) and isinstance(asel, (int, float)):
                        imp = float(ascore) - float(asel)
                        sign = "+" if imp >= 0 else ""
                        ade_text += f"  improve={sign}{imp:.4f}"
        except Exception:
            ade_text = ""

        auto_text = ""
        if s.auto_enabled is True:
            auto_text = "  auto_enabled=True"
        elif s.auto_enabled is False:
            auto_text = "  auto_enabled=False"

        title = (
            f"#{i}  ep={s.episode_id}  t={s.t:.3f}  clip={s.snapshot_index}  "
            f"pred={s.prediction}  label={s.label}"
            f"{auto_text}{ade_text}"
        )
        try:
            gd = s.goal_distance_m
            if gd is not None:
                gdf = float(gd)
                if gdf == gdf and gdf != float("inf") and gdf != float("-inf"):
                    title = title + f"  goal_dist_m={gdf:.2f}"
        except Exception:
            pass
        try:
            ed = s.selected_end_dist_to_goal_m
            if ed is not None:
                edf = float(ed)
                if edf == edf and edf != float("inf") and edf != float("-inf"):
                    title = title + f"  end2goal_m={edf:.2f}"
        except Exception:
            pass
        try:
            ad = s.selected_goal_ang_diff_deg
            if ad is not None:
                adf = float(ad)
                if adf == adf and adf != float("inf") and adf != float("-inf"):
                    title = title + f"  ang_diff_deg={adf:.1f}"
        except Exception:
            pass
        try:
            avgd = s.selected_traj_avg_dist_to_goal_m
            if avgd is not None:
                avgf = float(avgd)
                if avgf == avgf and avgf != float("inf") and avgf != float("-inf"):
                    title = title + f"  avg2goal_m={avgf:.2f}"
        except Exception:
            pass
        try:
            prog = s.selected_goal_progress_m
            if prog is not None:
                pf = float(prog)
                if pf == pf and pf != float("inf") and pf != float("-inf"):
                    title = title + f"  progress_m={pf:.2f}"
        except Exception:
            pass
        try:
            m = s.maoe_deg
            if m is not None:
                mf = float(m)
                if mf == mf and mf != float("inf") and mf != float("-inf"):
                    title = title + f"  maoe_deg={mf:.1f}"
        except Exception:
            pass
        try:
            dcr = s.dcr
            tcr = s.tcr
            if dcr is not None:
                df = float(dcr)
                if df == df and df != float("inf") and df != float("-inf"):
                    title = title + f"  dcr={df:.3f}"
            if tcr is not None:
                tf = float(tcr)
                if tf == tf and tf != float("inf") and tf != float("-inf"):
                    title = title + f"  tcr={tf:.3f}"
        except Exception:
            pass
        auto_attr = "unknown"
        if s.auto_enabled is True:
            auto_attr = "true"
        elif s.auto_enabled is False:
            auto_attr = "false"

        ade_val = -1.0
        ade_improvement = 0.0
        if isinstance(s.ade, dict):
            val = s.ade.get("selected")
            if isinstance(val, (int, float)):
                ade_val = float(val)
            # Compute ADE improvement = ade_score - ade_selected (positive means VLM is better)
            ade_score = s.ade.get("score")
            ade_selected = s.ade.get("selected")
            if isinstance(ade_score, (int, float)) and isinstance(ade_selected, (int, float)):
                ade_improvement = float(ade_score) - float(ade_selected)

        parts.append(
            f'<div class="sample" data-auto-enabled="{auto_attr}" data-ade="{ade_val}" data-ade-improvement="{ade_improvement:.6f}" data-original-index="{s.original_index}">'
        )
        parts.append(f"<div><strong>{html.escape(title)}</strong>{badge}</div>")
        parts.append('<div class="row">')

        # Image column.
        parts.append('<div class="col">')
        # If the prompt contains image_ref(s), show the last one as a "prompt overlay" preview.
        # This is especially useful for hierarchical selectors that re-render intermediate overlays.
        try:
            pref = None
            refs = _collect_image_refs_from_messages(s.prompt_messages)
            if refs:
                pref = refs[-1]
            if pref:
                rp = _resolve_ref_in_run(str(pref))
                if rp is not None and rp.exists():
                    img_src = _image_src(rp, embed=embed_images, run_dir=run_dir)
                    parts.append('<div class="imgwrap">')
                    parts.append(f'<img src="{img_src}" alt="prompt_overlay">')
                    parts.append("</div>")
                    parts.append(
                        f'<div class="muted">prompt image_ref: <code>{html.escape(str(pref))}</code></div>'
                    )
        except Exception:
            pass
        # Overlay image (planner candidates).
        if s.overlay_frame_ref is not None:
            overlay_abs = run_dir / Path(s.overlay_frame_ref)
            if overlay_abs.exists():
                img_src = _image_src(overlay_abs, embed=embed_images, run_dir=run_dir)
                parts.append('<div class="imgwrap">')
                parts.append(f'<img src="{img_src}" alt="overlay">')
                parts.append("</div>")
                parts.append(
                    f'<div class="muted">overlay: <code>{html.escape(s.overlay_frame_ref)}</code></div>'
                )
            else:
                parts.append("<div class='muted'>overlay missing on disk</div>")
                parts.append(
                    f'<div class="muted">overlay: <code>{html.escape(str(overlay_abs))}</code></div>'
                )
        else:
            parts.append("<div class='muted'>overlay_frame_ref: null</div>")
        if s.overlay_error:
            parts.append(
                f"<div class='muted'>overlay_error: {html.escape(str(s.overlay_error))}</div>"
            )

        # Local-frame plot (selected vs GT vs route).
        if s.local_plot_frame_ref is not None:
            lp_abs = run_dir / Path(s.local_plot_frame_ref)
            if lp_abs.exists():
                img_src = _image_src(lp_abs, embed=embed_images, run_dir=run_dir)
                parts.append('<div class="imgwrap" style="margin-top:10px">')
                parts.append(f'<img src="{img_src}" alt="local_plot">')
                parts.append("</div>")
                parts.append(
                    f'<div class="muted">local_plot: <code>{html.escape(s.local_plot_frame_ref)}</code></div>'
                )
            else:
                parts.append("<div class='muted'>local_plot missing on disk</div>")
                parts.append(
                    f'<div class="muted">local_plot: <code>{html.escape(str(lp_abs))}</code></div>'
                )
        else:
            parts.append("<div class='muted'>local_plot_frame_ref: null</div>")
        if s.local_plot_error:
            parts.append(
                f"<div class='muted'>local_plot_error: {html.escape(str(s.local_plot_error))}</div>"
            )

        parts.append("</div>")

        # Text column.
        parts.append('<div class="col">')
        # Extra metrics (if present).
        if any(
            x is not None for x in (s.traj_len_m, s.maoe_deg, s.dcr, s.tcr, s.compliance_source)
        ):
            parts.append("<div><strong>Open-loop metrics</strong></div>")
            parts.append("<div class='kv'>")
            for kk, vv in [
                ("traj_len_m", s.traj_len_m),
                ("maoe_deg", s.maoe_deg),
                ("dcr", s.dcr),
                ("tcr", s.tcr),
                ("compliance_source", s.compliance_source),
            ]:
                if vv is None:
                    continue
                parts.append(
                    f'<div class="item"><div class="muted">{escape(str(kk))}</div><div><code>{escape(_fmt_metric_value(vv))}</code></div></div>'
                )
            parts.append("</div>")
        parts.append("<div><strong>Prompt (rendered)</strong></div>")
        if s.prompt_messages is None:
            parts.append("<div class='muted'>No model_call trace found for this snapshot.</div>")
        else:
            parts.append(
                render_messages_human(
                    s.prompt_messages, run_dir=run_dir, embed_images=bool(embed_images)
                )
            )

            # Keep the raw JSON around for debugging (collapsed).
            parts.append("<details>")
            parts.append("<summary><strong>Prompt (raw JSON)</strong></summary>")
            parts.append(
                "<pre>"
                + escape_pre(json.dumps(s.prompt_messages, indent=2, sort_keys=True))
                + "</pre>"
            )
            parts.append("</details>")

        # Main (visible) model output: pretty JSON if possible.
        parts.append("<div><strong>Model output</strong></div>")
        raw_missing = s.raw_model_response is None or (
            isinstance(s.raw_model_response, str)
            and (s.raw_model_response.strip() == "" or s.raw_model_response.strip() == "None")
        )
        if raw_missing:
            parts.append(
                "<div class='muted'>No usable raw_response captured for this snapshot.</div>"
            )
            if s.parse_note:
                parts.append(
                    "<div style='margin-top:8px;padding:8px;background:#fff0f0;border:1px solid #ffcccc;border-radius:6px;'>"
                    "<strong>Parse note:</strong> "
                    f"<code>{escape(str(s.parse_note))}</code>"
                    "</div>"
                )
            if s.rationale:
                parts.append(
                    "<div style='margin-top:8px;padding:8px;background:#fff0f0;border:1px solid #ffcccc;border-radius:6px;'>"
                    "<strong>Rationale (e.g. error/exception):</strong>"
                    f"<pre style='background:none;border:none;padding:0;margin:0;color:#d00;'>{escape_pre(str(s.rationale))}</pre>"
                    "</div>"
                )
            if s.parsed_advice is not None:
                parts.append("<details>")
                parts.append("<summary><strong>Parsed advice (trace)</strong></summary>")
                parts.append(
                    "<pre>"
                    + escape_pre(json.dumps(s.parsed_advice, indent=2, sort_keys=True))
                    + "</pre>"
                )
                parts.append("</details>")
        else:
            if isinstance(s.raw_model_response, str):
                pretty = pretty_json_from_raw(s.raw_model_response)
                parts.append(
                    "<pre>"
                    + escape_pre(pretty if pretty is not None else s.raw_model_response)
                    + "</pre>"
                )
            else:
                parts.append(
                    "<pre>"
                    + escape_pre(json.dumps(s.raw_model_response, indent=2, sort_keys=True))
                    + "</pre>"
                )

        # Hierarchical selectors: render per-stage model outputs if provided.
        try:
            pm = s.provider_meta
            hier = pm.get("hierarchical") if isinstance(pm, dict) else None
            calls = hier.get("calls") if isinstance(hier, dict) else None
            raw_by_call = hier.get("raw_by_call") if isinstance(hier, dict) else None
            parse_note_by_call = hier.get("parse_note_by_call") if isinstance(hier, dict) else None
            if isinstance(calls, list) and calls:
                parts.append("<details open>")
                parts.append("<summary><strong>Hierarchical calls (per-stage)</strong></summary>")
                parts.append(
                    "<div class='muted'>Each call is one model request (rep-selection or leaf-selection).</div>"
                )
                for j, c in enumerate(calls):
                    if not isinstance(c, dict):
                        continue
                    stage = c.get("stage")
                    overlay_ref = c.get("overlay_ref")
                    sel = c.get("selected_index")
                    pnote = c.get("parse_note")
                    parts.append("<div class='msg' style='margin-top:10px'>")
                    parts.append(
                        f"<div><strong>Call {j + 1}</strong> <span class='badge'>{escape(str(stage))}</span></div>"
                    )
                    parts.append(
                        "<div class='muted'>"
                        f"selected_index: <code>{escape(str(sel))}</code> &nbsp;&nbsp; "
                        f"parse_note: <code>{escape(str(pnote))}</code>"
                        "</div>"
                    )
                    if isinstance(overlay_ref, str) and overlay_ref.strip():
                        parts.append(
                            f"<div class='muted'>overlay_ref: <code>{escape(str(overlay_ref))}</code></div>"
                        )
                    raw2 = c.get("raw")
                    if isinstance(raw2, str):
                        pretty2 = pretty_json_from_raw(raw2)
                        parts.append(
                            "<pre>"
                            + escape_pre(pretty2 if pretty2 is not None else raw2)
                            + "</pre>"
                        )
                    else:
                        parts.append(
                            "<pre>"
                            + escape_pre(json.dumps(raw2, indent=2, sort_keys=True))
                            + "</pre>"
                        )
                    parts.append("</div>")
                parts.append("</details>")
            elif isinstance(raw_by_call, list) and raw_by_call:
                parts.append("<details open>")
                parts.append("<summary><strong>Hierarchical calls (per-stage)</strong></summary>")
                parts.append(
                    "<div class='muted'>Per-call raw outputs (aggregated by the hierarchical advisor).</div>"
                )
                for j, raw2 in enumerate(raw_by_call):
                    pnote = None
                    if isinstance(parse_note_by_call, list) and j < len(parse_note_by_call):
                        pnote = parse_note_by_call[j]
                    parts.append("<div class='msg' style='margin-top:10px'>")
                    parts.append(f"<div><strong>Call {j + 1}</strong></div>")
                    parts.append(
                        f"<div class='muted'>parse_note: <code>{escape(str(pnote))}</code></div>"
                    )
                    if isinstance(raw2, str):
                        pretty2 = pretty_json_from_raw(raw2)
                        parts.append(
                            "<pre>"
                            + escape_pre(pretty2 if pretty2 is not None else raw2)
                            + "</pre>"
                        )
                    else:
                        parts.append(
                            "<pre>"
                            + escape_pre(json.dumps(raw2, indent=2, sort_keys=True))
                            + "</pre>"
                        )
                    parts.append("</div>")
                parts.append("</details>")
        except Exception:
            pass

        # Hidden-by-default: raw model output (verbatim).
        parts.append("<details>")
        parts.append("<summary><strong>Model raw output (verbatim)</strong></summary>")
        if raw_missing:
            parts.append(
                "<div class='muted'>No usable raw_response captured for this snapshot.</div>"
            )
        elif isinstance(s.raw_model_response, str):
            parts.append("<pre>" + escape_pre(s.raw_model_response) + "</pre>")
        else:
            parts.append(
                "<pre>"
                + escape_pre(json.dumps(s.raw_model_response, indent=2, sort_keys=True))
                + "</pre>"
            )
        parts.append("</details>")

        # NOTE: We intentionally do NOT render a separate "Model rationale" block.
        # The model's "reason" / "reasoning" should be visible inside the (pretty-printed) JSON output above.

        # Optional: model reasoning / thinking trace (if provided by the backend).
        if s.thoughts:
            parts.append("<div><strong>Model thoughts (if provided)</strong></div>")
            if isinstance(s.thoughts, str):
                parts.append("<pre>" + escape_pre(s.thoughts) + "</pre>")
            else:
                parts.append(
                    "<pre>"
                    + escape_pre(json.dumps(s.thoughts, indent=2, sort_keys=True))
                    + "</pre>"
                )

        if s.skipped:
            parts.append("<div><strong>Skip reason</strong></div>")
            parts.append("<pre>" + escape_pre(str(s.skip_reason)) + "</pre>")

        parts.append("</div>")  # col
        parts.append("</div>")  # row
        parts.append("</div>")  # sample
    parts.append("</div>")  # samples-container

    parts.append("</body>")
    parts.append("</html>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out_path
