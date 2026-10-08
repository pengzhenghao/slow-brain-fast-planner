from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol

from slow_brain_fast_planner.benchmarks.model_adapters import ModelAdapter
from slow_brain_fast_planner.benchmarks.vqa_trajectory import (
    VQATrajectoryPromptConfig,
    build_vqa_trajectory_selection_messages,
    parse_vqa_trajectory_action,
)
from slow_brain_fast_planner.utils.io import utc_now_iso

VlmAdvice = dict[str, Any]
QueryBundle = Mapping[str, Any]

# Versioned, portable cache schema (JSONL).
CACHE_SCHEMA_VERSION = "slow_brain_fast_planner.vlm_advice_cache/0.1"


class VlmAdvisor(Protocol):
    """VLM advisor interface (trajectory selection).

    The benchmark treats the advisor like a pure function:
      advise(query_bundle) -> advice

    Determinism is enforced via record/replay caches keyed by:
      (query_bundle_hash, model_id, prompt_version, decoding_config)
    """

    def advise(self, query_bundle: QueryBundle) -> VlmAdvice: ...


def _sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _is_probably_url(s: str) -> bool:
    s2 = str(s).strip().lower()
    return s2.startswith("http://") or s2.startswith("https://") or s2.startswith("data:image/")


def _read_file_sha256_hex(path: Path) -> str | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        return _sha256_hex(path.read_bytes())
    except Exception:
        return None


def _to_jsonable(x: Any) -> Any:
    """Best-effort conversion to a JSON-serializable structure.

    This is used ONLY for hashing / caching. It should be stable and avoid large binary blobs.
    """

    if x is None or isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        if math.isfinite(x):
            return float(x)
        return {"__nonfinite_float__": str(x)}
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (bytes, bytearray, memoryview)):
        b = bytes(x)
        # Avoid embedding raw bytes; store a short marker + hash.
        return {"__bytes_sha256__": _sha256_hex(b), "__bytes_len__": len(b)}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        # Coerce keys to strings for stable JSON.
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    # Pydantic v2 models (and similar) often expose `model_dump`.
    try:
        md = getattr(x, "model_dump", None)
        if callable(md):
            return _to_jsonable(md(mode="json"))
    except Exception:
        pass

    # numpy scalars / arrays (optional dependency in this repo).
    try:
        import numpy as np  # type: ignore

        if isinstance(x, np.generic):
            return _to_jsonable(x.item())
        if isinstance(x, np.ndarray):
            return _to_jsonable(x.tolist())
    except Exception:
        pass

    # Fallback: string repr (stable enough for caching keys in practice).
    return {"__repr__": repr(x)}


def canonical_json(obj: Any) -> str:
    """Canonical JSON serialization for stable hashing and cache keys."""

    # Ensure we only serialize JSON-safe objects.
    j = _to_jsonable(obj)
    return json.dumps(j, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _collect_image_refs(obj: Any) -> list[str]:
    """Collect image reference strings inside an arbitrary nested structure."""

    refs: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            # Common patterns used in this repo:
            # - {"type":"image_ref","image_ref": "..."}
            # - {"vision":{"overlay_frame_ref":"..."}}
            for k, v in x.items():
                kk = str(k)
                if kk in ("image_ref", "overlay_frame_ref", "rgb_frame_ref"):
                    if isinstance(v, str) and v.strip():
                        refs.append(v.strip())
                if kk in ("history_frame_refs",):
                    if isinstance(v, list):
                        for vv in v:
                            if isinstance(vv, str) and vv.strip():
                                refs.append(vv.strip())
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    # Preserve order but dedup.
    out: list[str] = []
    seen: set[str] = set()
    for r in refs:
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def normalize_query_bundle(query_bundle: QueryBundle, *, base_dir: Path | None) -> dict[str, Any]:
    """Normalize a query bundle for hashing, including hashing any referenced images."""

    qb = _to_jsonable(dict(query_bundle))
    if not isinstance(qb, dict):
        qb = {"__query_bundle__": qb}

    img_hashes: dict[str, str | None] = {}
    for ref in _collect_image_refs(qb):
        if _is_probably_url(ref):
            img_hashes[ref] = None
            continue
        p = Path(ref)
        if not p.is_absolute():
            p = (base_dir or Path.cwd()) / p
        p = p.resolve()
        img_hashes[ref] = _read_file_sha256_hex(p)

    if img_hashes:
        qb.setdefault("__resolved_refs__", {})
        if isinstance(qb["__resolved_refs__"], dict):
            qb["__resolved_refs__"]["image_sha256_by_ref"] = img_hashes
    return qb


def query_bundle_hash(query_bundle: QueryBundle, *, base_dir: Path | None) -> str:
    qb_norm = normalize_query_bundle(query_bundle, base_dir=base_dir)
    return _sha256_hex(canonical_json(qb_norm).encode("utf-8"))


def decoding_config_hash(decoding_config: Mapping[str, Any]) -> str:
    return _sha256_hex(canonical_json(dict(decoding_config)).encode("utf-8"))


@dataclass(frozen=True)
class VlmAdviceCacheKey:
    """Stable cache identity for one prompt, model, and decoding configuration."""

    query_bundle_hash: str
    model_id: str
    prompt_version: str
    decoding_config: dict[str, Any]

    @property
    def decoding_config_hash(self) -> str:
        return decoding_config_hash(self.decoding_config)

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (
            str(self.query_bundle_hash),
            str(self.model_id),
            str(self.prompt_version),
            str(self.decoding_config_hash),
        )

    def to_compact_string(self) -> str:
        qh, mid, pv, dch = self.as_tuple()
        return f"{qh}:{mid}:{pv}:{dch}"


class VlmCacheMissError(KeyError):
    def __init__(self, key: VlmAdviceCacheKey) -> None:
        super().__init__(f"vlm_cache_miss:{key.to_compact_string()}")
        self.key = key


class VlmAdviceCache:
    """Append-only JSONL cache for VLM advisor calls (record/replay)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict[str, Any]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
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
                    key = rec.get("key")
                    if not isinstance(key, dict):
                        continue
                    key_str = key.get("key_str")
                    if isinstance(key_str, str) and key_str:
                        self._index[key_str] = rec
        except Exception:
            # Best-effort: treat cache as empty on read errors.
            self._index = {}

    def get_record(self, key: VlmAdviceCacheKey) -> dict[str, Any] | None:
        return self._index.get(key.to_compact_string())

    def put_record(self, record: dict[str, Any]) -> None:
        # Append-only.
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        key = record.get("key")
        if isinstance(key, dict):
            key_str = key.get("key_str")
            if isinstance(key_str, str) and key_str:
                self._index[key_str] = record

    def put(
        self,
        *,
        key: VlmAdviceCacheKey,
        query_bundle: QueryBundle,
        advice: VlmAdvice,
        base_dir: Path | None,
        raw_response: str | None,
        messages: list[dict[str, Any]] | None,
        thoughts: list[str] | None,
        latency_ms: float | None,
        provider_meta: dict[str, Any] | None,
    ) -> None:
        record: dict[str, Any] = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "created_at_utc": utc_now_iso(),
            "key": {
                "key_str": key.to_compact_string(),
                "query_bundle_hash": str(key.query_bundle_hash),
                "model_id": str(key.model_id),
                "prompt_version": str(key.prompt_version),
                "decoding_config_hash": str(key.decoding_config_hash),
                "decoding_config": _to_jsonable(key.decoding_config),
            },
            # Store a normalized query bundle (portable, contains image hashes).
            "query_bundle": normalize_query_bundle(query_bundle, base_dir=base_dir),
            "value": {
                "advice": _to_jsonable(advice),
                "raw_response": (None if raw_response is None else str(raw_response)),
                "messages": _to_jsonable(messages) if messages is not None else None,
                "thoughts": _to_jsonable(thoughts) if thoughts is not None else None,
                "latency_ms": (None if latency_ms is None else float(latency_ms)),
                "provider_meta": _to_jsonable(provider_meta) if provider_meta is not None else None,
            },
        }
        self.put_record(record)


class CachedVlmAdvisor:
    """Wrap a VlmAdvisor with record/replay caching.

    Modes:
    - replay_only: cache must contain every query; no live calls.
    - cache_first: replay if present; otherwise call backend and write-through.
    """

    def __init__(
        self,
        *,
        backend: VlmAdvisor,
        cache: VlmAdviceCache,
        mode: str,
        model_id: str,
        prompt_version: str,
        decoding_config: Mapping[str, Any],
        base_dir: Path | None,
    ) -> None:
        self.backend = backend
        self.cache = cache
        self.mode = str(mode)
        self.model_id = str(model_id)
        self.prompt_version = str(prompt_version)
        self.decoding_config = dict(decoding_config)
        self.base_dir = base_dir

        # Debug info for the last call (mirrors style of Gemini adapter).
        self.last_key: VlmAdviceCacheKey | None = None
        self.last_cache_hit: bool | None = None
        self.last_record: dict[str, Any] | None = None

        if self.mode not in ("replay_only", "cache_first"):
            raise ValueError(f"Unsupported VLM cache mode: {self.mode!r}")

    def _make_key(self, query_bundle: QueryBundle) -> VlmAdviceCacheKey:
        qh = query_bundle_hash(query_bundle, base_dir=self.base_dir)
        return VlmAdviceCacheKey(
            query_bundle_hash=qh,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            decoding_config=dict(self.decoding_config),
        )

    def advise(self, query_bundle: QueryBundle) -> VlmAdvice:
        key = self._make_key(query_bundle)
        self.last_key = key

        rec = self.cache.get_record(key)
        if rec is not None:
            self.last_cache_hit = True
            self.last_record = rec
            val = rec.get("value") if isinstance(rec.get("value"), dict) else {}
            advice = val.get("advice")
            if isinstance(advice, dict):
                return advice
            # Corrupt record; fall through to miss behavior.

        self.last_cache_hit = False
        self.last_record = None
        if self.mode == "replay_only":
            raise VlmCacheMissError(key)

        t0 = time.perf_counter()
        advice2 = self.backend.advise(query_bundle)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # Best-effort: pull useful debug payloads off the backend (if it exposes them).
        raw_response = getattr(self.backend, "last_raw_response", None)
        if raw_response is not None:
            raw_response = str(raw_response)
            if not raw_response.strip():
                raw_response = None
        messages = getattr(self.backend, "last_messages", None)
        thoughts = getattr(self.backend, "last_thoughts", None)

        provider_meta: dict[str, Any] | None = None
        try:
            provider_meta = dict(getattr(self.backend, "last_provider_meta", None) or {})
        except Exception:
            provider_meta = None

        self.cache.put(
            key=key,
            query_bundle=query_bundle,
            advice=advice2,
            base_dir=self.base_dir,
            raw_response=raw_response,
            messages=messages if isinstance(messages, list) else None,
            thoughts=thoughts if isinstance(thoughts, list) else None,
            latency_ms=latency_ms,
            provider_meta=provider_meta,
        )

        return advice2


class VQATrajectorySelectionAdvisor:
    """Task-specific VLM advisor: VQA-style "select a trajectory index"."""

    # Bump this when you change prompt wording or message structure in this class.
    PROMPT_VERSION = "vqa_traj_select_v12"

    def __init__(self, *, model: ModelAdapter, prompt_cfg: VQATrajectoryPromptConfig) -> None:
        self.model = model
        self.prompt_cfg = prompt_cfg

        # Debug info for traces/reports.
        self.last_messages: list[dict[str, Any]] | None = None
        self.last_raw_response: str | None = None
        self.last_parse_note: str | None = None
        self.last_thoughts: list[str] | None = None
        self.last_provider_meta: dict[str, Any] | None = None

    def advise(self, query_bundle: QueryBundle) -> VlmAdvice:
        # Reset debug state at start of call to avoid leaking stale data from previous calls
        # if this call raises an exception before completion.
        self.last_messages = None
        self.last_raw_response = None
        self.last_parse_note = None
        self.last_thoughts = None
        self.last_provider_meta = None

        qb = dict(query_bundle)
        overlay_frame_ref = qb.get("overlay_frame_ref")
        if overlay_frame_ref is None:
            vision = qb.get("vision")
            if isinstance(vision, dict):
                overlay_frame_ref = vision.get("overlay_frame_ref")

        num_candidates = qb.get("num_candidates")
        if num_candidates is None:
            # Fallback: infer from planner/candidates table.
            planner = qb.get("planner")
            if isinstance(planner, dict) and isinstance(planner.get("candidates"), list):
                num_candidates = len(planner.get("candidates") or [])
        if num_candidates is None:
            raise ValueError("query_bundle missing num_candidates")
        K = int(num_candidates)

        history_frame_refs = qb.get("history_frame_refs") if isinstance(qb.get("history_frame_refs"), list) else None

        # Visual prompting metadata (best-effort).
        image_info = qb.get("image_info") if isinstance(qb.get("image_info"), dict) else None
        if image_info is None:
            try:
                obs0 = qb.get("obs")
                if isinstance(obs0, dict) and isinstance(obs0.get("vision"), dict):
                    image_info = dict(obs0.get("vision") or {})
            except Exception:
                image_info = None

        messages = build_vqa_trajectory_selection_messages(
            cfg=self.prompt_cfg,
            overlay_frame_ref=None if overlay_frame_ref is None else str(overlay_frame_ref),
            history_frame_refs=(
                [str(x) for x in history_frame_refs if isinstance(x, str) and str(x).strip()]
                if history_frame_refs
                else None
            ),
            num_candidates=K,
            goal_text=(qb.get("goal_text") if isinstance(qb.get("goal_text"), str) else None),
            goal_xy=(qb.get("goal_xy") if isinstance(qb.get("goal_xy"), list) else None),
            goal_distance_m=(qb.get("goal_distance_m") if qb.get("goal_distance_m") is not None else None),
            candidate_scores=(qb.get("candidate_scores") if isinstance(qb.get("candidate_scores"), list) else None),
            image_info=image_info,
        )
        obs = qb.get("obs")
        if not isinstance(obs, dict):
            obs = qb.get("observation")
        if not isinstance(obs, dict):
            obs = {}

        t0 = time.perf_counter()
        raw = self.model.call(messages=messages, obs=obs)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        self.last_messages = messages
        raw_str = "" if raw is None else str(raw)
        # Important: don't stringify None into the literal string "None" (it shows up in reports as
        # a confusing "model output: None"). Prefer null + a parse_note like "empty_response".
        self.last_raw_response = raw_str if raw_str.strip() else None
        # Provider metadata (best-effort): token usage, request ids, etc.
        self.last_provider_meta = None
        try:
            usage = getattr(self.model, "last_usage", None)
            if isinstance(usage, dict):
                self.last_provider_meta = {"usage": usage, "latency_ms": float(latency_ms)}
            else:
                self.last_provider_meta = {"latency_ms": float(latency_ms)}
        except Exception:
            self.last_provider_meta = {"latency_ms": float(latency_ms)}

        # Some adapters expose per-call debug state (best-effort).
        thoughts: list[str] | None = None
        try:
            t = getattr(self.model, "last_thoughts", None)
            if isinstance(t, list):
                thoughts = [str(x) for x in t]
        except Exception:
            thoughts = None
        self.last_thoughts = thoughts

        action_obj, parse_note = parse_vqa_trajectory_action(raw_str, num_candidates=K)

        # One-shot retry for common non-JSON failure modes:
        # - Model prints a preamble like "Here is the JSON..." or starts a code fence but never emits the object.
        # This happens occasionally with some Gemini variants. Retrying with a very short, strict reminder
        # is usually enough and is cheaper than letting the whole evaluation silently degrade to "stop".
        if action_obj is None:
            try:
                s0 = str(raw_str or "")
            except Exception:
                s0 = ""
            looks_like_preamble = ("here is the json" in s0.lower()) or ("```" in s0)
            no_object_brace = ("{" not in s0)
            if looks_like_preamble or no_object_brace:
                try:
                    # Append a tiny hard constraint to the system prompt (keep everything else identical).
                    messages2 = []
                    for m in messages:
                        if isinstance(m, dict) and m.get("role") == "system":
                            sys_txt = m.get("content")
                            sys_txt2 = ("" if sys_txt is None else str(sys_txt)).rstrip() + (
                                "\n\nCRITICAL OUTPUT FORMAT: Reply with ONLY the JSON object. "
                                "No prose, no markdown, no ``` code fences. "
                                "The first character must be '{' and the last must be '}'."
                            )
                            messages2.append({"role": "system", "content": sys_txt2})
                        else:
                            messages2.append(m)
                    raw2 = self.model.call(messages=messages2, obs=obs)
                    raw2_str = "" if raw2 is None else str(raw2)
                    # If retry succeeds, we treat it as the effective response for parsing/debugging.
                    action_obj2, parse_note2 = parse_vqa_trajectory_action(raw2_str, num_candidates=K)
                    if action_obj2 is not None:
                        raw_str = raw2_str
                        self.last_messages = messages2
                        self.last_raw_response = raw_str if raw_str.strip() else None
                        action_obj = action_obj2
                        parse_note = f"retry_ok:{parse_note2}" if parse_note2 else "retry_ok"
                    else:
                        parse_note = f"retry_failed:{parse_note or 'parse_failed'}"
                except Exception:
                    # If the retry itself fails (API error etc), fall back to original parse result.
                    parse_note = parse_note or "parse_failed"

        self.last_parse_note = parse_note

        if action_obj is None:
            # This benchmark does not support ask_for_help; fall back to stop.
            return {"type": "stop", "confidence": 0.0, "debug": {"parse_note": parse_note or "parse_failed"}}

        act = action_obj.get("action")
        if act == "select_trajectory":
            try:
                idx = int(action_obj.get("selected_index"))
            except Exception:
                return {
                    "type": "stop",
                    "confidence": 0.0,
                    "debug": {"parse_note": "selected_index_not_int", "raw_action": _to_jsonable(action_obj)},
                }
            advice: dict[str, Any] = {"type": "select_trajectory", "selected_index": idx, "confidence": 0.0}
            rat = action_obj.get("rationale")
            if isinstance(rat, str) and rat.strip():
                advice["debug"] = {"rationale": rat.strip(), "parse_note": parse_note}
            elif parse_note:
                advice["debug"] = {"parse_note": parse_note}
            # Optional: surface model-identified hazards for analysis/debugging.
            co = action_obj.get("critical_object")
            if isinstance(co, list):
                advice["critical_object"] = [str(x) for x in co if str(x).strip()]
            elif isinstance(co, str) and co.strip():
                advice["critical_object"] = [co.strip()]
            risks = action_obj.get("risks")
            if isinstance(risks, list):
                dbg = advice.get("debug") if isinstance(advice.get("debug"), dict) else {}
                dbg["risks"] = [str(x) for x in risks if str(x).strip()]
                advice["debug"] = dbg
            return advice

        if act == "stop":
            out = {"type": "stop", "confidence": 0.0, "debug": {"parse_note": parse_note}}
            co = action_obj.get("critical_object")
            if isinstance(co, list):
                out["critical_object"] = [str(x) for x in co if str(x).strip()]
            elif isinstance(co, str) and co.strip():
                out["critical_object"] = [co.strip()]
            risks = action_obj.get("risks")
            if isinstance(risks, list):
                out_dbg = out.get("debug") if isinstance(out.get("debug"), dict) else {}
                out_dbg["risks"] = [str(x) for x in risks if str(x).strip()]
                out["debug"] = out_dbg
            return out

        # Unknown action: fall back to stop (no ask_for_help action in this benchmark).
        out2 = {"type": "stop", "confidence": 0.0, "debug": {"parse_note": parse_note, "raw_action": _to_jsonable(action_obj)}}
        co = action_obj.get("critical_object")
        if isinstance(co, list):
            out2["critical_object"] = [str(x) for x in co if str(x).strip()]
        elif isinstance(co, str) and co.strip():
            out2["critical_object"] = [co.strip()]
        risks = action_obj.get("risks")
        if isinstance(risks, list):
            out_dbg2 = out2.get("debug") if isinstance(out2.get("debug"), dict) else {}
            out_dbg2["risks"] = [str(x) for x in risks if str(x).strip()]
            out2["debug"] = out_dbg2
        return out2


class VQAHierarchicalTrajectorySelectionAdvisor:
    """Hierarchical trajectory selection (multi-call) for large candidate sets (e.g., 64).

    This is intentionally implemented as a new advisor class so existing single-call behavior remains unchanged.
    """

    PROMPT_VERSION = "vqa_traj_select_hier_v2"

    def __init__(
        self,
        *,
        model: ModelAdapter,
        prompt_cfg: VQATrajectoryPromptConfig,
        overlay_cfg: Any,
        out_dir: Path,
        branch: int = 4,
        max_leaf: int = 16,
        final_topk: int = 6,
        max_levels: int = 3,
        rep_prompt_retries: int = 1,
        use_chat_history: bool = False,
        hier_overlay_mode: str = "single",  # "single" | "grid"
        cluster_space: str = "traj",  # "traj" | "endpoint"
        traj_feature_points: int = 6,
        balance_clusters: bool = True,
        **_unused_kwargs: Any,
    ) -> None:
        self.model = model
        self.prompt_cfg = prompt_cfg
        self.overlay_cfg = overlay_cfg
        self.out_dir = Path(out_dir).resolve()
        self.branch = int(branch)
        self.max_leaf = int(max_leaf)
        self.final_topk = int(final_topk)
        self.max_levels = int(max_levels)
        self.rep_prompt_retries = int(rep_prompt_retries)
        self.use_chat_history = bool(use_chat_history)
        self.hier_overlay_mode = str(hier_overlay_mode)
        self.cluster_space = str(cluster_space)
        self.traj_feature_points = int(traj_feature_points)
        self.balance_clusters = bool(balance_clusters)

        # Debug info for traces/reports (keep same field names as other advisors).
        self.last_messages: list[dict[str, Any]] | None = None
        self.last_raw_response: str | None = None
        self.last_parse_note: str | None = None
        self.last_thoughts: list[str] | None = None
        self.last_provider_meta: dict[str, Any] | None = None

    @staticmethod
    def _traj_dist_m(points_xy: list[list[float]] | None) -> float | None:
        if not points_xy or len(points_xy) < 2:
            return None
        try:
            dist = 0.0
            for j in range(1, len(points_xy)):
                dx = float(points_xy[j][0]) - float(points_xy[j - 1][0])
                dy = float(points_xy[j][1]) - float(points_xy[j - 1][1])
                dist += math.sqrt(dx * dx + dy * dy)
            return float(dist)
        except Exception:
            return None

    @staticmethod
    def _kcenter_reps(endpoints: dict[int, tuple[float, float]], *, start_idx: int, k: int) -> list[int]:
        """Greedy farthest-first k-center on endpoint positions."""
        if k <= 0:
            return []
        idxs = [int(i) for i in endpoints.keys()]
        if not idxs:
            return []
        if int(start_idx) not in endpoints:
            start_idx = idxs[0]
        selected = [int(start_idx)]

        def d2(a: int, b: int) -> float:
            ax, ay = endpoints[int(a)]
            bx, by = endpoints[int(b)]
            dx = float(ax) - float(bx)
            dy = float(ay) - float(by)
            return dx * dx + dy * dy

        while len(selected) < int(min(k, len(idxs))):
            best_i = None
            best_min = -1.0
            for i in idxs:
                if int(i) in selected:
                    continue
                try:
                    m = min(d2(i, c) for c in selected)
                except Exception:
                    continue
                if math.isfinite(m) and m > best_min:
                    best_min = m
                    best_i = int(i)
            if best_i is None:
                break
            selected.append(int(best_i))
        return selected

    @staticmethod
    def _assign_to_reps(endpoints: dict[int, tuple[float, float]], reps: list[int]) -> dict[int, list[int]]:
        """Assign each candidate to nearest rep in endpoint space. Returns rep->members (includes rep)."""
        if not reps:
            return {}
        reps2 = [int(r) for r in reps]
        out: dict[int, list[int]] = {int(r): [] for r in reps2}
        for idx, (x, y) in endpoints.items():
            best_r = reps2[0]
            best_d = float("inf")
            for r in reps2:
                rx, ry = endpoints[int(r)]
                dx = float(x) - float(rx)
                dy = float(y) - float(ry)
                d = dx * dx + dy * dy
                if d < best_d:
                    best_d = d
                    best_r = int(r)
            out[int(best_r)].append(int(idx))
        for r in list(out.keys()):
            out[r] = sorted(set(out[r]))
        return out

    @staticmethod
    def _assign_to_reps_balanced(endpoints: dict[int, tuple[float, float]], reps: list[int]) -> dict[int, list[int]]:
        """Balanced assignment with explicit per-cluster capacities.

        This enforces near-equal cluster sizes (difference <= 1) rather than only capping the
        maximum size. It also guarantees a rep is assigned to itself.
        """
        if not reps:
            return {}
        reps2 = [int(r) for r in sorted(set(int(x) for x in reps)) if int(r) in endpoints]
        if not reps2:
            return {}
        items = [int(i) for i in sorted(set(int(x) for x in endpoints.keys()))]
        if not items:
            return {int(r): [int(r)] for r in reps2}

        def d2(i: int, r: int) -> float:
            ix, iy = endpoints[int(i)]
            rx, ry = endpoints[int(r)]
            dx = float(ix) - float(rx)
            dy = float(iy) - float(ry)
            return dx * dx + dy * dy

        N = int(len(items))
        K = int(len(reps2))
        base = int(N // max(1, K))
        rem = int(N % max(1, K))
        base = max(1, base)

        # Decide which reps get the extra +1 capacity based on nearest assignment mass.
        init_counts = {int(r): 0 for r in reps2}
        for i in items:
            try:
                r0 = int(min(reps2, key=lambda r: d2(int(i), int(r))))
            except Exception:
                r0 = int(reps2[0])
            init_counts[int(r0)] = int(init_counts.get(int(r0), 0)) + 1
        reps_by_mass = sorted(reps2, key=lambda r: (-int(init_counts.get(int(r), 0)), int(r)))
        big_reps = set(int(r) for r in reps_by_mass[:rem])
        cap = {int(r): int(base + (1 if int(r) in big_reps else 0)) for r in reps2}

        # Pre-assign reps to themselves.
        assign: dict[int, int] = {}
        remaining = dict(cap)
        for r in reps2:
            assign[int(r)] = int(r)
            remaining[int(r)] = int(remaining.get(int(r), 0)) - 1
            remaining[int(r)] = max(0, int(remaining[int(r)]))

        # Assign non-reps using a regret-ordered greedy under capacity constraints.
        reps_set = set(int(r) for r in reps2)
        others = [int(i) for i in items if int(i) not in reps_set]
        ranked: list[tuple[float, float, int, list[int]]] = []
        for i in others:
            dlist = sorted(((float(d2(int(i), int(r))), int(r)) for r in reps2), key=lambda t: (t[0], t[1]))
            best_d = float(dlist[0][0]) if dlist else float("inf")
            second_d = float(dlist[1][0]) if len(dlist) > 1 else best_d
            regret = float(second_d - best_d)
            ranked.append((-regret, best_d, int(i), [int(r) for _d, r in dlist]))
        ranked.sort(key=lambda t: (t[0], t[1], t[2]))

        for _neg_regret, _best_d, i, order in ranked:
            placed = False
            for r in order:
                if int(remaining.get(int(r), 0)) > 0:
                    assign[int(i)] = int(r)
                    remaining[int(r)] = int(remaining[int(r)]) - 1
                    placed = True
                    break
            if not placed:
                # Should be rare (capacities should match exactly); fall back to nearest rep.
                assign[int(i)] = int(order[0]) if order else int(reps2[0])

        clusters: dict[int, list[int]] = {int(r): [] for r in reps2}
        for i, r in assign.items():
            clusters.setdefault(int(r), []).append(int(i))
        for r in reps2:
            clusters.setdefault(int(r), [])
            if int(r) not in clusters[int(r)]:
                clusters[int(r)].append(int(r))
            clusters[int(r)] = sorted(set(int(x) for x in clusters[int(r)]))
        return clusters

    @staticmethod
    def _traj_feature_vector(points_xy: Any, *, k: int) -> list[float] | None:
        """Subsample a trajectory polyline into a fixed-length feature vector.

        Uses `k` points evenly spaced along the polyline; feature is [x0,y0,x1,y1,...].
        """
        if not isinstance(points_xy, list) or len(points_xy) < 2:
            return None
        try:
            pts: list[tuple[float, float]] = []
            for p in points_xy:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
            if len(pts) < 2:
                return None
            kk = max(2, int(k))
            idxs = [int(round(i * (len(pts) - 1) / float(kk - 1))) for i in range(kk)]
            out: list[float] = []
            for j in idxs:
                x, y = pts[int(min(max(j, 0), len(pts) - 1))]
                out.extend([float(x), float(y)])
            return out
        except Exception:
            return None

    @staticmethod
    def _kcenter_reps_vec(vecs: dict[int, list[float]], *, start_idx: int, k: int) -> list[int]:
        """Greedy farthest-first k-center on vector embeddings (squared L2)."""
        if k <= 0:
            return []
        idxs = [int(i) for i in vecs.keys()]
        if not idxs:
            return []
        if int(start_idx) not in vecs:
            start_idx = idxs[0]
        selected = [int(start_idx)]

        def d2(a: int, b: int) -> float:
            va = vecs[int(a)]
            vb = vecs[int(b)]
            n = min(len(va), len(vb))
            s = 0.0
            for ii in range(n):
                dx = float(va[ii]) - float(vb[ii])
                s += dx * dx
            return s

        while len(selected) < int(min(k, len(idxs))):
            best_i = None
            best_min = -1.0
            for i in idxs:
                if int(i) in selected:
                    continue
                try:
                    m = min(d2(i, c) for c in selected)
                except Exception:
                    continue
                if math.isfinite(m) and m > best_min:
                    best_min = m
                    best_i = int(i)
            if best_i is None:
                break
            selected.append(int(best_i))
        return selected

    @staticmethod
    def _assign_to_reps_vec(vecs: dict[int, list[float]], reps: list[int]) -> dict[int, list[int]]:
        """Assign each candidate to nearest rep in embedding space. Returns rep->members (includes rep)."""
        if not reps:
            return {}
        reps2 = [int(r) for r in reps]
        out: dict[int, list[int]] = {int(r): [] for r in reps2}

        def d2(a: int, b: int) -> float:
            va = vecs[int(a)]
            vb = vecs[int(b)]
            n = min(len(va), len(vb))
            s = 0.0
            for ii in range(n):
                dx = float(va[ii]) - float(vb[ii])
                s += dx * dx
            return s

        for idx in vecs.keys():
            best_r = reps2[0]
            best_d = float("inf")
            for r in reps2:
                dd = d2(int(idx), int(r))
                if dd < best_d:
                    best_d = dd
                    best_r = int(r)
            out[int(best_r)].append(int(idx))
        for r in list(out.keys()):
            out[r] = sorted(set(out[r]))
        return out

    @staticmethod
    def _assign_to_reps_vec_balanced(vecs: dict[int, list[float]], reps: list[int]) -> dict[int, list[int]]:
        """Balanced assignment in embedding space with explicit per-cluster capacities."""
        if not reps:
            return {}
        reps2 = [int(r) for r in sorted(set(int(x) for x in reps)) if int(r) in vecs]
        if not reps2:
            return {}
        items = [int(i) for i in sorted(set(int(x) for x in vecs.keys()))]
        if not items:
            return {int(r): [int(r)] for r in reps2}

        def d2(a: int, b: int) -> float:
            va = vecs[int(a)]
            vb = vecs[int(b)]
            n = min(len(va), len(vb))
            s = 0.0
            for ii in range(n):
                dx = float(va[ii]) - float(vb[ii])
                s += dx * dx
            return float(s)

        N = int(len(items))
        K = int(len(reps2))
        base = int(N // max(1, K))
        rem = int(N % max(1, K))
        base = max(1, base)

        init_counts = {int(r): 0 for r in reps2}
        for i in items:
            try:
                r0 = int(min(reps2, key=lambda r: d2(int(i), int(r))))
            except Exception:
                r0 = int(reps2[0])
            init_counts[int(r0)] = int(init_counts.get(int(r0), 0)) + 1
        reps_by_mass = sorted(reps2, key=lambda r: (-int(init_counts.get(int(r), 0)), int(r)))
        big_reps = set(int(r) for r in reps_by_mass[:rem])
        cap = {int(r): int(base + (1 if int(r) in big_reps else 0)) for r in reps2}

        assign: dict[int, int] = {}
        remaining = dict(cap)
        for r in reps2:
            assign[int(r)] = int(r)
            remaining[int(r)] = int(remaining.get(int(r), 0)) - 1
            remaining[int(r)] = max(0, int(remaining[int(r)]))

        reps_set = set(int(r) for r in reps2)
        others = [int(i) for i in items if int(i) not in reps_set]
        ranked: list[tuple[float, float, int, list[int]]] = []
        for i in others:
            dlist = sorted(((float(d2(int(i), int(r))), int(r)) for r in reps2), key=lambda t: (t[0], t[1]))
            best_d = float(dlist[0][0]) if dlist else float("inf")
            second_d = float(dlist[1][0]) if len(dlist) > 1 else best_d
            regret = float(second_d - best_d)
            ranked.append((-regret, best_d, int(i), [int(r) for _d, r in dlist]))
        ranked.sort(key=lambda t: (t[0], t[1], t[2]))

        for _neg_regret, _best_d, i, order in ranked:
            placed = False
            for r in order:
                if int(remaining.get(int(r), 0)) > 0:
                    assign[int(i)] = int(r)
                    remaining[int(r)] = int(remaining[int(r)]) - 1
                    placed = True
                    break
            if not placed:
                assign[int(i)] = int(order[0]) if order else int(reps2[0])

        clusters: dict[int, list[int]] = {int(r): [] for r in reps2}
        for i, r in assign.items():
            clusters.setdefault(int(r), []).append(int(i))
        for r in reps2:
            clusters.setdefault(int(r), [])
            if int(r) not in clusters[int(r)]:
                clusters[int(r)].append(int(r))
            clusters[int(r)] = sorted(set(int(x) for x in clusters[int(r)]))
        return clusters

    def _call_model(self, *, messages: list[dict[str, Any]], obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Call backend model and return (raw_text, per_call_meta)."""
        t0 = time.perf_counter()
        raw = self.model.call(messages=messages, obs=obs)
        t1 = time.perf_counter()
        latency_ms = float(max(0.0, (t1 - t0) * 1000.0))
        usage = getattr(self.model, "last_usage", None)
        meta: dict[str, Any] = {"latency_ms": latency_ms}
        if isinstance(usage, dict):
            meta["usage"] = dict(usage)
        return ("" if raw is None else str(raw)), meta

    def advise(self, query_bundle: QueryBundle) -> VlmAdvice:
        # Reset debug state
        self.last_messages = None
        self.last_raw_response = None
        self.last_parse_note = None
        self.last_thoughts = None
        self.last_provider_meta = None

        qb = dict(query_bundle)
        ep_id = str(qb.get("episode_id") or "")
        t = float(qb.get("t") or 0.0)
        snap_idx = int(qb.get("snapshot_index") or 0)

        base_rgb_ref = qb.get("__rgb_frame_ref")
        cand_pts = qb.get("__candidates_points_xy")
        if not isinstance(base_rgb_ref, str) or not base_rgb_ref.strip():
            return {"type": "stop", "confidence": 0.0, "debug": {"parse_note": "missing_base_rgb_frame_ref"}}
        if not isinstance(cand_pts, list) or not cand_pts:
            return {"type": "stop", "confidence": 0.0, "debug": {"parse_note": "missing_candidates_points_xy"}}

        # Extract goal fields (optional).
        goal_xy = qb.get("goal_xy") if isinstance(qb.get("goal_xy"), list) else None
        goal_distance_m = qb.get("goal_distance_m") if qb.get("goal_distance_m") is not None else None
        goal_text = qb.get("goal_text") if isinstance(qb.get("goal_text"), str) else None

        # Build endpoints + best-effort scores from obs.
        scores_by_idx: dict[int, float] = {}
        endpoints: dict[int, tuple[float, float]] = {}
        try:
            obs = qb.get("obs")
            if isinstance(obs, dict):
                planner = obs.get("planner")
                if isinstance(planner, dict) and isinstance(planner.get("candidates"), list):
                    for c in planner.get("candidates") or []:
                        if not isinstance(c, dict):
                            continue
                        try:
                            ii = int(c.get("index"))
                        except Exception:
                            continue
                        try:
                            if c.get("score") is not None:
                                scores_by_idx[ii] = float(c.get("score"))
                        except Exception:
                            pass
                        end = c.get("endpoint_xy")
                        if isinstance(end, (list, tuple)) and len(end) >= 2:
                            try:
                                endpoints[ii] = (float(end[0]), float(end[1]))
                            except Exception:
                                pass
        except Exception:
            pass

        # Prefer endpoints from points (always consistent).
        for ii, pts in enumerate(cand_pts):
            if not isinstance(pts, list) or not pts:
                continue
            try:
                last = pts[-1]
                if isinstance(last, (list, tuple)) and len(last) >= 2:
                    endpoints[int(ii)] = (float(last[0]), float(last[1]))
            except Exception:
                continue
            scores_by_idx.setdefault(int(ii), 0.0)

        all_indices = sorted(endpoints.keys())
        if not all_indices:
            return {"type": "stop", "confidence": 0.0, "debug": {"parse_note": "no_valid_candidates"}}

        # Minimal record-like object for overlay rendering.
        from types import SimpleNamespace

        max_idx = max(all_indices)
        candidates_obj = []
        for ii in range(int(max_idx) + 1):
            pts = cand_pts[ii] if ii < len(cand_pts) else []
            sc = float(scores_by_idx.get(int(ii), 0.0))
            candidates_obj.append(SimpleNamespace(points_xy=pts, score=sc))
        record = SimpleNamespace(candidates=candidates_obj)

        # Resolve base image path.
        base_path = Path(str(base_rgb_ref))
        if not base_path.is_absolute():
            base_path = (self.out_dir / base_path).resolve()
        if not base_path.exists():
            return {"type": "stop", "confidence": 0.0, "debug": {"parse_note": "base_rgb_file_missing"}}

        # Accumulate provider stats across calls, and keep full stage prompt history across calls
        # so traces/reports can show multi-round prompting.
        totals = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0, "latency_ms": 0.0, "calls": 0}
        calls_detail: list[dict[str, Any]] = []
        all_messages: list[dict[str, Any]] = []
        raw_by_call: list[str] = []
        parse_note_by_call: list[str | None] = []
        hierarchical_calls: list[dict[str, Any]] = []

        base_system_msg: dict[str, Any] | None = None
        session_messages: list[dict[str, Any]] | None = [] if self.use_chat_history else None

        def _accum(call_meta: dict[str, Any]) -> None:
            totals["calls"] += 1
            totals["latency_ms"] += float(call_meta.get("latency_ms") or 0.0)
            u = call_meta.get("usage")
            if isinstance(u, dict):
                try:
                    totals["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                    totals["output_tokens"] += int(u.get("output_tokens") or 0)
                    totals["total_tokens"] += int(u.get("total_tokens") or 0)
                except Exception:
                    pass
            calls_detail.append(call_meta)

        # Render an overlay for a given subset.
        def _render_overlay(
            selected: list[int],
            *,
            tag: str,
            cluster_groups: dict[int, list[int]] | None = None,
            label_indices_subset: list[int] | None = None,
        ) -> str:
            t_str = f"{float(t):.3f}".rstrip("0").rstrip(".")
            outp = (
                self.out_dir
                / "artifacts"
                / "hier_overlays"
                / str(ep_id)
                / f"{t_str}_snap{int(snap_idx)}_{tag}.png"
            ).resolve()
            if outp.exists():
                return str(outp.relative_to(self.out_dir))
            outp.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            from slow_brain_fast_planner.benchmarks.overlays import render_overlay_pil

            with Image.open(base_path) as im:
                base = im.convert("RGB")
                # IMPORTANT:
                # - Leaf stages should render normal candidate lines (cluster_member_style=None).
                # - Rep stages can use special styles (e.g., endpoints-only or per-cluster grid) for readability.
                member_style: str | None = None
                if isinstance(cluster_groups, dict) and cluster_groups:
                    member_style = "endpoints" if self.hier_overlay_mode != "grid" else "faint_lines"

                if str(self.hier_overlay_mode) == "grid" and isinstance(cluster_groups, dict) and cluster_groups:
                    # Grid overlay: each panel shows ONE cluster (members faint + rep bold).
                    reps_order = sorted(int(r) for r in cluster_groups.keys())
                    n = int(len(reps_order))
                    cols = 1 if n <= 1 else 2
                    rows = int(math.ceil(float(n) / float(cols)))
                    gap = 10

                    # Keep total size roughly the same as a single overlay image so tokens don't blow up.
                    out_w = int(base.size[0]) + int(getattr(self.overlay_cfg, "pad_left", 0)) + int(getattr(self.overlay_cfg, "pad_right", 0))
                    out_h = int(base.size[1]) + int(getattr(self.overlay_cfg, "pad_top", 0)) + int(getattr(self.overlay_cfg, "pad_bottom", 0))
                    tile_w = int(max(1, (out_w - gap * (cols - 1)) // cols))
                    tile_h = int(max(1, (out_h - gap * (rows - 1)) // rows))
                    scale = float(min(float(tile_w) / float(max(1, out_w)), float(tile_h) / float(max(1, out_h))))
                    scale = float(min(1.0, max(0.05, scale)))

                    canvas = Image.new("RGB", (int(out_w), int(out_h)), (0, 0, 0))

                    # Font for per-panel header.
                    try:
                        from slow_brain_fast_planner.benchmarks.overlays import _load_font  # type: ignore

                        header_font = _load_font(max(12, int(round(tile_h * 0.06))))
                    except Exception:
                        header_font = None

                    for k_i, rep in enumerate(reps_order):
                        members = [int(x) for x in (cluster_groups.get(int(rep)) or [])]
                        if int(rep) not in members:
                            members.append(int(rep))
                        members = sorted(set(members))
                        cg_one = {int(rep): members}
                        # In grid mode, point-dots tend to dominate after downsampling. Prefer clean polylines.
                        cfg_grid = self.overlay_cfg
                        try:
                            lw = int(getattr(self.overlay_cfg, "line_width", 2))
                            hlw = int(getattr(self.overlay_cfg, "highlight_line_width", lw))
                            alpha0 = float(getattr(self.overlay_cfg, "alpha", 0.75))
                            factor = float(1.0 / scale)
                            # Compensate for later downsampling so strokes remain visible.
                            # Caps are deliberately higher in grid mode since each panel is downsampled.
                            lw2 = int(min(14, max(lw, int(round(float(lw) * factor)))))
                            hlw2 = int(min(16, max(hlw, int(round(float(hlw) * factor)))))
                            # Make grid overlays more VLM-readable: higher opacity helps after downsampling.
                            alpha2 = float(min(1.0, max(alpha0, 0.95)))
                            cfg_grid = replace(
                                self.overlay_cfg,
                                draw_points=False,
                                point_radius=0,
                                line_width=lw2,
                                highlight_line_width=hlw2,
                                alpha=alpha2,
                                label_indices=False,  # use the per-panel header instead of many endpoint labels
                            )
                        except Exception:
                            cfg_grid = self.overlay_cfg
                        tile = render_overlay_pil(
                            base_image=base,
                            record=record,
                            cfg=cfg_grid,
                            selected_indices=members,
                            cluster_groups=cg_one,
                            label_indices_subset=[int(rep)],
                            cluster_member_style="faint_lines",
                            pred_index=None,
                            label_index=None,
                            goal_text=None,
                            goal_xy=(
                                (float(goal_xy[0]), float(goal_xy[1]))
                                if isinstance(goal_xy, list) and len(goal_xy) >= 2
                                else None
                            ),
                        ).convert("RGB")
                        tile = tile.resize((int(tile_w), int(tile_h)), resample=Image.BILINEAR)

                        rr = int(k_i // cols)
                        cc = int(k_i % cols)
                        x0 = int(cc * (tile_w + gap))
                        y0 = int(rr * (tile_h + gap))
                        canvas.paste(tile, (int(x0), int(y0)))

                        # Panel header: "rep=<idx>  n=<size>"
                        try:
                            from PIL import ImageDraw

                            draw = ImageDraw.Draw(canvas, "RGBA")
                            txt = f"rep {int(rep)}   n={len(members)}"
                            bx0, by0, bx1, by1 = draw.textbbox((0, 0), txt, font=header_font) if header_font else (0, 0, 0, 0)
                            tw = int(bx1 - bx0) if header_font else int(8 * len(txt))
                            th = int(by1 - by0) if header_font else 16
                            pad = 6
                            hx0 = int(x0 + 8)
                            hy0 = int(y0 + 8)
                            draw.rectangle(
                                (hx0, hy0, hx0 + tw + pad * 2, hy0 + th + pad * 2),
                                fill=(0, 0, 0, 160),
                                outline=(255, 255, 255, 180),
                                width=2,
                            )
                            draw.text((hx0 + pad, hy0 + pad), txt, fill=(255, 255, 255, 255), font=header_font)
                        except Exception:
                            pass

                    canvas.save(outp)
                else:
                    overlay = render_overlay_pil(
                        base_image=base,
                        record=record,
                        cfg=self.overlay_cfg,
                        selected_indices=[int(x) for x in selected],
                        cluster_groups=cluster_groups,
                        label_indices_subset=label_indices_subset,
                        cluster_member_style=member_style,
                        pred_index=None,
                        label_index=None,
                        goal_text=None,
                        goal_xy=(
                            (float(goal_xy[0]), float(goal_xy[1]))
                            if isinstance(goal_xy, list) and len(goal_xy) >= 2
                            else None
                        ),
                    )
                    overlay.save(outp)
            return str(outp.relative_to(self.out_dir))

        def _candidate_rows(subset: list[int]) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for ii in subset:
                pts = cand_pts[int(ii)] if int(ii) < len(cand_pts) else None
                end = endpoints.get(int(ii))
                td = self._traj_dist_m(pts if isinstance(pts, list) else None)
                rr: dict[str, Any] = {
                    "index": int(ii),
                    "score": float(scores_by_idx.get(int(ii), 0.0)),
                    "traj_dist_m": float(td) if td is not None else None,
                    "end_xy": [float(end[0]), float(end[1])] if end is not None else None,
                }
                rows.append(rr)
            return rows

        def _select_from(
            allowed: list[int],
            *,
            stage_tag: str,
            note: str,
            overlay_indices: list[int] | None = None,
            cluster_groups: dict[int, list[int]] | None = None,
            label_indices_subset: list[int] | None = None,
        ) -> tuple[int | None, str | None]:
            overlay_sel = overlay_indices if isinstance(overlay_indices, list) and overlay_indices else allowed
            overlay_ref = _render_overlay(
                overlay_sel,
                tag=stage_tag,
                cluster_groups=cluster_groups,
                label_indices_subset=label_indices_subset,
            )
            K_global = int(len(candidates_obj))
            messages_step = build_vqa_trajectory_selection_messages(
                cfg=self.prompt_cfg,
                overlay_frame_ref=str(overlay_ref),
                history_frame_refs=None,
                num_candidates=K_global,
                goal_text=goal_text,
                goal_xy=goal_xy,
                goal_distance_m=goal_distance_m,
                candidate_scores=_candidate_rows(allowed),
                image_info={"goal_direction_arrow": bool(getattr(self.overlay_cfg, "draw_goal_direction_arrow", False))},
            )
            # Prepend instruction into user text.
            try:
                if isinstance(messages_step[1].get("content"), list) and messages_step[1]["content"]:
                    p0 = messages_step[1]["content"][0]
                    if isinstance(p0, dict) and p0.get("type") == "text":
                        p0["text"] = (note.strip() + "\n\n" + str(p0.get("text", ""))).strip()
            except Exception:
                pass

            obs0 = qb.get("obs") if isinstance(qb.get("obs"), dict) else {}
            if session_messages is None:
                # No-chat-history mode: each stage is an independent call, but for trace/report clarity
                # we still want to preserve the full multi-stage prompt history.
                #
                # IMPORTANT: append the prompt BEFORE calling the backend so that if the backend
                # raises (e.g., 429 quota / network errors), the trace still contains the stage prompt
                # that triggered the failure (and any prior stage prompts).
                all_messages.extend(messages_step)
                self.last_messages = list(all_messages)
                messages = messages_step
            else:
                # Initialize base system once.
                nonlocal base_system_msg
                if base_system_msg is None:
                    base_system_msg = messages_step[0] if isinstance(messages_step[0], dict) else {"role": "system", "content": ""}
                    session_messages.append(base_system_msg)
                session_messages.append(messages_step[1])
                messages = list(session_messages)

            raw, call_meta = self._call_model(messages=messages, obs=obs0)
            _accum(call_meta)
            self.last_messages = messages
            self.last_raw_response = raw if raw.strip() else None
            raw_by_call.append(raw if isinstance(raw, str) else "")
            # Best-effort thoughts
            try:
                tt = getattr(self.model, "last_thoughts", None)
                if isinstance(tt, list):
                    self.last_thoughts = [str(x) for x in tt]
            except Exception:
                self.last_thoughts = None

            action_obj, parse_note = parse_vqa_trajectory_action(raw, num_candidates=K_global)
            parse_note_by_call.append(parse_note)
            call_rec: dict[str, Any] = {
                "stage": str(stage_tag),
                "overlay_ref": str(overlay_ref),
                "allowed_count": int(len(allowed)),
                "allowed_indices_preview": [int(x) for x in allowed[: min(32, len(allowed))]],
                "note": str(note),
                "raw": raw,
                "parse_note": parse_note,
            }
            if action_obj is None:
                call_rec["action"] = None
                call_rec["selected_index"] = None
                hierarchical_calls.append(call_rec)
                if session_messages is not None:
                    session_messages.append({"role": "assistant", "content": raw})
                return None, parse_note or "parse_failed"
            if action_obj.get("action") != "select_trajectory":
                call_rec["action"] = str(action_obj.get("action"))
                call_rec["selected_index"] = None
                hierarchical_calls.append(call_rec)
                if session_messages is not None:
                    session_messages.append({"role": "assistant", "content": raw})
                return None, parse_note or "not_select"
            try:
                idx = int(action_obj.get("selected_index"))
            except Exception:
                call_rec["action"] = "select_trajectory"
                call_rec["selected_index"] = None
                hierarchical_calls.append(call_rec)
                if session_messages is not None:
                    session_messages.append({"role": "assistant", "content": raw})
                return None, "selected_index_not_int"
            if int(idx) not in set(int(x) for x in allowed):
                call_rec["action"] = "select_trajectory"
                call_rec["selected_index"] = int(idx)
                hierarchical_calls.append(call_rec)
                if session_messages is not None:
                    session_messages.append({"role": "assistant", "content": raw})
                return None, "selected_index_not_in_allowed_set"
            call_rec["action"] = "select_trajectory"
            call_rec["selected_index"] = int(idx)
            hierarchical_calls.append(call_rec)
            if session_messages is not None:
                session_messages.append({"role": "assistant", "content": raw})
            return int(idx), None

        current = sorted(all_indices)
        chosen: int | None = None
        last_err: str | None = None

        for level in range(max(1, int(self.max_levels))):
            if len(current) <= max(1, int(self.max_leaf)) or level == int(self.max_levels) - 1:
                # Hard cap the final pick set to keep the last overlay/prompt readable.
                leaf_allowed = list(current)
                ft = max(1, int(self.final_topk))
                if len(leaf_allowed) > ft:
                    try:
                        start = int(max(leaf_allowed, key=lambda ii: float(scores_by_idx.get(int(ii), 0.0))))
                    except Exception:
                        start = int(leaf_allowed[0])
                    ep_map2 = {int(ii): endpoints[int(ii)] for ii in leaf_allowed if int(ii) in endpoints}
                    down = self._kcenter_reps(ep_map2, start_idx=int(start), k=int(ft)) if ep_map2 else []
                    if down:
                        leaf_allowed = sorted(set(int(x) for x in down))
                    else:
                        leaf_allowed = sorted(set(int(x) for x in leaf_allowed[: int(ft)]))

                note = (
                    "Final step: pick the best trajectory index among the candidates shown."
                    + (f" (Downselected to {len(leaf_allowed)}/{len(current)} for diversity.)" if len(leaf_allowed) < len(current) else "")
                )
                chosen, last_err = _select_from(
                    leaf_allowed,
                    stage_tag=f"L{level}_leaf{len(current)}_k{len(leaf_allowed)}",
                    note=note,
                )
                break

            start = max(current, key=lambda ii: float(scores_by_idx.get(int(ii), 0.0)))
            if str(self.cluster_space) == "endpoint":
                reps = self._kcenter_reps({ii: endpoints[ii] for ii in current if ii in endpoints}, start_idx=int(start), k=max(2, int(self.branch)))
                ep_map = {ii: endpoints[ii] for ii in current if ii in endpoints}
                clusters = self._assign_to_reps_balanced(ep_map, reps) if bool(self.balance_clusters) else self._assign_to_reps(ep_map, reps)
            else:
                vecs: dict[int, list[float]] = {}
                for ii in current:
                    if int(ii) >= len(cand_pts):
                        continue
                    fv = self._traj_feature_vector(cand_pts[int(ii)], k=int(self.traj_feature_points))
                    if fv is None:
                        # Fallback: repeat endpoint to match dimensionality (keeps clustering defined).
                        end = endpoints.get(int(ii))
                        if end is not None:
                            ex, ey = float(end[0]), float(end[1])
                            fv = [ex, ey] * max(2, int(self.traj_feature_points))
                    if fv is not None:
                        vecs[int(ii)] = fv
                if vecs:
                    reps = self._kcenter_reps_vec(vecs, start_idx=int(start), k=max(2, int(self.branch)))
                    clusters = (
                        self._assign_to_reps_vec_balanced(vecs, reps)
                        if bool(self.balance_clusters)
                        else self._assign_to_reps_vec(vecs, reps)
                    )
                else:
                    reps = self._kcenter_reps({ii: endpoints[ii] for ii in current if ii in endpoints}, start_idx=int(start), k=max(2, int(self.branch)))
                    ep_map = {ii: endpoints[ii] for ii in current if ii in endpoints}
                    clusters = self._assign_to_reps_balanced(ep_map, reps) if bool(self.balance_clusters) else self._assign_to_reps(ep_map, reps)
            reps2 = sorted(set(int(x) for x in reps))
            note = (
                f"Stage {level+1}: choose ONE cluster representative index (coarse choice). "
                f"Each representative stands for a cluster of trajectories; after you pick one, "
                f"the next step will show ONLY trajectories from that cluster. "
                f"Allowed indices: {', '.join(str(x) for x in reps2)}"
            )
            cluster_sizes = {int(r): int(len(clusters.get(int(r)) or [])) for r in reps2}
            note = note + "\n" + "Cluster sizes: " + ", ".join([f"{int(r)}→{int(cluster_sizes[int(r)])}" for r in reps2])
            if str(self.hier_overlay_mode) == "grid":
                note = note + "\n" + "Overlay is a grid (left-to-right, top-to-bottom) ordered by reps: " + ", ".join(
                    [str(int(r)) for r in reps2]
                )
            cg = {int(r): [int(x) for x in (clusters.get(int(r)) or [])] for r in reps2}
            # For the rep-selection overlay, visualize the FULL clusters (same color per cluster),
            # but only label the representative indices to keep the image readable.
            overlay_indices = sorted({int(x) for _r, xs in cg.items() for x in xs})
            idx = None
            err = None
            for _ in range(max(1, int(self.rep_prompt_retries))):
                idx, err = _select_from(
                    reps2,
                    stage_tag=f"L{level}_reps{len(reps2)}",
                    note=note,
                    overlay_indices=overlay_indices,
                    cluster_groups=cg,
                    label_indices_subset=reps2,
                )
                if idx is not None:
                    break
            if idx is None:
                chosen, last_err = None, err
                break
            current = sorted(set(int(x) for x in (clusters.get(int(idx)) or [int(idx)])))

        if chosen is None:
            # Fallback: best score in current pool.
            try:
                pool = current if current else all_indices
                chosen = int(max(pool, key=lambda ii: float(scores_by_idx.get(int(ii), 0.0))))
            except Exception:
                chosen = int(all_indices[0])

        self.last_provider_meta = {
            "usage": {
                "prompt_tokens": int(totals["prompt_tokens"]),
                "output_tokens": int(totals["output_tokens"]),
                "total_tokens": int(totals["total_tokens"]),
                "calls": int(max(1, totals["calls"])),
            },
            "latency_ms": float(totals["latency_ms"]),
            "calls_detail": calls_detail[:8],
            "hierarchical": {
                "raw_by_call": raw_by_call,
                "parse_note_by_call": parse_note_by_call,
                "calls": hierarchical_calls,
                "use_chat_history": bool(self.use_chat_history),
                "cluster_space": str(self.cluster_space),
            },
        }
        if session_messages is not None and session_messages:
            # Expose the multi-turn session transcript (no duplication).
            self.last_messages = session_messages
        elif all_messages:
            # Expose the full multi-call prompt history to traces/reports.
            self.last_messages = all_messages
        self.last_parse_note = last_err
        if last_err:
            return {"type": "select_trajectory", "selected_index": int(chosen), "confidence": 0.0, "debug": {"parse_note": last_err}}
        return {"type": "select_trajectory", "selected_index": int(chosen), "confidence": 0.0}

