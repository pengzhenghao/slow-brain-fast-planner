from __future__ import annotations

import base64
import importlib
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class ModelAdapter(Protocol):
    """Minimal model adapter interface used by benchmark scripts.

    Note: we keep this extremely small on purpose so we can plug in different backends later.
    """

    def call(self, *, messages: list[dict[str, Any]], obs: dict[str, Any]) -> str: ...


class DummyTrajectoryModelAdapter:
    """Offline-safe, deterministic adapter for tests/demos (no network).

    This adapter does NOT try to "understand" the prompt; it simply uses
    `obs["planner"]["candidates"]`.
    """

    def __init__(self, *, mode: str, seed: int, topk: int = 1) -> None:
        self.mode = str(mode)
        self.rng = np.random.default_rng(int(seed))
        self.topk = int(topk)

    def call(self, *, messages: list[dict[str, Any]], obs: dict[str, Any]) -> str:
        _ = messages  # present for interface parity with real adapters
        planner = obs.get("planner") or {}
        cand = planner.get("candidates") or []
        cand_conf = planner.get("candidate_confidence") or []
        scores = []
        for c in cand:
            try:
                scores.append(float(c.get("score")))
            except Exception:
                scores.append(float("nan"))

        K = len(scores)
        if K <= 0:
            return json.dumps({"action": "ask_for_help"})

        if self.mode == "argmax_score":
            # If scores are missing/NaN, this will default to index 0.
            arr = np.asarray(scores, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                idx = 0
            else:
                idx = int(np.argmax(arr))
            return json.dumps({"action": "select_trajectory", "selected_index": idx})

        if self.mode == "random_topk":
            # Select randomly from the top K candidates by score.
            arr = np.asarray(scores, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                # Fallback to index 0 if scores are missing.
                idx = 0
            else:
                # Get indices of top K scores.
                k = min(self.topk, len(arr))
                if k <= 0:
                    idx = 0
                else:
                    top_k_idxs = np.argpartition(arr, -k)[-k:]
                    idx = int(self.rng.choice(top_k_idxs))
            return json.dumps({"action": "select_trajectory", "selected_index": idx})

        if self.mode == "always_0":
            return json.dumps({"action": "select_trajectory", "selected_index": 0})

        if self.mode == "random":
            # Sample from the NMS-normalized probability over the *kept* candidates (as shown in
            # overlay).
            # If NMS probs are unavailable, sample uniformly across *kept* candidates.
            idxs: list[int] = []
            if isinstance(cand_conf, list):
                for r in cand_conf:
                    if isinstance(r, dict) and "index" in r:
                        try:
                            idx = int(r.get("index"))
                            if 0 <= idx < K:
                                idxs.append(idx)
                        except Exception:
                            continue
            if idxs:
                chosen = int(self.rng.choice(np.asarray(idxs, dtype=np.int64)))
                return json.dumps({"action": "select_trajectory", "selected_index": chosen})
            # Fallback: uniform across all.
            return json.dumps(
                {"action": "select_trajectory", "selected_index": int(self.rng.integers(0, K))}
            )

        if self.mode == "random_all":
            return json.dumps(
                {"action": "select_trajectory", "selected_index": int(self.rng.integers(0, K))}
            )

        if self.mode == "sample_nms_prob":
            # Sample from the NMS-normalized probability over the *kept* candidates (as shown in
            # overlay).
            # `candidate_confidence` rows store ORIGINAL planner indices (not rank-based).
            idxs: list[int] = []
            probs: list[float] = []
            if isinstance(cand_conf, list):
                for r in cand_conf:
                    if not isinstance(r, dict):
                        continue
                    try:
                        idx = int(r.get("index"))
                        p = float(r.get("nms_prob"))
                    except Exception:
                        continue
                    if not (0 <= idx < K):
                        continue
                    if not np.isfinite(p) or p < 0:
                        continue
                    idxs.append(idx)
                    probs.append(float(p))

            if idxs and probs:
                s = float(np.sum(np.asarray(probs, dtype=np.float64)))
                if np.isfinite(s) and s > 0:
                    pnorm = (np.asarray(probs, dtype=np.float64) / s).astype(np.float64)
                    chosen = int(self.rng.choice(np.asarray(idxs, dtype=np.int64), p=pnorm))
                    return json.dumps({"action": "select_trajectory", "selected_index": chosen})

            # Fallback: if NMS probs are unavailable, sample uniformly across all candidates.
            return json.dumps(
                {"action": "select_trajectory", "selected_index": int(self.rng.integers(0, K))}
            )

        return json.dumps({"action": "ask_for_help", "error": f"unknown_mode:{self.mode}"})


@dataclass(frozen=True)
class OpenAIChatCompletionsConfig:
    """Config for an OpenAI-style /v1/chat/completions compatible endpoint."""

    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    timeout_s: float = 60.0
    temperature: float = 0.0
    max_tokens: int | None = None
    # Set true if your backend supports response_format={"type":"json_object"}.
    response_format_json: bool = False
    # Used to resolve {"type":"image_ref","image_ref": "..."} parts.
    image_base_dir: Path | None = None
    # Best-effort retries for transient backend errors (429/5xx/timeouts).
    max_retries: int = 10
    retry_initial_backoff_s: float = 0.5
    retry_max_backoff_s: float = 300.0


def _is_url(s: str) -> bool:
    s2 = s.strip().lower()
    return s2.startswith("http://") or s2.startswith("https://") or s2.startswith("data:image/")


def _is_retryable_http_status(code: int | None) -> bool:
    if code is None:
        return False
    return int(code) in (408, 409, 425, 429, 500, 502, 503, 504)


def _is_retryable_exception(e: BaseException) -> bool:
    # urllib errors (OpenAI-compatible adapters).
    if isinstance(e, urllib.error.HTTPError):
        return _is_retryable_http_status(getattr(e, "code", None))
    if isinstance(e, urllib.error.URLError):
        return True

    # SDK errors (Gemini) and generic transient signals.
    msg = str(e).lower()
    transient_markers = (
        "empty_response",
        "overload",
        "overloaded",
        "unavailable",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "connection refused",
        "broken pipe",
        "rate limit",
        "too many requests",
        "429",
        "500",
        "502",
        "503",
        "504",
    )
    return any(m in msg for m in transient_markers)


def _sleep_backoff(attempt_idx: int, *, initial_s: float, max_s: float) -> None:
    """Exponential backoff with full jitter.

    attempt_idx: 0 for first retry, 1 for second retry, etc.
    """
    base = float(initial_s) * (2.0 ** float(attempt_idx))
    cap = min(base, float(max_s))
    dt = random.random() * cap
    time.sleep(dt)


def _env_flag(name: str, *, default: bool = False) -> bool:
    """Parse a boolean-ish environment variable."""
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off", ""):
        return False
    return bool(default)


def _env_float(name: str, *, default: float | None = None) -> float | None:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _extract_first_complete_json_object_prefix(s: str) -> str | None:
    """Return the prefix containing the first complete JSON object, if present.

    Intended for streaming: once we have a complete `{...}` object, we can stop reading more.
    This is a lightweight brace-matcher that is string/escape-aware (best-effort).
    """
    if not s:
        return None

    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return s[: i + 1]
            continue

    return None


def _shard_tag() -> str:
    try:
        from slow_brain_fast_planner.benchmarks.sharding import shard_tag_from_env

        return str(shard_tag_from_env())
    except Exception:
        return "[shard ?/?]"


def _is_empty_model_text(x: Any) -> bool:
    if x is None:
        return True
    try:
        s = str(x).strip()
    except Exception:
        return True
    # Some SDKs yield the literal string "None" when no text content exists.
    return s == "" or s.lower() == "none"


def _guess_mime_from_path(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "image/png"


def _image_ref_to_data_uri(image_ref: str, *, base_dir: Path | None) -> str:
    if _is_url(image_ref):
        return image_ref

    p = Path(image_ref)
    if not p.is_absolute():
        if base_dir is None:
            base_dir = Path.cwd()
        p = (base_dir / p).resolve()

    b = p.read_bytes()
    mime = _guess_mime_from_path(p)
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _to_openai_messages(
    messages: list[dict[str, Any]],
    *,
    image_base_dir: Path | None,
) -> list[dict[str, Any]]:
    """Convert our internal message format into OpenAI chat-completions message format.

    Supported user content parts:
    - {"type":"text","text": "..."}
    - {"type":"image_ref","image_ref": "relative/or/absolute/path.png"}
      (converted to image_url data URI)
    """

    out: list[dict[str, Any]] = []
    for m in messages:
        role = str(m.get("role"))
        content = m.get("content")

        if isinstance(content, list):
            parts_out: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    parts_out.append({"type": "text", "text": str(part.get("text", ""))})
                elif ptype == "image_ref":
                    image_ref = str(part.get("image_ref", ""))
                    url = _image_ref_to_data_uri(image_ref, base_dir=image_base_dir)
                    parts_out.append({"type": "image_url", "image_url": {"url": url}})
            out.append({"role": role, "content": parts_out})
        else:
            out.append({"role": role, "content": "" if content is None else str(content)})
    return out


class OpenAIChatCompletionsHttpAdapter:
    """HTTP adapter for OpenAI-style /v1/chat/completions endpoints.

    This is intentionally dependency-free (stdlib urllib) so the benchmark core stays lightweight.
    """

    def __init__(self, cfg: OpenAIChatCompletionsConfig) -> None:
        self.cfg = cfg
        # Best-effort token usage info from the last call (if provided by backend).
        self.last_usage: dict[str, Any] | None = None

    def call(self, *, messages: list[dict[str, Any]], obs: dict[str, Any]) -> str:
        _ = obs  # adapter only uses the prompt/messages
        self.last_usage = None

        base_url = str(self.cfg.base_url).rstrip("/")
        url = f"{base_url}/chat/completions"
        api_key = self.cfg.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OpenAI API key (set cfg.api_key or env OPENAI_API_KEY)")

        openai_messages = _to_openai_messages(messages, image_base_dir=self.cfg.image_base_dir)

        max_attempts = int(max(1, 1 + int(self.cfg.max_retries)))
        # Some OpenAI models reject certain parameters (e.g., temperature) unless omitted.
        # We'll adaptively retry once by omitting specific fields when the backend reports
        # a parameter-specific "unsupported_value" error.
        omit_temperature = False
        for attempt in range(max_attempts):
            payload: dict[str, Any] = {
                "model": str(self.cfg.model),
                "messages": openai_messages,
            }
            if not omit_temperature:
                payload["temperature"] = float(self.cfg.temperature)
            if self.cfg.max_tokens is not None:
                payload["max_tokens"] = int(self.cfg.max_tokens)
            if self.cfg.response_format_json:
                payload["response_format"] = {"type": "json_object"}

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )

            try:
                with urllib.request.urlopen(req, timeout=float(self.cfg.timeout_s)) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                obj = json.loads(raw)

                # Best-effort: usage accounting (OpenAI-compatible backends usually provide this).
                usage = obj.get("usage") if isinstance(obj, dict) else None
                if isinstance(usage, dict):
                    try:
                        pt = usage.get("prompt_tokens")
                        ct = usage.get("completion_tokens")
                        tt = usage.get("total_tokens")
                        out_usage: dict[str, Any] = {"raw": usage}
                        if pt is not None:
                            out_usage["prompt_tokens"] = int(pt)
                        if ct is not None:
                            out_usage["output_tokens"] = int(ct)
                        if tt is not None:
                            out_usage["total_tokens"] = int(tt)
                        self.last_usage = out_usage
                    except Exception:
                        self.last_usage = {"raw": usage}

                # Standard chat-completions: choices[0].message.content
                try:
                    content = obj["choices"][0]["message"]["content"]
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(f"Unexpected OpenAI response shape: {obj}") from e

                if _is_empty_model_text(content):
                    raise RuntimeError("empty_response")
                return str(content)

            except Exception as e:  # noqa: BLE001
                # Special-case: some HTTP 429s are NOT retryable (e.g., insufficient_quota /
                # billing).
                # We want to fail fast instead of sleeping/backing off for minutes.
                http_body: str | None = None
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        http_body = (
                            e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
                        )
                    except Exception:
                        http_body = ""

                    # Best-effort parse of OpenAI-style error payload:
                    # {"error": {"message": "...", "type": "...", "param": "...", "code": "..."}}
                    err_type = None
                    err_code = None
                    err_param = None
                    try:
                        obj2 = json.loads(http_body) if http_body else None
                        if (
                            isinstance(obj2, dict)
                            and isinstance(obj2.get("error"), dict)
                            and obj2["error"] is not None
                        ):
                            err_type = obj2["error"].get("type")
                            err_code = obj2["error"].get("code")
                            err_param = obj2["error"].get("param")
                    except Exception:
                        err_type = None
                        err_code = None
                        err_param = None

                    # Non-retryable conditions:
                    # - Auth errors (401/403)
                    # - Billing/quota exhausted (429 but with specific error markers)
                    try:
                        status = int(getattr(e, "code", 0) or 0)
                    except Exception:
                        status = 0
                    body_l = (http_body or "").lower()
                    if status in (401, 403):
                        raise RuntimeError(
                            f"OpenAI chat/completions HTTP {e.code}: {http_body}"
                        ) from e
                    if status == 429 and (
                        str(err_code).lower() == "insufficient_quota"
                        or str(err_type).lower() == "insufficient_quota"
                        or "check your plan and billing" in body_l
                        or "billing" in body_l
                    ):
                        raise RuntimeError(
                            f"OpenAI chat/completions HTTP {e.code}: {http_body}"
                        ) from e

                    # Parameter-specific adaptation:
                    # Some models only support the default temperature and reject explicit values.
                    if (
                        status == 400
                        and (str(err_code).lower() == "unsupported_value")
                        and (str(err_param).lower() == "temperature")
                        and (not omit_temperature)
                    ):
                        # Retry immediately with temperature omitted (use model default).
                        omit_temperature = True
                        continue

                is_last = attempt >= (max_attempts - 1)
                if (not is_last) and _is_retryable_exception(e):
                    tag = _shard_tag()
                    try:
                        # Keep logs short but informative (goes into launcher_logs/rank*.log).
                        logger.warning(
                            "%s retrying OpenAI chat/completions (attempt %d/%d) due to: %s",
                            tag,
                            int(attempt + 1),
                            int(max_attempts - 1),
                            str(e),
                        )
                    except Exception:
                        pass
                    _sleep_backoff(
                        attempt,
                        initial_s=float(self.cfg.retry_initial_backoff_s),
                        max_s=float(self.cfg.retry_max_backoff_s),
                    )
                    continue

                # Preserve the adapter's original error wording for urllib failures.
                if isinstance(e, urllib.error.HTTPError):
                    body2 = http_body
                    if body2 is None:
                        try:
                            body2 = (
                                e.read().decode("utf-8", errors="replace")
                                if hasattr(e, "read")
                                else ""
                            )
                        except Exception:
                            body2 = ""
                    raise RuntimeError(f"OpenAI chat/completions HTTP {e.code}: {body2}") from e
                if isinstance(e, urllib.error.URLError):
                    raise RuntimeError(f"OpenAI chat/completions request failed: {e}") from e
                raise

        raise RuntimeError("OpenAI chat/completions request failed (exhausted retries)")


@dataclass(frozen=True)
class GeminiGenAIConfig:
    """Config for the `google-genai` Python SDK (imported as `from google import genai`)."""

    model: str
    api_key: str | None = None  # defaults to env GOOGLE_API_KEY (or GEMINI_API_KEY) if omitted
    temperature: float = 0.0

    image_base_dir: Path | None = None  # resolves {"type":"image_ref","image_ref": "..."} parts
    include_thoughts: bool = False
    thinking_budget: int | None = None
    thinking_level: str | None = None
    timeout_s: float = 600.0
    # Best-effort retries for transient backend errors (overload/UNAVAILABLE/timeouts).
    max_retries: int = 10
    retry_initial_backoff_s: float = 0.5
    retry_max_backoff_s: float = 60.0
    # Gemini performance knobs.
    #
    # If None, the adapter will consult env vars:
    # - GEMINI_ENABLE_STREAMING (default false)
    streaming: bool | None = None
    # Streaming: if true, stop early once a complete JSON object is emitted.
    # If None, defaults to true when streaming is enabled.
    stream_early_stop_json: bool | None = None


def _messages_to_plain_text(messages: list[dict[str, Any]]) -> str:
    """Convert our internal chat messages format to a single text prompt.

    This keeps the adapter lightweight and backend-agnostic. It is sufficient for text-only
    selectors whose structured output is emitted as JSON-in-text.
    """

    parts: list[str] = []
    for m in messages:
        role = str(m.get("role", ""))
        content = m.get("content")
        if isinstance(content, list):
            # Best-effort: include only text parts.
            text_chunks: list[str] = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    text_chunks.append(str(p.get("text", "")))
            content_str = "\n".join([c for c in text_chunks if c])
        else:
            content_str = "" if content is None else str(content)
        if role:
            parts.append(f"[{role}]\n{content_str}".strip())
        else:
            parts.append(str(content_str).strip())
    return "\n\n".join([p for p in parts if p])


def _resolve_image_ref_bytes(image_ref: str, *, base_dir: Path | None) -> tuple[bytes, str]:
    """Load an image_ref (local path) and return (bytes, mime_type)."""

    p = Path(str(image_ref))
    if not p.is_absolute():
        base_dir = base_dir or Path.cwd()
        p = (base_dir / p).resolve()
    b = p.read_bytes()
    mime = _guess_mime_from_path(p)
    return b, mime


def _messages_to_gemini_vision_payload(
    messages: list[dict[str, Any]],
    *,
    image_base_dir: Path | None,
) -> tuple[str | None, list[Any]]:
    """Convert internal messages to (system_instruction_text, contents_list) for Gemini vision.

    - system_instruction_text: concatenated system message content (best-effort)
    - contents_list: list mixing strings and `types.Part` (text and images), preserving part order
    """

    # Import types lazily so this module stays importable without google-genai installed.
    from google.genai import types  # type: ignore

    system_chunks: list[str] = []
    contents: list[Any] = []

    def flush_text(buf: list[str]) -> None:
        if not buf:
            return
        s = "\n".join([x for x in buf if x]).strip()
        if s:
            contents.append(s)
        buf.clear()

    text_buf: list[str] = []

    for m in messages:
        role = str(m.get("role", ""))
        content = m.get("content")

        if role == "system":
            if isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        system_chunks.append(str(p.get("text", "")))
            else:
                system_chunks.append("" if content is None else str(content))
            continue

        if isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                ptype = p.get("type")
                if ptype == "text":
                    text_buf.append(str(p.get("text", "")))
                elif ptype == "image_ref":
                    flush_text(text_buf)
                    ref = str(p.get("image_ref", ""))
                    if not ref:
                        continue
                    img_bytes, mime = _resolve_image_ref_bytes(ref, base_dir=image_base_dir)
                    contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
        else:
            text_buf.append("" if content is None else str(content))

    flush_text(text_buf)

    system_text = "\n".join([s for s in system_chunks if s]).strip() or None
    return system_text, contents


def _gemini_usage_metadata_to_usage_dict(um: Any) -> dict[str, Any]:
    """Normalize Gemini usage_metadata into a stable dict.

    Notes (per Gemini docs):
    - `prompt_token_count` may exclude cached content tokens when context caching is used.
    - `cached_content_token_count` (if present) reports cached input tokens.
    - Thinking models may report `thoughts_token_count`.
    - `total_token_count` may include cached + thoughts (SDK/version dependent).
    """

    out: dict[str, Any] = {"raw": str(um)}
    pt = getattr(um, "prompt_token_count", None)
    ct = getattr(um, "candidates_token_count", None)
    tt = getattr(um, "total_token_count", None)
    cached = getattr(um, "cached_content_token_count", None)
    thoughts = getattr(um, "thoughts_token_count", None)

    # Preserve raw breakdown keys for debugging/comparisons.
    #
    # IMPORTANT: Different Gemini SDK/model versions interpret these fields differently.
    # We therefore DO NOT try to "recompute" totals; we simply expose the raw counters.
    if pt is not None:
        # `prompt_token_count` as reported by the SDK.
        out["prompt_tokens"] = int(pt)
    if cached is not None:
        # `cached_content_token_count` as reported by the SDK.
        # Some versions treat this as a separate counter; others treat it as a subset of prompt
        # tokens.
        out["cached_prompt_tokens"] = int(cached)
    if ct is not None:
        out["output_tokens"] = int(ct)
    if thoughts is not None:
        out["thoughts_tokens"] = int(thoughts)
    if tt is not None:
        out["total_tokens"] = int(tt)

    return out


class GeminiGenAIAdapter:
    """Model adapter for Gemini via the `google-genai` SDK (optional dependency).

    This adapter supports:
    - text-only prompts, and
    - multimodal prompts when messages contain `{"type":"image_ref","image_ref": ...}` parts
      (e.g., overlay PNGs).

    It does not use Gemini native function calling; tools (if any) should be emitted as
    JSON-in-text.
    """

    def __init__(self, cfg: GeminiGenAIConfig) -> None:
        self.cfg = cfg
        # Debug info for the last call (for trace/reporting). Not thread-safe by design.
        self.last_thoughts: list[str] = []
        self.last_usage: dict[str, Any] | None = None
        try:
            importlib.import_module("google.genai")
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "GeminiGenAIAdapter requires the `google-genai` package. "
                "Install it (e.g., `pip install google-genai`) and set env GEMINI_API_KEY."
            ) from e

        # Collect available API keys from config and environment.
        self._api_keys: list[str] = []
        if self.cfg.api_key:
            self._api_keys.append(self.cfg.api_key)

        # Support GEMINI_API_KEY, and backups GEMINI_API_KEY_2, etc.
        env_vars = ["GEMINI_API_KEY"]
        for i in range(2, 10):  # Check up to GEMINI_API_KEY_5
            env_vars.append(f"GEMINI_API_KEY_{i}")

        for var in env_vars:
            val = os.environ.get(var)
            if val and val not in self._api_keys:
                self._api_keys.append(val)

        if not self._api_keys:
            # Fallback to default client behavior (it might still find a key in env).
            self._api_keys = [None]  # type: ignore

        self._current_key_idx = 0
        self._init_client()

    def _init_client(self) -> None:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        key = self._api_keys[self._current_key_idx]
        timeout_ms = int(self.cfg.timeout_s * 1000)
        http_opts = types.HttpOptions(timeout=timeout_ms)

        self._client = (
            genai.Client(api_key=key, http_options=http_opts)
            if key
            else genai.Client(http_options=http_opts)
        )

    def _rotate_key(self) -> bool:
        """Switch to the next available API key. Returns True if a switch occurred."""
        if len(self._api_keys) <= 1:
            return False
        self._current_key_idx = (self._current_key_idx + 1) % len(self._api_keys)
        self._init_client()
        return True

    def _streaming_enabled(self) -> bool:
        if self.cfg.streaming is not None:
            return bool(self.cfg.streaming)
        # Support both legacy/short and explicit names.
        return _env_flag(
            "GEMINI_ENABLE_STREAMING", default=_env_flag("GEMINI_STREAMING", default=False)
        )

    def call(self, *, messages: list[dict[str, Any]], obs: dict[str, Any]) -> str:
        _ = obs  # adapter uses only the prompt/messages
        self.last_usage = None

        from google.genai import types  # type: ignore

        cfg = types.GenerateContentConfig(
            temperature=float(self.cfg.temperature),
        )
        try:
            cfg.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
        except Exception:
            pass
        if bool(self.cfg.include_thoughts):
            # ThinkingConfig shape varies across SDK versions; populate best-effort.
            try:
                tc_kwargs: dict[str, Any] = {"include_thoughts": True}
                if self.cfg.thinking_budget is not None:
                    tc_kwargs["thinking_budget"] = int(self.cfg.thinking_budget)
                if self.cfg.thinking_level is not None:
                    tc_kwargs["thinking_level"] = str(self.cfg.thinking_level)
                cfg.thinking_config = types.ThinkingConfig(**tc_kwargs)
            except Exception:
                try:
                    cfg.thinking_config = types.ThinkingConfig(include_thoughts=True)
                except Exception:
                    pass

        # If any image_ref parts exist, use multimodal mode; otherwise use plain text prompt.
        has_image = False
        for m in messages:
            c = m.get("content")
            if isinstance(c, list) and any(
                isinstance(p, dict) and p.get("type") == "image_ref" for p in c
            ):
                has_image = True
                break

        max_attempts = int(max(1, 1 + int(self.cfg.max_retries)))

        # Cap the total wall-clock time of call() across all retries at 30 minutes.
        # The budget is checked before each attempt (an in-flight request is not
        # interrupted); once exhausted, we raise instead of retrying further.
        start_time = time.time()
        global_timeout_s = 1800.0
        use_streaming = self._streaming_enabled()
        # Streaming early-stop: default to true when streaming is enabled unless explicitly set.
        early_stop_json = (
            bool(self.cfg.stream_early_stop_json)
            if self.cfg.stream_early_stop_json is not None
            else _env_flag("GEMINI_STREAM_EARLY_STOP_JSON", default=bool(use_streaming))
        )

        for attempt in range(max_attempts):
            if time.time() - start_time > global_timeout_s:
                raise RuntimeError(f"Gemini request timed out (total time > {global_timeout_s}s)")

            try:
                if has_image:
                    system_text, contents = _messages_to_gemini_vision_payload(
                        messages, image_base_dir=self.cfg.image_base_dir
                    )
                    if system_text:
                        cfg.system_instruction = system_text

                    if use_streaming:
                        stream = self._client.models.generate_content_stream(
                            model=str(self.cfg.model),
                            contents=contents,
                            config=cfg,
                        )
                        acc: list[str] = []
                        # Best-effort: usage for streaming (may only appear on final chunks).
                        stream_usage = None
                        for chunk in stream:
                            try:
                                stream_usage = getattr(chunk, "usage_metadata", stream_usage)
                            except Exception:
                                pass
                            t = getattr(chunk, "text", None)
                            if isinstance(t, str) and t:
                                acc.append(t)
                                if early_stop_json:
                                    prefix = _extract_first_complete_json_object_prefix(
                                        "".join(acc)
                                    )
                                    if prefix is not None:
                                        try:
                                            if hasattr(stream, "close"):
                                                stream.close()
                                        except Exception:
                                            pass
                                        self.last_thoughts = []
                                        try:
                                            self.last_usage = (
                                                _gemini_usage_metadata_to_usage_dict(stream_usage)
                                                if stream_usage is not None
                                                else None
                                            )
                                        except Exception:
                                            self.last_usage = None
                                        return str(prefix)
                        text = "".join(acc)
                        # Best-effort: attach usage metadata if available (not guaranteed).
                        try:
                            if stream_usage is not None:
                                self.last_usage = _gemini_usage_metadata_to_usage_dict(stream_usage)
                        except Exception:
                            self.last_usage = None
                        self.last_thoughts = []
                        if _is_empty_model_text(text):
                            raise RuntimeError("empty_response")
                        return str(text)

                    resp = self._client.models.generate_content(
                        model=str(self.cfg.model),
                        contents=contents,
                        config=cfg,
                    )
                else:
                    prompt = _messages_to_plain_text(messages)
                    # Keep legacy behavior for text-only prompts (no system_instruction split).
                    if use_streaming:
                        stream = self._client.models.generate_content_stream(
                            model=str(self.cfg.model),
                            contents=str(prompt),
                            config=cfg,
                        )
                        acc: list[str] = []
                        stream_usage = None
                        for chunk in stream:
                            try:
                                stream_usage = getattr(chunk, "usage_metadata", stream_usage)
                            except Exception:
                                pass
                            t = getattr(chunk, "text", None)
                            if isinstance(t, str) and t:
                                acc.append(t)
                                if early_stop_json:
                                    prefix = _extract_first_complete_json_object_prefix(
                                        "".join(acc)
                                    )
                                    if prefix is not None:
                                        try:
                                            if hasattr(stream, "close"):
                                                stream.close()
                                        except Exception:
                                            pass
                                        self.last_thoughts = []
                                        try:
                                            self.last_usage = (
                                                _gemini_usage_metadata_to_usage_dict(stream_usage)
                                                if stream_usage is not None
                                                else None
                                            )
                                        except Exception:
                                            self.last_usage = None
                                        return str(prefix)
                        text = "".join(acc)
                        try:
                            if stream_usage is not None:
                                self.last_usage = _gemini_usage_metadata_to_usage_dict(stream_usage)
                        except Exception:
                            self.last_usage = None
                        self.last_thoughts = []
                        if _is_empty_model_text(text):
                            raise RuntimeError("empty_response")
                        return str(text)

                    resp = self._client.models.generate_content(
                        model=str(self.cfg.model),
                        contents=str(prompt),
                        config=cfg,
                    )

                # Best-effort: usage accounting (Gemini usage metadata).
                try:
                    um = getattr(resp, "usage_metadata", None)
                    if um is not None:
                        self.last_usage = _gemini_usage_metadata_to_usage_dict(um)
                except Exception:
                    self.last_usage = None

                # Extract returned "thoughts" if present.
                #
                # In google-genai types:
                # - `Part.thought` is typically an Optional[bool] flag (not the reasoning text)
                # - thought text (when provided) typically appears as `Part.text` with
                #   `Part.thought == True`
                self.last_thoughts = []
                try:
                    cands = getattr(resp, "candidates", None)
                    if isinstance(cands, list) and cands:
                        content = getattr(cands[0], "content", None)
                        parts = getattr(content, "parts", None)
                        if isinstance(parts, list):
                            for p in parts:
                                is_thought = getattr(p, "thought", None)
                                if is_thought is True:
                                    t = getattr(p, "text", None)
                                    if isinstance(t, str) and t.strip():
                                        self.last_thoughts.append(t.strip())
                                    else:
                                        self.last_thoughts.append(str(p))
                except Exception:
                    self.last_thoughts = []

                # The SDK provides `response.text` when content is text.
                try:
                    text = resp.text
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(f"Unexpected Gemini response: {resp}") from e

                if _is_empty_model_text(text):
                    try:
                        cands = getattr(resp, "candidates", None)
                        n_cands = len(cands) if isinstance(cands, list) else None
                        logger.warning(
                            "%s Gemini returned empty text (candidates=%s)", _shard_tag(), n_cands
                        )
                    except Exception:
                        pass
                    raise RuntimeError("empty_response")
                return str(text)

            except Exception as e:  # noqa: BLE001
                is_last = attempt >= (max_attempts - 1)
                if (not is_last) and _is_retryable_exception(e):
                    # If quota is exhausted or rate limited, try switching to a backup API key.
                    err_msg = str(e).lower()
                    switched_key = False
                    if (
                        "429" in err_msg
                        or "resource_exhausted" in err_msg
                        or "RESOURCE_EXHAUSTED" in err_msg
                        or "quota" in err_msg
                        or "empty_response" in err_msg
                        or "504" in err_msg
                        or "deadline" in err_msg
                        or "Resource has been exhausted" in err_msg
                        or "You exceeded your current quota" in err_msg
                    ):
                        if self._rotate_key():
                            switched_key = True
                            logger.warning(
                                "%s Gemini quota hit/empty/timeout. Switched to backup API key "
                                "(index %d/%d).",
                                _shard_tag(),
                                int(self._current_key_idx + 1),
                                len(self._api_keys),
                            )

                    tag = _shard_tag()
                    try:
                        logger.warning(
                            "%s retrying Gemini generate_content (attempt %d/%d) due to: %s",
                            tag,
                            int(attempt + 1),
                            int(max_attempts - 1),
                            str(e),
                        )
                    except Exception:
                        pass

                    if switched_key:
                        # If we rotated keys, retry immediately (with tiny jitter) rather than full
                        # backoff.
                        time.sleep(0.5 + 2.0 * random.random())
                    else:
                        _sleep_backoff(
                            attempt,
                            initial_s=float(self.cfg.retry_initial_backoff_s),
                            max_s=float(self.cfg.retry_max_backoff_s),
                        )
                    continue
                raise

        raise RuntimeError("Gemini request failed (exhausted retries)")
