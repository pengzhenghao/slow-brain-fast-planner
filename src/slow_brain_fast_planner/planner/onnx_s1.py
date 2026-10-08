from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OnnxS1PlannerConfig:
    model_path: str
    max_trajectories: int = 6
    nms_distance_threshold_m: float = 2.0
    # Model expects (B, T, C, H, W) float32 in [0,1], with H=288 and W=512.
    input_h: int = 288
    input_w: int = 512


def _preload_pip_nvidia_cuda_libs() -> None:
    """Best-effort preload of pip-installed NVIDIA libs for ORT CUDA EP.

    On some systems, the venv's `site-packages/nvidia/**/lib/*.so` directories are not in the
    dynamic loader search path. ONNXRuntime may then fail to load CUDA EP with errors like:
      "libcudnn.so.9: cannot open shared object file"

    Preloading these shared libraries with `RTLD_GLOBAL` makes them discoverable by SONAME
    when ORT dlopens its CUDA provider library.
    """

    try:
        import ctypes
        import os
        import sys
    except Exception:
        return

    # Keep this conservative and order-sensitive (cublasLt -> cublas -> cudnn).
    candidates: list[str] = []
    for path in sys.path:
        if "site-packages" not in path:
            continue
        candidates.extend(
            [
                os.path.join(path, "nvidia", "cublas", "lib", "libcublasLt.so.12"),
                os.path.join(path, "nvidia", "cublas", "lib", "libcublas.so.12"),
                os.path.join(path, "nvidia", "cudnn", "lib", "libcudnn.so.9"),
            ]
        )

    for lib in candidates:
        try:
            if os.path.exists(lib):
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            # Best-effort: if preloading fails, ORT may still work (CPU fallback, or system libs).
            continue


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


def _nms_endpoints(
    traj_xy: np.ndarray,
    scores: np.ndarray,
    *,
    max_trajectories: int,
    distance_threshold: float,
) -> np.ndarray:
    """Endpoint-distance NMS with deterministic tie-break."""

    traj_xy = np.asarray(traj_xy, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    K = int(scores.shape[0])
    if K == 0:
        return np.asarray([], dtype=np.int64)

    endpoints = traj_xy[:, -1, :2]
    order = sorted(range(K), key=lambda i: (-float(scores[i]), int(i)))

    keep: list[int] = []
    for i in order:
        if len(keep) >= int(max_trajectories):
            break
        ok = True
        for j in keep:
            dx = float(endpoints[i, 0] - endpoints[j, 0])
            dy = float(endpoints[i, 1] - endpoints[j, 1])
            if math.hypot(dx, dy) < float(distance_threshold):
                ok = False
                break
        if ok:
            keep.append(int(i))

    if not keep:
        keep = [int(order[0])]
    while len(keep) < int(max_trajectories):
        keep.append(int(keep[-1]))
    return np.asarray(keep[: int(max_trajectories)], dtype=np.int64)


class OnnxS1Planner:
    """Minimal ONNXRuntime wrapper for S1 planner inference (goal-less model)."""

    def __init__(self, cfg: OnnxS1PlannerConfig):
        self.cfg = cfg
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "onnxruntime is required for on-the-fly planner inference. "
                "Install it (e.g., `uv pip install onnxruntime`)."
            ) from e

        model_path = Path(cfg.model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Planner ONNX not found: {model_path}")

        available = list(ort.get_available_providers())
        # Prefer CUDA only if it is actually available in this environment.
        preferred: list[str] = []
        if "CUDAExecutionProvider" in available:
            preferred.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available:
            preferred.append("CPUExecutionProvider")
        if not preferred:
            # Extremely defensive fallback; ORT usually has CPU.
            preferred = available

        # If CUDA EP is available, try to preload pip-installed cuDNN/cuBLAS so ORT can dlopen them.
        if "CUDAExecutionProvider" in preferred:
            _preload_pip_nvidia_cuda_libs()

        # If CUDA is present-but-broken (missing cuDNN, etc.), fall back to CPU without spamming
        # errors.
        try:
            self._session = ort.InferenceSession(str(model_path), providers=preferred)
        except Exception:
            self._session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )

        # Cache input names for speed / robustness.
        self._input_names = {i.name for i in self._session.get_inputs()}

    def run(
        self,
        *,
        obs: np.ndarray,
        goal_point: np.ndarray | None = None,
        metric_spacing: np.ndarray | None = None,
        embodiment_id: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the model.

        Args:
          obs: float32 array (1, T=21, C=3, H=288, W=512) in [0,1]

        Returns:
          traj_raw: (1, N, P, 2)  (typically N=64, P=20)
          score_raw: (1, N)
        """

        x = np.asarray(obs, dtype=np.float32)
        if x.ndim != 5 or x.shape[0] != 1:
            raise ValueError(f"obs must be (1,T,C,H,W), got {x.shape}")

        feeds: dict[str, Any] = {"observation": x}
        # Most models require these two extra inputs.
        if "embodiment_id" in self._input_names:
            feeds["embodiment_id"] = np.asarray([[int(embodiment_id)]], dtype=np.int64)
        if "metric_spacing" in self._input_names:
            if metric_spacing is None:
                # Common default for non-goal planner runs; callers may override.
                feeds["metric_spacing"] = np.asarray([[0.0, 0.51, -0.2, 0.2]], dtype=np.float32)
            else:
                ms = np.asarray(metric_spacing, dtype=np.float32).reshape(1, 4)
                feeds["metric_spacing"] = ms
        if "goal_point" in self._input_names:
            if goal_point is None:
                raise ValueError(
                    "ONNX model requires goal_point input but goal_point was not provided."
                )
            gp = np.asarray(goal_point, dtype=np.float32).reshape(1, 3)
            feeds["goal_point"] = gp

        outs = self._session.run(None, feeds)
        if len(outs) < 2:
            raise RuntimeError("Unexpected ONNX outputs (expected traj, score).")
        traj_raw = np.asarray(outs[0])
        score_raw = np.asarray(outs[1])
        return traj_raw, score_raw

    def postprocess_to_k(
        self, traj_raw: np.ndarray, score_raw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (traj_k, score_k, kept_indices)."""

        traj = np.asarray(traj_raw, dtype=np.float64)
        score = np.asarray(score_raw, dtype=np.float64)
        if traj.ndim != 4 or traj.shape[0] != 1:
            raise ValueError(f"traj_raw must be (1,N,P,2), got {traj.shape}")
        if score.ndim != 2 or score.shape[0] != 1:
            raise ValueError(f"score_raw must be (1,N), got {score.shape}")

        traj0 = traj[0]
        score0 = score[0]
        keep = _nms_endpoints(
            traj0,
            score0,
            max_trajectories=int(self.cfg.max_trajectories),
            distance_threshold=float(self.cfg.nms_distance_threshold_m),
        )
        traj_k = traj0[keep]
        score_k = score0[keep]
        # Keep scores as raw logits (evaluation will softmax when needed).
        return np.asarray(traj_k, dtype=np.float64), np.asarray(score_k, dtype=np.float64), keep
