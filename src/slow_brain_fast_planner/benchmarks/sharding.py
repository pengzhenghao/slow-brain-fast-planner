from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass


def _stable_hash_u64(s: str) -> int:
    """Deterministic hash for sharding (stable across processes/runs/machines)."""
    h = hashlib.sha1(s.encode("utf-8")).digest()  # stable, fast enough
    return int.from_bytes(h[:8], byteorder="little", signed=False)


def shard_id_for_key(key: str, *, num_shards: int) -> int:
    if int(num_shards) <= 0:
        raise ValueError("num_shards must be > 0")
    return int(_stable_hash_u64(str(key)) % int(num_shards))


def key_in_shard(key: str, *, shard_id: int, num_shards: int) -> bool:
    return int(shard_id_for_key(key, num_shards=int(num_shards))) == int(shard_id)


@dataclass(frozen=True)
class ShardConfig:
    """Episode-based sharding configuration.

    This is intentionally simple: shard assignment is a deterministic function of `episode_id`.
    """

    shard_id: int = 0
    num_shards: int = 1

    @property
    def enabled(self) -> bool:
        return int(self.num_shards) > 1

    def contains_episode(self, episode_id: str) -> bool:
        return key_in_shard(
            str(episode_id), shard_id=int(self.shard_id), num_shards=int(self.num_shards)
        )


def shard_config_from_env(*, default_num_shards: int = 1, default_shard_id: int = 0) -> ShardConfig:
    """Read sharding config from common distributed env vars.

    Supports:
    - PyTorch DDP: WORLD_SIZE / RANK
    - SLURM: SLURM_NTASKS / SLURM_PROCID

    If env vars are missing, falls back to defaults.
    """

    def _int(name: str) -> int | None:
        v = os.environ.get(name)
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    world = _int("WORLD_SIZE")
    rank = _int("RANK")
    if world is None or rank is None:
        world = _int("SLURM_NTASKS")
        rank = _int("SLURM_PROCID")

    num_shards = int(world) if world is not None else int(default_num_shards)
    shard_id = int(rank) if rank is not None else int(default_shard_id)
    if num_shards <= 0:
        num_shards = 1
    if shard_id < 0:
        shard_id = 0
    if shard_id >= num_shards:
        # Don't hard-crash; keep deterministic but safe.
        shard_id = shard_id % num_shards
    return ShardConfig(shard_id=shard_id, num_shards=num_shards)


def shard_episode_ids(episode_ids: list[str], *, shard: ShardConfig) -> list[str]:
    """Filter episode_ids to those assigned to this shard (stable order preserved)."""
    return [eid for eid in episode_ids if shard.contains_episode(str(eid))]


def shard_tag_from_env() -> str:
    """Best-effort tag for logging (e.g. "[shard 3/16]") based on distributed env vars."""
    try:
        cfg = shard_config_from_env()
        return f"[shard {int(cfg.shard_id)}/{int(cfg.num_shards)}]"
    except Exception:
        return "[shard ?/?]"
