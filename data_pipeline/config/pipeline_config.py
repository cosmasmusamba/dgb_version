"""
data_pipeline/config/pipeline_config.py
=========================================
Pipeline-specific config loader with validation and dynamic overrides.

Wraps the top-level DGBConfig to provide typed, validated access to
all pipeline-specific configuration sections. Also provides runtime
override support so live adjustments can be applied without restart.

All pipeline behaviour — source enables, quota limits, stage toggles,
concurrency, thresholds — flows through this single entry point.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class PipelineCfg:
    """Validated pipeline runtime configuration."""
    max_concurrent_sources: int   = 2
    max_shard_bytes:        int   = 536_870_912    # 512 MB
    max_shard_records:      int   = 1_000_000
    batch_size:             int   = 1_000
    checkpoint_interval:    int   = 10             # batches between saves
    log_every_batches:      int   = 10

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineCfg":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SourceCfg:
    """Per-source configuration validated and normalised."""
    source_name:        str
    enabled:            bool  = True
    extractor_type:     str   = ""
    language:           str   = "en"
    dump_urls:          List[str] = field(default_factory=list)
    batch_size:         int   = 1_000
    max_docs:           int   = 0
    max_bytes:          int   = 0
    download_timeout:   int   = 60
    max_retries:        int   = 5
    retry_backoff:      float = 2.0
    stream_chunk_bytes: int   = 65_536
    rate_limit_rps:     float = 0.0
    extra:              Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "SourceCfg":
        if hasattr(d, "__dict__"):
            d = d.__dict__
        return cls(
            source_name=name,
            enabled=d.get("enabled", True),
            extractor_type=d.get("extractor_type", name),
            language=d.get("language", "en"),
            dump_urls=d.get("dump_urls", []),
            batch_size=d.get("batch_size", 1_000),
            max_docs=d.get("max_docs", 0),
            max_bytes=d.get("max_bytes", 0),
            download_timeout=d.get("download_timeout", 60),
            max_retries=d.get("max_retries", 5),
            retry_backoff=d.get("retry_backoff", 2.0),
            stream_chunk_bytes=d.get("stream_chunk_bytes", 65_536),
            rate_limit_rps=d.get("rate_limit_rps", 0.0),
            extra=d.get("extra", {}),
        )


@dataclass
class StorageQuotas:
    global_max_gb:   float = 0.0
    safety_margin_gb: float = 10.0
    per_source:      Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "StorageQuotas":
        if hasattr(d, "__dict__"):
            d = d.__dict__
        per_source = {}
        for k, v in d.items():
            if k.endswith("_gb") and k not in ("global_max_gb", "safety_margin_gb"):
                per_source[k[:-3]] = float(v)
        return cls(
            global_max_gb=d.get("global_max_gb", 0.0),
            safety_margin_gb=d.get("safety_margin_gb", 10.0),
            per_source=per_source,
        )

    def bytes_for(self, source: str) -> int:
        gb = self.per_source.get(source, 0.0)
        return int(gb * 1024**3)


class PipelineConfigManager:
    """
    Thread-safe pipeline configuration manager with dynamic override support.

    Wraps DGBConfig and exposes typed accessors for all pipeline sections.
    Override keys can be pushed at runtime (e.g. from admin API) using
    set_override(), which takes effect on the next config read without
    requiring a process restart.
    """

    def __init__(self, cfg) -> None:
        self._cfg       = cfg
        self._overrides: Dict[str, Any] = {}
        self._lock      = threading.RLock()

    def set_override(self, key: str, value: Any) -> None:
        """
        Set a runtime override.
        key format: "section.field"  e.g. "pipeline.max_concurrent_sources"
        """
        with self._lock:
            self._overrides[key] = value
            logger.info("Pipeline config override: %s = %s", key, value)

    def clear_overrides(self) -> None:
        with self._lock:
            self._overrides.clear()

    @property
    def pipeline(self) -> PipelineCfg:
        raw = self._get_section("pipeline") or {}
        raw.update({
            k.split(".",1)[1]: v
            for k, v in self._overrides.items()
            if k.startswith("pipeline.")
        })
        return PipelineCfg.from_dict(raw)

    def source(self, name: str) -> Optional[SourceCfg]:
        sources = self._get_section("sources") or {}
        if hasattr(sources, "__dict__"):
            sources = sources.__dict__
        raw = sources.get(name)
        if raw is None:
            return None
        return SourceCfg.from_dict(name, raw)

    def enabled_sources(self) -> List[str]:
        sources = self._get_section("sources") or {}
        if hasattr(sources, "__dict__"):
            sources = sources.__dict__
        result = []
        for name, scfg in sources.items():
            if hasattr(scfg, "__dict__"):
                scfg = scfg.__dict__
            if scfg.get("enabled", True):
                result.append(name)
        return result

    @property
    def quotas(self) -> StorageQuotas:
        raw = self._get_section("storage_quotas") or {}
        return StorageQuotas.from_dict(raw)

    def stage_enabled(self, stage_name: str) -> bool:
        override_key = f"pipeline_stages.{stage_name}"
        if override_key in self._overrides:
            return bool(self._overrides[override_key])
        stages = self._get_section("pipeline_stages") or {}
        if hasattr(stages, "__dict__"):
            stages = stages.__dict__
        return stages.get(stage_name, True)

    def toggle_source(self, source: str, enabled: bool) -> None:
        """Dynamically enable or disable a source."""
        self.set_override(f"sources.{source}.enabled", enabled)

    def _get_section(self, name: str) -> Any:
        with self._lock:
            val = getattr(self._cfg, name, None)
            if val is None:
                return {}
            if hasattr(val, "__dict__"):
                d = dict(val.__dict__)
                # Apply overrides
                for k, v in self._overrides.items():
                    if k.startswith(f"{name}."):
                        field = k[len(name)+1:]
                        d[field] = v
                return d
            return val

    def summary(self) -> str:
        lines = [f"PipelineConfigManager (overrides: {len(self._overrides)})"]
        lines.append(f"  enabled sources: {self.enabled_sources()}")
        p = self.pipeline
        lines.append(
            f"  pipeline: concurrent={p.max_concurrent_sources}  "
            f"shard={p.max_shard_bytes//1024**2}MB  batch={p.batch_size}"
        )
        q = self.quotas
        lines.append(
            f"  quotas: global={q.global_max_gb}GB  safety={q.safety_margin_gb}GB"
        )
        return "\n".join(lines)


# ── Singleton ─────────────────────────────────────────────────────────────────
_manager: Optional[PipelineConfigManager] = None


def get_pipeline_config(cfg=None) -> PipelineConfigManager:
    global _manager
    if _manager is None:
        if cfg is None:
            from configs.loader import get_config
            cfg = get_config()
        _manager = PipelineConfigManager(cfg)
    return _manager
