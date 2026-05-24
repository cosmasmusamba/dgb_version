"""
configs/loader.py
=================
Loads runtime_config.json, validates every section with Pydantic v2,
resolves {model_id} placeholders, and provides a singleton accessor.

v3.0.0: DGBConfig uses ConfigDict(extra="allow") so pipeline sections
(sources, pipeline_stages, pipeline, storage_quotas, normalizer,
language_filter, toxicity_filter, quality_scorer, deduplicator)
are preserved as attributes and accessible by pipeline consumers.
"""
from __future__ import annotations
import json, logging, os, re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from configs.constants import DEFAULT_MODEL_ID, RUNTIME_CONFIG_PATH, DeviceType

logger = logging.getLogger(__name__)


class ProjectConfig(BaseModel):
    name: str = "DGB"; version: str = "3.0.0"
    model_id: str = DEFAULT_MODEL_ID; description: str = ""

class PathsConfig(BaseModel):
    datasets_dir: str = "datasets"; checkpoints_dir: str = "checkpoints"
    logs_dir: str = "checkpoints/logs"; tokenizer_dir: str = "checkpoints/{model_id}/tokenizer"
    models_dir: str = "checkpoints/{model_id}/models"; cleaned_dir: str = "datasets/{model_id}/cleaned"
    raw_dir: str = "datasets/{model_id}/wk_raw"

class DatasetConfig(BaseModel):
    raw_glob: str = "wk_*.txt"; cleaned_prefix: str = "dgb1_cleaned_"
    chunk_size_chars: int = 500_000; min_line_length: int = 20
    max_line_length: int = 2_000; encoding: str = "utf-8"
    shuffle: bool = True; dedup: bool = True; validation_split: float = 0.1

class TokenizerConfig(BaseModel):
    vocab_size: int = 8_000; min_freq: int = 2
    pad_token: str = "<PAD>"; unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"; eos_token: str = "<EOS>"
    mask_token: str = "<MASK>"; sep_token: str = "<SEP>"
    special_tokens_file: str = "tokenizer/special_tokens/special_tokens.json"
    stop_words_file: str = "tokenizer/special_tokens/en_stop_words.json"
    num_merges: int = 7_000; byte_level: bool = False

    @field_validator("vocab_size","num_merges","min_freq")
    @classmethod
    def must_be_positive(cls, v):
        if v <= 0: raise ValueError(f"Expected positive integer, got {v}")
        return v

class TransformerConfig(BaseModel):
    vocab_size: int = 8_000; d_model: int = 256; n_heads: int = 8
    n_encoder_layers: int = 4; n_decoder_layers: int = 4; d_ff: int = 1_024
    dropout: float = 0.1; max_seq_len: int = 512; pad_idx: int = 0
    tie_embeddings: bool = True; layer_norm_eps: float = 1e-6
    attention_type: str = "scaled_dot_product"

    @model_validator(mode="after")
    def heads_divide_d_model(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        return self

class TrainingConfig(BaseModel):
    batch_size: int = 32; epochs: int = 30; learning_rate: float = 3e-4
    weight_decay: float = 0.01; warmup_steps: int = 1_000; grad_clip: float = 1.0
    save_every_epochs: int = 5; log_every_steps: int = 1; eval_every_steps: int = 500
    mixed_precision: bool = False; seed: int = 42; num_workers: int = 0; device: str = "auto"

    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        if v not in {d.value for d in DeviceType}: raise ValueError(f"device must be auto/cpu/cuda/mps")
        return v

class ApiConfig(BaseModel):
    host: str = "0.0.0.0"; port: int = 8000; reload: bool = False; log_level: str = "info"
    cors_origins: List[str] = Field(default_factory=list)
    cors_methods: List[str] = Field(default_factory=lambda: ["GET","POST","PUT","DELETE","OPTIONS"])
    cors_headers: List[str] = Field(default_factory=lambda: ["*"])
    api_prefix: str = "/api/v1"; docs_url: str = "/docs"; redoc_url: str = "/redoc"

class AuthConfig(BaseModel):
    algorithm: str = "HS256"; access_token_ttl_min: int = 30
    refresh_token_ttl_days: int = 7; secret_key_env: str = "DGB_SECRET_KEY"; bcrypt_rounds: int = 12

class StreamingConfig(BaseModel):
    log_server_port: int = 5555; sse_retry_ms: int = 3_000; ws_ping_interval: int = 20
    ws_ping_timeout: int = 10; queue_maxsize: int = 1_000; flush_interval_sec: float = 0.1

class LoggingConfig(BaseModel):
    level: str = "INFO"; format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"; file_enabled: bool = True
    file_max_bytes: int = 10_485_760; file_backup_count: int = 5
    training_log: str = "checkpoints/logs/{model_id}/training/training.log"
    progress_log: str = "checkpoints/logs/{model_id}/training/progress.txt"
    port_file: str = "checkpoints/logs/{model_id}/training/training_log_server_port.txt"

class MemoryConfig(BaseModel):
    warn_threshold_gb: float = 1.0; error_threshold_gb: float = 0.25; gc_interval_steps: int = 100

class SearchConfig(BaseModel):
    default_provider: str = "brave"; brave_api_key_env: str = "BRAVE_API_KEY"
    serpapi_key_env: str = "SERPAPI_KEY"; tavily_key_env: str = "TAVILY_KEY"
    num_results: int = 5; fetch_full_text: bool = True; rewrite_queries: bool = True

class PipelineRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    max_concurrent_sources: int = 2; max_shard_bytes: int = 536_870_912
    max_shard_records: int = 1_000_000; batch_size: int = 1_000
    checkpoint_interval: int = 10; log_every_batches: int = 10

class PipelineStagesConfig(BaseModel):
    normalize: bool = True; language_filter: bool = True
    toxicity_filter: bool = True; quality_scorer: bool = True; deduplicator: bool = True

class StorageQuotasConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    global_max_gb: float = 0.0; safety_margin_gb: float = 10.0
    wikipedia_gb: float = 100.0; stackexchange_gb: float = 50.0
    arxiv_gb: float = 30.0; gutenberg_gb: float = 20.0
    commoncrawl_gb: float = 200.0; github_gb: float = 80.0


class DGBConfig(BaseModel):
    """
    Root configuration. ConfigDict(extra="allow") ensures any key present in
    runtime_config.json that is not explicitly modelled is still accessible
    as cfg.<key> — critical for pipeline sections like sources, normalizer,
    language_filter, toxicity_filter, quality_scorer, deduplicator.
    """
    model_config = ConfigDict(extra="allow")

    project:     ProjectConfig     = Field(default_factory=ProjectConfig)
    paths:       PathsConfig       = Field(default_factory=PathsConfig)
    dataset:     DatasetConfig     = Field(default_factory=DatasetConfig)
    tokenizer:   TokenizerConfig   = Field(default_factory=TokenizerConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    training:    TrainingConfig    = Field(default_factory=TrainingConfig)
    api:         ApiConfig         = Field(default_factory=ApiConfig)
    auth:        AuthConfig        = Field(default_factory=AuthConfig)
    streaming:   StreamingConfig   = Field(default_factory=StreamingConfig)
    logging:     LoggingConfig     = Field(default_factory=LoggingConfig)
    memory:      MemoryConfig      = Field(default_factory=MemoryConfig)
    search:      SearchConfig      = Field(default_factory=SearchConfig)
    pipeline:        PipelineRuntimeConfig = Field(default_factory=PipelineRuntimeConfig)
    pipeline_stages: PipelineStagesConfig  = Field(default_factory=PipelineStagesConfig)
    storage_quotas:  StorageQuotasConfig   = Field(default_factory=StorageQuotasConfig)
    # Pipeline processor configs — kept as plain dicts so consumers read them directly
    sources:           Optional[Dict[str, Any]] = None
    normalizer:        Optional[Dict[str, Any]] = None
    language_filter:   Optional[Dict[str, Any]] = None
    toxicity_filter:   Optional[Dict[str, Any]] = None
    quality_scorer:    Optional[Dict[str, Any]] = None
    deduplicator:      Optional[Dict[str, Any]] = None
    metadata_enricher: Optional[Dict[str, Any]] = None

    def resolve_path(self, template: str) -> str:
        return template.replace("{model_id}", self.project.model_id)
    def tokenizer_dir(self) -> Path: return Path(self.resolve_path(self.paths.tokenizer_dir))
    def models_dir(self)   -> Path: return Path(self.resolve_path(self.paths.models_dir))
    def cleaned_dir(self)  -> Path: return Path(self.resolve_path(self.paths.cleaned_dir))
    def raw_dir(self)      -> Path: return Path(self.resolve_path(self.paths.raw_dir))
    def training_log(self) -> Path: return Path(self.resolve_path(self.logging.training_log))
    def progress_log(self) -> Path: return Path(self.resolve_path(self.logging.progress_log))
    def port_file(self)    -> Path: return Path(self.resolve_path(self.logging.port_file))


_ENV_PREFIX = "DGB_"

def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX): continue
        parts = key[len(_ENV_PREFIX):].lower().split("__", 1)
        if len(parts) != 2: continue
        section, field = parts
        if section not in data or not isinstance(data[section], dict): continue
        if field not in data[section]: continue
        existing = data[section][field]
        try:
            if isinstance(existing, bool):   data[section][field] = value.lower() in ("1","true","yes")
            elif isinstance(existing, int):   data[section][field] = int(value)
            elif isinstance(existing, float): data[section][field] = float(value)
            elif isinstance(existing, list):  data[section][field] = [v.strip() for v in value.split(",")]
            else:                             data[section][field] = value
        except (ValueError, TypeError) as exc:
            logger.warning("Cannot apply env override %s=%s: %s", key, value, exc)
    return data


@lru_cache(maxsize=1)
def get_config(config_path: Optional[str] = None) -> DGBConfig:
    path = Path(config_path) if config_path else RUNTIME_CONFIG_PATH
    if not path.exists():
        logger.warning("Config not found at %s — using defaults.", path)
        raw: Dict[str, Any] = {}
    else:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for sk in list(raw.keys()):
            if isinstance(raw[sk], dict):
                raw[sk] = {k: v for k, v in raw[sk].items() if not k.startswith("_")}
        logger.info("Config loaded from %s", path)
    raw = _apply_env_overrides(raw)
    cfg = DGBConfig(**raw)
    logger.debug("Config validated: %s v%s model_id=%s", cfg.project.name, cfg.project.version, cfg.project.model_id)
    return cfg


def reload_config(config_path: Optional[str] = None) -> DGBConfig:
    get_config.cache_clear()
    return get_config(config_path)