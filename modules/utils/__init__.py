"""
modules/utils/__init__.py
==========================
Public surface for all DGB utility modules.

Import anything from here:
    from modules.utils import (
        get_config,
        configure_logging, LogStage, TrainingLogger,
        RunContext, create_run_context, get_run_context,
        PathResolver, init_path_resolver, get_path_resolver,
        DynamicResourceManager, ResourceHandle,
        SystemProfile, get_system_profile,
        DeviceMonitor, DeviceSnapshot, SleepGuard,
        MemoryManager,
        chunk_file_lines,
        clean_lines, cleaning_stats,
        ProgressTracker, ProgressState,
        MetricsLogger, StepMetric, EpochMetric,
        UnifiedLogWriter, init_unified_log, get_unified_log,
        PipelineState, StageRecord, StageStatus,
        StreamEvent, StreamQueue, BroadcastHub,
        get_training_hub, reset_training_hub, sse_format,
        atomic_write_json, atomic_write_text, atomic_write_bytes,
        compute_checksum, write_checksum, verify_checksum,
        locked_file,
        list_files, iter_lines, ensure_dir, read_json, write_text,
        file_size_mb, file_line_count, latest_file_by_name,
        DGBError, ConfigError, DataError, FileError,
        TokenizerError, ModelError, TrainingError,
        InferenceError, OutOfMemoryError, APIError, AuthError,
        handle_errors, log_exception, safe_call, retry,
    )
"""

# ── Config ────────────────────────────────────────────────────────────────────
from configs.loader import get_config

# ── Logging ───────────────────────────────────────────────────────────────────
from modules.logging_config import (
    configure_logging,
    LogStage,
    set_log_stage,
    get_log_stage,
    TrainingLogger,
)

# ── Run identity ──────────────────────────────────────────────────────────────
from modules.utils.run_context import (
    RunContext,
    create_run_context,
    get_run_context,
    reset_run_context,
    latest_run_id,
)

# ── Path resolution ───────────────────────────────────────────────────────────
from modules.utils.path_resolver import (
    PathResolver,
    init_path_resolver,
    get_path_resolver,
)

# ── System & device ───────────────────────────────────────────────────────────
from modules.utils.system_detector import get_system_profile, SystemProfile
from modules.utils.device_monitor import DeviceMonitor, DeviceSnapshot, SleepGuard
from modules.utils.memory_manager import MemoryManager

# ── Resource management ───────────────────────────────────────────────────────
from modules.utils.dynamic_resource_manager import (
    DynamicResourceManager,
    ResourceHandle,
)

# ── Data utilities ────────────────────────────────────────────────────────────
from modules.utils.chunk_processor import chunk_file_lines
from modules.utils.data_cleaner import clean_lines, cleaning_stats

# ── File I/O ──────────────────────────────────────────────────────────────────
from modules.utils.file_handler import (
    list_files,
    iter_lines,
    ensure_dir,
    read_json,
    write_text,
    file_size_mb,
    file_line_count,
    latest_file_by_name,
)
from modules.utils.safe_writer import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_bytes,
    compute_checksum,
    write_checksum,
    verify_checksum,
)
from modules.utils.file_locker import locked_file

# ── Progress & metrics ────────────────────────────────────────────────────────
from modules.utils.progress_tracking import ProgressTracker, ProgressState
from modules.utils.metrics_logger import MetricsLogger, StepMetric, EpochMetric

# ── Structured logging ────────────────────────────────────────────────────────
from modules.utils.unified_log import (
    UnifiedLogWriter,
    init_unified_log,
    get_unified_log,
)

# ── Pipeline state machine ────────────────────────────────────────────────────
from modules.utils.pipeline_state import (
    PipelineState,
    StageRecord,
    StageStatus,
)

# ── Streaming ─────────────────────────────────────────────────────────────────
from modules.utils.streaming import (
    StreamEvent,
    StreamQueue,
    BroadcastHub,
    get_training_hub,
    reset_training_hub,
    sse_format,
)

# ── Error handling ────────────────────────────────────────────────────────────
from modules.utils.error_handler import (
    DGBError,
    ConfigError,
    DataError,
    DatasetError,
    FileError,
    FileLockError,
    TokenizerError,
    TokenizerLoadError,
    ModelError,
    ModelInitError,
    TrainingError,
    TrainingAlreadyRunningError,
    InferenceError,
    OutOfMemoryError,
    APIError,
    AuthError,
    GradientError,
    CheckpointError,
    PipelineStageError,
    handle_errors,
    log_exception,
    safe_call,
    retry,
)

__all__ = [
    # config
    "get_config",
    # logging
    "configure_logging", "LogStage", "set_log_stage", "get_log_stage", "TrainingLogger",
    # run identity
    "RunContext", "create_run_context", "get_run_context", "reset_run_context", "latest_run_id",
    # paths
    "PathResolver", "init_path_resolver", "get_path_resolver",
    # system
    "get_system_profile", "SystemProfile",
    "DeviceMonitor", "DeviceSnapshot", "SleepGuard",
    "MemoryManager",
    # resources
    "DynamicResourceManager", "ResourceHandle",
    # data
    "chunk_file_lines", "clean_lines", "cleaning_stats",
    # file I/O
    "list_files", "iter_lines", "ensure_dir", "read_json", "write_text",
    "file_size_mb", "file_line_count", "latest_file_by_name",
    "atomic_write_json", "atomic_write_text", "atomic_write_bytes",
    "compute_checksum", "write_checksum", "verify_checksum",
    "locked_file",
    # progress & metrics
    "ProgressTracker", "ProgressState",
    "MetricsLogger", "StepMetric", "EpochMetric",
    # unified log
    "UnifiedLogWriter", "init_unified_log", "get_unified_log",
    # pipeline state
    "PipelineState", "StageRecord", "StageStatus",
    # streaming
    "StreamEvent", "StreamQueue", "BroadcastHub",
    "get_training_hub", "reset_training_hub", "sse_format",
    # errors (canonical names)
    "DGBError", "ConfigError", "DataError", "FileError",
    "TokenizerError", "ModelError", "TrainingError",
    "InferenceError", "OutOfMemoryError", "APIError", "AuthError",
    "handle_errors", "log_exception", "safe_call", "retry",
    # errors (original names — kept for backward compat)
    "DatasetError", "FileLockError", "TokenizerLoadError",
    "ModelInitError", "TrainingAlreadyRunningError",
    "GradientError", "CheckpointError", "PipelineStageError",
]