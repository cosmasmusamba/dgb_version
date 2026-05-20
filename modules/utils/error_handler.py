"""
modules/utils/error_handler.py
================================
All custom exceptions for the DGB platform. Centralising them here
lets every module import one well-known location and lets the API
layer map exception types to HTTP status codes without circular imports.
"""
from __future__ import annotations


class DGBError(Exception):
    """Base class for all DGB platform errors."""
    http_status: int = 500
    error_code:  str = "DGB_ERROR"

    def __init__(self, message: str = "", *args) -> None:
        super().__init__(message, *args)
        self.message = message

    def to_dict(self) -> dict:
        return {"error_code": self.error_code, "message": self.message}


# ── Training errors ───────────────────────────────────────────────────────────

class GradientError(DGBError):
    """NaN or Inf gradient detected — training must halt."""
    http_status = 500
    error_code  = "GRADIENT_ERROR"


class OutOfMemoryError(DGBError):
    """Available RAM dropped below safe threshold."""
    http_status = 500
    error_code  = "OUT_OF_MEMORY"


class CheckpointError(DGBError):
    """Checkpoint save/load/verification failure."""
    http_status = 500
    error_code  = "CHECKPOINT_ERROR"


class ModelInitError(DGBError):
    """Model failed to initialise (bad config, missing weights, etc.)."""
    http_status = 500
    error_code  = "MODEL_INIT_ERROR"


class DatasetError(DGBError):
    """Dataset loading or streaming failure."""
    http_status = 500
    error_code  = "DATASET_ERROR"


class TrainingAlreadyRunningError(DGBError):
    """Attempt to launch training while a run is already active."""
    http_status = 409
    error_code  = "TRAINING_ALREADY_RUNNING"


# ── Tokenizer errors ──────────────────────────────────────────────────────────

class TokenizerNotTrainedError(DGBError):
    """Tokenizer method called before training or loading."""
    http_status = 503
    error_code  = "TOKENIZER_NOT_TRAINED"

    def __init__(self) -> None:
        super().__init__(
            "Tokenizer is not trained. Run: python main_pipeline.py"
        )


class TokenizerLoadError(DGBError):
    """Failed to load tokenizer checkpoint."""
    http_status = 503
    error_code  = "TOKENIZER_LOAD_ERROR"


# ── API / auth errors ─────────────────────────────────────────────────────────

class AuthError(DGBError):
    """Authentication or authorisation failure."""
    http_status = 401
    error_code  = "AUTH_ERROR"


class ForbiddenError(DGBError):
    """User does not have the required role."""
    http_status = 403
    error_code  = "FORBIDDEN"


class RateLimitError(DGBError):
    """Per-key rate limit exceeded."""
    http_status = 429
    error_code  = "RATE_LIMIT_EXCEEDED"

    def __init__(self, limit: int, window_sec: int) -> None:
        super().__init__(
            f"Rate limit of {limit} requests per {window_sec}s exceeded"
        )
        self.limit      = limit
        self.window_sec = window_sec


class NotFoundError(DGBError):
    """Requested resource does not exist."""
    http_status = 404
    error_code  = "NOT_FOUND"


class ValidationError(DGBError):
    """Request payload failed validation."""
    http_status = 422
    error_code  = "VALIDATION_ERROR"


class InferenceError(DGBError):
    """Model inference failed."""
    http_status = 500
    error_code  = "INFERENCE_ERROR"


# ── I/O errors ────────────────────────────────────────────────────────────────

class FileLockError(DGBError):
    """File lock could not be acquired within timeout."""
    http_status = 500
    error_code  = "FILE_LOCK_ERROR"

    def __init__(self, path: str, timeout: float) -> None:
        super().__init__(
            f"Could not acquire lock on {path} within {timeout:.1f}s"
        )


class ConfigError(DGBError):
    """Configuration file missing or invalid."""
    http_status = 500
    error_code  = "CONFIG_ERROR"


class PathError(DGBError):
    """Required path does not exist or cannot be created."""
    http_status = 500
    error_code  = "PATH_ERROR"


# ── Pipeline errors ───────────────────────────────────────────────────────────

class PipelineStageError(DGBError):
    """A pipeline stage failed — wraps the underlying exception."""
    http_status = 500
    error_code  = "PIPELINE_STAGE_ERROR"

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"Stage '{stage}' failed: {cause}")
        self.stage = stage
        self.cause = cause
