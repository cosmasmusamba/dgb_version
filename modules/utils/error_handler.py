"""
modules/utils/error_handler.py
================================
All custom exceptions and error-handling utilities for the DGB platform.

Centralising exceptions here lets every module import one well-known
location, and lets the API layer map exception types to HTTP status codes
without circular imports.

Exception hierarchy
-------------------
DGBError
  ├── Training
  │     GradientError, OutOfMemoryError, CheckpointError,
  │     ModelInitError (= ModelError), TrainingError,
  │     TrainingAlreadyRunningError
  ├── Data
  │     DatasetError (= DataError), DataError
  ├── Tokenizer
  │     TokenizerNotTrainedError, TokenizerLoadError (= TokenizerError)
  ├── API / Auth
  │     AuthError, APIError, ForbiddenError, RateLimitError,
  │     NotFoundError, ValidationError
  ├── Inference
  │     InferenceError
  └── I/O / Config / Pipeline
        FileLockError, FileError, ConfigError, PathError,
        PipelineStageError

Helper functions
----------------
log_exception(exc, context)     — log with traceback, return exc
handle_errors(*exc_types)       — decorator that catches and re-raises as DGBError
safe_call(fn, *a, default, **k) — call fn; return default on any exception
retry(fn, attempts, delay, *a)  — retry fn up to N times with delay
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class DGBError(Exception):
    """Base class for all DGB platform errors."""
    http_status: int = 500
    error_code:  str = "DGB_ERROR"

    def __init__(self, message: str = "", *args) -> None:
        super().__init__(message, *args)
        self.message = message

    def to_dict(self) -> dict:
        return {"error_code": self.error_code, "message": self.message}


# ---------------------------------------------------------------------------
# Training errors
# ---------------------------------------------------------------------------

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


# Alias used across patched pipeline files
ModelError = ModelInitError


class TrainingError(DGBError):
    """General training failure not covered by a more specific type."""
    http_status = 500
    error_code  = "TRAINING_ERROR"


class TrainingAlreadyRunningError(DGBError):
    """Attempt to launch training while a run is already active."""
    http_status = 409
    error_code  = "TRAINING_ALREADY_RUNNING"


# ---------------------------------------------------------------------------
# Data errors
# ---------------------------------------------------------------------------

class DatasetError(DGBError):
    """Dataset loading or streaming failure."""
    http_status = 500
    error_code  = "DATASET_ERROR"


# Alias used across patched pipeline files
DataError = DatasetError


# ---------------------------------------------------------------------------
# Tokenizer errors
# ---------------------------------------------------------------------------

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


# Alias used across patched pipeline files
TokenizerError = TokenizerLoadError


# ---------------------------------------------------------------------------
# API / Auth errors
# ---------------------------------------------------------------------------

class AuthError(DGBError):
    """Authentication or authorisation failure."""
    http_status = 401
    error_code  = "AUTH_ERROR"


class APIError(DGBError):
    """General API-layer error not covered by a more specific type."""
    http_status = 500
    error_code  = "API_ERROR"


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


# ---------------------------------------------------------------------------
# Inference errors
# ---------------------------------------------------------------------------

class InferenceError(DGBError):
    """Model inference failed."""
    http_status = 500
    error_code  = "INFERENCE_ERROR"


# ---------------------------------------------------------------------------
# I/O errors
# ---------------------------------------------------------------------------

class FileLockError(DGBError):
    """File lock could not be acquired within timeout."""
    http_status = 500
    error_code  = "FILE_LOCK_ERROR"

    def __init__(self, path: str, timeout: float) -> None:
        super().__init__(
            f"Could not acquire lock on {path} within {timeout:.1f}s"
        )


# Alias used across patched pipeline files
FileError = FileLockError


class ConfigError(DGBError):
    """Configuration file missing or invalid."""
    http_status = 500
    error_code  = "CONFIG_ERROR"


class PathError(DGBError):
    """Required path does not exist or cannot be created."""
    http_status = 500
    error_code  = "PATH_ERROR"


# ---------------------------------------------------------------------------
# Pipeline errors
# ---------------------------------------------------------------------------

class PipelineStageError(DGBError):
    """A pipeline stage failed — wraps the underlying exception."""
    http_status = 500
    error_code  = "PIPELINE_STAGE_ERROR"

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"Stage '{stage}' failed: {cause}")
        self.stage = stage
        self.cause = cause


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def log_exception(
    exc:     Exception,
    context: str = "",
    level:   int = logging.ERROR,
) -> Exception:
    """
    Log `exc` with a full traceback and return it.

    Usage
    -----
        except Exception as exc:
            raise log_exception(exc, context="FinetuneTrainer.run")
    """
    prefix = f"[{context}] " if context else ""
    logger.log(level, "%s%s: %s", prefix, type(exc).__name__, exc, exc_info=True)
    return exc


def handle_errors(
    *exc_types: Type[Exception],
    wrap_as: Type[DGBError] = DGBError,
    context: str = "",
) -> Callable:
    """
    Decorator — catches `exc_types` and re-raises as `wrap_as`.

    If the caught exception is already a DGBError subclass it is
    re-raised unchanged.

    Usage
    -----
        @handle_errors(OSError, ValueError, wrap_as=DataError, context="loader")
        def load_file(path): ...
    """
    catch = exc_types or (Exception,)

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except DGBError:
                raise
            except catch as exc:
                ctx = context or fn.__qualname__
                log_exception(exc, context=ctx)
                raise wrap_as(f"{ctx}: {exc}") from exc
        return wrapper
    return decorator


def safe_call(
    fn:      Callable,
    *args:   Any,
    default: Any = None,
    context: str = "",
    **kwargs: Any,
) -> Any:
    """
    Call `fn(*args, **kwargs)` and return `default` on any exception.

    Logs the exception at WARNING level.  Never raises.

    Usage
    -----
        size = safe_call(path.stat, default=None)
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        ctx = context or getattr(fn, "__qualname__", str(fn))
        logger.warning("[safe_call:%s] %s: %s", ctx, type(exc).__name__, exc)
        return default


def retry(
    fn:       Callable,
    *args:    Any,
    attempts: int   = 3,
    delay:    float = 1.0,
    backoff:  float = 2.0,
    context:  str   = "",
    **kwargs: Any,
) -> Any:
    """
    Retry `fn(*args, **kwargs)` up to `attempts` times.

    Waits `delay` seconds between tries, multiplied by `backoff` each time.
    Raises the last exception if all attempts fail.

    Usage
    -----
        data = retry(fetch_page, url, attempts=3, delay=2.0)
    """
    ctx   = context or getattr(fn, "__qualname__", str(fn))
    wait  = delay
    last: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            logger.warning(
                "[retry:%s] attempt %d/%d failed: %s — retrying in %.1fs",
                ctx, attempt, attempts, exc, wait,
            )
            if attempt < attempts:
                time.sleep(wait)
                wait *= backoff
    logger.error("[retry:%s] all %d attempts failed", ctx, attempts)
    raise last