"""
configs/constants.py
====================
Single source of truth for all platform constants, enums, and fixed values.
"""
from __future__ import annotations
import os
from enum import Enum
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
PROJECT_NAME:    Final[str] = "DGB"
PROJECT_VERSION: Final[str] = "3.0.0"
DEFAULT_MODEL_ID: Final[str] = "dgb1"
DATASETS_DIR:    Final[Path] = REPO_ROOT / "datasets"
CHECKPOINTS_DIR: Final[Path] = REPO_ROOT / "checkpoints"
LOGS_DIR:        Final[Path] = REPO_ROOT / "checkpoints" / "logs"
CONFIGS_DIR:     Final[Path] = REPO_ROOT / "configs"
RUNTIME_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "runtime_config.json"
SPECIAL_TOKENS_PATH: Final[Path] = REPO_ROOT / "tokenizer" / "special_tokens" / "special_tokens.json"
STOP_WORDS_PATH:     Final[Path] = REPO_ROOT / "tokenizer" / "special_tokens" / "en_stop_words.json"

def model_tokenizer_dir(model_id: str = DEFAULT_MODEL_ID) -> Path:
    return CHECKPOINTS_DIR / model_id / "tokenizer"
def model_weights_dir(model_id: str = DEFAULT_MODEL_ID) -> Path:
    return CHECKPOINTS_DIR / model_id / "models"
def model_logs_dir(model_id: str = DEFAULT_MODEL_ID) -> Path:
    return LOGS_DIR / model_id / "training"
def model_cleaned_dir(model_id: str = DEFAULT_MODEL_ID) -> Path:
    return DATASETS_DIR / model_id / "cleaned"
def model_raw_dir(model_id: str = DEFAULT_MODEL_ID) -> Path:
    return DATASETS_DIR / model_id / "wk_raw"

PAD_TOKEN: Final[str] = "<PAD>"
UNK_TOKEN: Final[str] = "<UNK>"
BOS_TOKEN: Final[str] = "<BOS>"
EOS_TOKEN: Final[str] = "<EOS>"
MASK_TOKEN: Final[str] = "<MASK>"
SEP_TOKEN: Final[str] = "<SEP>"
PAD_IDX: Final[int] = 0
UNK_IDX: Final[int] = 1
BOS_IDX: Final[int] = 2
EOS_IDX: Final[int] = 3
MASK_IDX: Final[int] = 4
SEP_IDX: Final[int] = 5
SPECIAL_TOKEN_LIST: Final[list] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN, SEP_TOKEN]
DEFAULT_VOCAB_SIZE: Final[int] = 8000
DEFAULT_NUM_MERGES: Final[int] = 7000
DEFAULT_MIN_FREQ: Final[int] = 2
BPE_EOW_SUFFIX: Final[str] = "</w>"
DEFAULT_D_MODEL: Final[int] = 256
DEFAULT_N_HEADS: Final[int] = 8
DEFAULT_N_LAYERS: Final[int] = 4
DEFAULT_D_FF: Final[int] = 1024
DEFAULT_DROPOUT: Final[float] = 0.1
DEFAULT_MAX_SEQ: Final[int] = 512
DEFAULT_LR: Final[float] = 3e-4
DEFAULT_WEIGHT_DECAY: Final[float] = 0.01
DEFAULT_GRAD_CLIP: Final[float] = 1.0
LAYER_NORM_EPS: Final[float] = 1e-6
DEFAULT_ENCODING: Final[str] = "utf-8"
TEMP_FILE_SUFFIX: Final[str] = ".dgb_tmp"
LOCK_TIMEOUT_SEC: Final[float] = 30.0
CHUNK_SIZE_BYTES: Final[int] = 65536
MAX_LINE_LENGTH: Final[int] = 2000
MIN_LINE_LENGTH: Final[int] = 20
LOG_SERVER_PORT: Final[int] = 5555
SSE_RETRY_MS: Final[int] = 3000
WS_PING_INTERVAL: Final[int] = 20
STREAM_QUEUE_MAXSIZE: Final[int] = 1000
FLUSH_INTERVAL_SEC: Final[float] = 0.1
ACCESS_TOKEN_TTL_MIN: Final[int] = 30
REFRESH_TOKEN_TTL_DAYS: Final[int] = 7
JWT_ALGORITHM: Final[str] = "HS256"
SECRET_KEY_ENV_VAR: Final[str] = "DGB_SECRET_KEY"
_FALLBACK_SECRET_KEY: Final[str] = "dgb-dev-secret-change-in-production-please"
MEM_WARN_GB: Final[float] = 1.0
MEM_ERROR_GB: Final[float] = 0.25
GC_INTERVAL: Final[int] = 100
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
LOG_MAX_BYTES: Final[int] = 10 * 1024 * 1024
LOG_BACKUP_COUNT: Final[int] = 5

class DeviceType(str, Enum):
    AUTO = "auto"; CPU = "cpu"; CUDA = "cuda"; MPS = "mps"
class LogLevel(str, Enum):
    DEBUG = "DEBUG"; INFO = "INFO"; WARNING = "WARNING"; ERROR = "ERROR"; CRITICAL = "CRITICAL"
class UserRole(str, Enum):
    ADMIN = "admin"; USER = "user"; READONLY = "readonly"
class TrainingPhase(str, Enum):
    DATASET_PREP = "dataset_preparation"; TOKENIZER = "tokenizer_training"
    MODEL = "model_training"; EVALUATION = "evaluation"; INFERENCE = "inference"
class TokenizerState(str, Enum):
    UNTRAINED = "untrained"; TRAINING = "training"; READY = "ready"; ERROR = "error"
class CheckpointStatus(str, Enum):
    PENDING = "pending"; SAVED = "saved"; VERIFIED = "verified"; CORRUPT = "corrupt"
class StreamEventType(str, Enum):
    LOG = "log"; PROGRESS = "progress"; METRIC = "metric"
    STATUS = "status"; ERROR = "error"; DONE = "done"
