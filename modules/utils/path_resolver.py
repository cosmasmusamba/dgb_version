"""
modules/utils/path_resolver.py
================================
Centralised path resolution for all DGB artifacts.

All paths are driven by runtime_config.json via DGBConfig.
Environment variable overrides allow external storage on large disks:

    DGB_RAW_DIR=E:\\Wikipedia\\dumps
    DGB_CLEANED_DIR=E:\\Wikipedia\\cleaned
    DGB_MODELS_DIR=D:\\models\\dgb
    DGB_LOGS_DIR=D:\\logs\\dgb

PathResolver resolves {model_id} placeholders, creates directories
on first access, and validates required paths exist before training.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ENV_OVERRIDES = {
    "raw_dir":       "DGB_RAW_DIR",
    "cleaned_dir":   "DGB_CLEANED_DIR",
    "tokenizer_dir": "DGB_TOKENIZER_DIR",
    "models_dir":    "DGB_MODELS_DIR",
    "logs_dir":      "DGB_LOGS_DIR",
}


class PathResolver:
    """
    Resolves all platform paths from config with optional env overrides.

    Parameters
    ----------
    model_id:   e.g. "dgb1"
    cfg:        DGBConfig instance
    """

    def __init__(self, model_id: str, cfg) -> None:
        self._model_id = model_id
        self._cfg      = cfg
        self._paths    = cfg.paths

    def _resolve(self, template: str, key: str = "") -> Path:
        """Apply {model_id} substitution, check env override, return Path."""
        env_key = _ENV_OVERRIDES.get(key, "")
        if env_key:
            env_val = os.environ.get(env_key, "")
            if env_val:
                return Path(env_val)
        return Path(template.replace("{model_id}", self._model_id))

    def _ensure(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── Public path accessors ──────────────────────────────────────────

    def raw_dir(self, create: bool = True) -> Path:
        p = self._resolve(self._paths.raw_dir, "raw_dir")
        return self._ensure(p) if create else p

    def cleaned_dir(self, create: bool = True) -> Path:
        p = self._resolve(self._paths.cleaned_dir, "cleaned_dir")
        return self._ensure(p) if create else p

    def tokenizer_dir(self, create: bool = True) -> Path:
        p = self._resolve(self._paths.tokenizer_dir, "tokenizer_dir")
        return self._ensure(p) if create else p

    def models_dir(self, create: bool = True) -> Path:
        p = self._resolve(self._paths.models_dir, "models_dir")
        return self._ensure(p) if create else p

    def logs_dir(self, create: bool = True) -> Path:
        p = self._resolve(self._paths.logs_dir, "logs_dir") / self._model_id / "training"
        return self._ensure(p) if create else p

    def training_log(self) -> Path:
        return self.logs_dir() / "training.log"

    def port_file(self) -> Path:
        return self.logs_dir() / "training_log_server_port.txt"

    # ── Validation ─────────────────────────────────────────────────────

    def require_cleaned(self) -> None:
        """Raise if no cleaned text files are available."""
        from modules.utils.file_handler import list_files
        cleaned = self.cleaned_dir(create=False)
        if not cleaned.exists() or not list_files(cleaned, "*.txt"):
            from modules.utils.error_handler import PathError
            raise PathError(
                f"No cleaned .txt files in {cleaned}. "
                "Run: python main_dataset_clean.py"
            )

    def require_tokenizer(self) -> None:
        """Raise if tokenizer checkpoint is not present."""
        tok_dir = self.tokenizer_dir(create=False)
        if not any(tok_dir.glob("*vocabulary.json")):
            from modules.utils.error_handler import TokenizerLoadError
            raise TokenizerLoadError(
                f"No tokenizer checkpoint in {tok_dir}. "
                "Run: python main_train_tokenizer.py"
            )

    def require_model(self) -> None:
        """Raise if no model checkpoint is present."""
        m_dir = self.models_dir(create=False)
        if not any(m_dir.glob("*.pt")):
            from modules.utils.error_handler import CheckpointError
            raise CheckpointError(
                f"No model checkpoint in {m_dir}. "
                "Run: python model_trainer.py"
            )

    def summary(self) -> str:
        return "\n".join([
            f"raw_dir:       {self.raw_dir(create=False)}",
            f"cleaned_dir:   {self.cleaned_dir(create=False)}",
            f"tokenizer_dir: {self.tokenizer_dir(create=False)}",
            f"models_dir:    {self.models_dir(create=False)}",
            f"logs_dir:      {self.logs_dir(create=False)}",
        ])


def init_path_resolver(
    model_id: str,
    cfg=None,
) -> PathResolver:
    """Construct a PathResolver from config (or load default config if None)."""
    if cfg is None:
        from configs.loader import get_config
        cfg = get_config()
    return PathResolver(model_id=model_id, cfg=cfg)


# Global resolver instance for convenience
_global_resolver = None

def get_path_resolver(model_id: str = "dgb1", cfg=None):
    """
    Get or create a global PathResolver instance.
    
    Parameters
    ----------
    model_id: Model identifier (default: "dgb1")
    cfg: DGBConfig instance (loads default if None)
    
    Returns
    -------
    PathResolver instance
    """
    global _global_resolver
    
    if _global_resolver is None:
        if cfg is None:
            from configs.loader import get_config
            cfg = get_config()
        _global_resolver = PathResolver(model_id=model_id, cfg=cfg)
    else:
        # If model_id or cfg changed, you might want to reinitialize
        # For simplicity, we'll just return the existing instance
        pass
    
    return _global_resolver

def reset_path_resolver():
    """Reset the global PathResolver instance (useful for testing)."""
    global _global_resolver
    _global_resolver = None