"""Configuration loader.

Loads:
- ``system.ini``  → ``settings.system``  (non-secret system-wide defaults)
- ``strategy.yaml`` → ``settings.runtime`` (run-specific backtest/gym config)
- ``.secrets.env`` / env vars → ``settings`` fields (secrets never logged)

Usage
-----
>>> from config import settings
>>> settings.system["llm"]["model"]
'openrouter/o1'
"""

from __future__ import annotations

import configparser
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SYSTEM_INI_PATH = "system.ini"
STRATEGY_YAML_PATH = "strategy.yaml"

try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YAML_AVAILABLE = False


class Settings:
    """Unified settings object for the trading research system."""

    def __init__(self) -> None:
        self.system: Dict[str, Any] = {}
        self.runtime: Dict[str, Any] = {}
        # Secret fields (populated from env; never logged or returned via API)
        self.openrouter_api_key: str = ""
        self.openai_api_key: str = ""

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_settings(
        self,
        system_ini_path: str = SYSTEM_INI_PATH,
        strategy_yaml_path: str = STRATEGY_YAML_PATH,
    ) -> None:
        """Load all configuration sources."""
        self._load_system_ini(system_ini_path)
        self._load_strategy_yaml(strategy_yaml_path)
        self._load_secrets()

    def _load_system_ini(self, path: str) -> None:
        """Load ``system.ini`` into ``self.system``."""
        cfg = configparser.ConfigParser()
        cfg.read(path)
        for section in cfg.sections():
            self.system[section] = dict(cfg.items(section))
        logger.debug("Loaded system.ini: sections=%s", list(self.system.keys()))

    def _load_strategy_yaml(self, path: str) -> None:
        """Load ``strategy.yaml`` into ``self.runtime``."""
        if not os.path.exists(path):
            return
        if not _YAML_AVAILABLE:
            logger.warning("PyYAML not installed; skipping %s.", path)
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            # Support both top-level mapping and nested `settings:` key
            self.runtime = raw.get("settings", raw)
            logger.debug("Loaded strategy.yaml.")
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)

    def _load_secrets(self) -> None:
        """Load secret values from environment (never from files in source control)."""
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def redacted_dict(self) -> Dict[str, Any]:
        """Return a safe dict with secret fields replaced by presence flags."""
        return {
            "system": self.system,
            "runtime": self.runtime,
            "secrets": {
                "openrouter_api_key": bool(self.openrouter_api_key),
                "openai_api_key": bool(self.openai_api_key),
            },
        }

    def get_io_dir(self) -> str:
        """Return the configured IO root directory."""
        return self.system.get("paths", {}).get("io_dir", "./io")

    def get_epochs_dir(self) -> str:
        """Return the canonical epoch artifacts directory."""
        return os.path.join(self.get_io_dir(), "epochs")


# Module-level singleton
settings = Settings()
settings.load_settings()

