from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # PyYAML
except ImportError as e:
    raise RuntimeError("PyYAML is required. Add 'pyyaml' to your dependencies.") from e

try:
    # Pydantic v2
    from pydantic import Field, SecretStr
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError as e:
    raise RuntimeError(
        "pydantic>=2 and pydantic-settings are required. Add 'pydantic' and 'pydantic-settings' to your dependencies."
    ) from e


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base (override wins)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/object at root: {path}")
    return data


class Settings(BaseSettings):
    """Environment + secrets (Pydantic) plus a separate YAML runtime config.

    Intentional separation:
      - env/.secrets.env: pre-run config + secrets
      - YAML (strategy_yaml): runtime configuration (non-secret)

    Merge order for *runtime YAML* only: defaults < YAML.
    Secrets/config in env always win for their fields.
    """

    model_config = SettingsConfigDict(
        env_file=(".secrets.env", ".env"),
        env_file_encoding="utf-8",
        extra="allow",
    )

    # runtime (env)
    mode: str = "paper"  # paper | live | backtest
    log_level: str = "INFO"

    # runtime YAML location (env override allowed)
    strategy_yaml: str = "strategy.yaml"

    # OpenRouter / DSPy (secret in env / secrets file)
    openrouter_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    openrouter_model: str = "openrouter/o1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Redis
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"

    # Provider toggles
    mt5_enabled: bool = False
    ccxt_enabled: bool = True
    yfinance_enabled: bool = True
    cftc_enabled: bool = True

    # legacy compat
    base_url: str = "http://example.com"
    trading_api_key: SecretStr = Field(default_factory=lambda: SecretStr("your_api_key_here"))

    # loaded from YAML (runtime config)
    runtime: Dict[str, Any] = Field(default_factory=dict)


def load_settings(*, yaml_path: Optional[str] = None) -> Settings:
    """Load env/secrets via Pydantic and attach YAML runtime config.

    - Pydantic reads env vars and .secrets.env/.env.
    - YAML is read from Settings.strategy_yaml (or yaml_path override).
    - YAML is NOT used as a source for SecretStr fields.
    """

    s = Settings()

    effective_yaml_path = yaml_path or s.strategy_yaml
    y = load_yaml(effective_yaml_path)
    # Support either top-level runtime keys or {settings: {...}} style.
    y_runtime = y.get("settings", y)

    # Make sure runtime is always a dict
    if not isinstance(y_runtime, dict):
        raise ValueError(f"YAML runtime config must be a mapping/object: {effective_yaml_path}")

    # Attach runtime config; env never merges into runtime automatically.
    s.runtime = _deep_merge({}, y_runtime)

    return s


# single source of truth
settings = load_settings()

# backwards compatible constants
BASE_URL = settings.base_url
TRADING_API_KEY = settings.trading_api_key.get_secret_value()