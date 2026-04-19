from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import yaml  # PyYAML
except ImportError as e:
    raise RuntimeError(
        "PyYAML is required. Add 'pyyaml' to your dependencies."
    ) from e

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

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

@dataclass(frozen=True)
class Settings:
    # runtime
    mode: str = "paper"          # paper | live | backtest
    log_level: str = "INFO"

    # strategy
    strategy_yaml: str = "strategies/mtf_confluence.yaml"

    # OpenRouter / DSPy
    openrouter_api_key: str = ""
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
    trading_api_key: str = "your_api_key_here"

    # anything not modeled yet
    extra: Dict[str, Any] = field(default_factory=dict)

def load_settings(*, yaml_path: Optional[str] = None, env_prefix: str = "") -> Settings:
    """Merge order: defaults < YAML < environment overrides."""
    defaults = Settings()

    # choose YAML path
    effective_yaml_path = (
        yaml_path
        or os.getenv(f"{env_prefix}STRATEGY_YAML")
        or defaults.strategy_yaml
    )

    y = load_yaml(effective_yaml_path)
    y_settings = y.get("settings", y)

    # env overrides
    env_overrides: Dict[str, Any] = {}

    def env(name: str) -> Optional[str]:
        return os.getenv(f"{env_prefix}{name}")

    if env("MODE") is not None:
        env_overrides["mode"] = env("MODE")
    if env("LOG_LEVEL") is not None:
        env_overrides["log_level"] = env("LOG_LEVEL")
    if env("STRATEGY_YAML") is not None:
        env_overrides["strategy_yaml"] = env("STRATEGY_YAML")

    if env("OPENROUTER_API_KEY") is not None:
        env_overrides["openrouter_api_key"] = env("OPENROUTER_API_KEY")
    if env("OPENROUTER_MODEL") is not None:
        env_overrides["openrouter_model"] = env("OPENROUTER_MODEL")
    if env("OPENROUTER_BASE_URL") is not None:
        env_overrides["openrouter_base_url"] = env("OPENROUTER_BASE_URL")

    if env("REDIS_ENABLED") is not None:
        env_overrides["redis_enabled"] = _env_bool(f"{env_prefix}REDIS_ENABLED", defaults.redis_enabled)
    if env("REDIS_URL") is not None:
        env_overrides["redis_url"] = env("REDIS_URL")

    if env("MT5_ENABLED") is not None:
        env_overrides["mt5_enabled"] = _env_bool(f"{env_prefix}MT5_ENABLED", defaults.mt5_enabled)
    if env("CCXT_ENABLED") is not None:
        env_overrides["ccxt_enabled"] = _env_bool(f"{env_prefix}CCXT_ENABLED", defaults.ccxt_enabled)
    if env("YFINANCE_ENABLED") is not None:
        env_overrides["yfinance_enabled"] = _env_bool(f"{env_prefix}YFINANCE_ENABLED", defaults.yfinance_enabled)
    if env("CFTC_ENABLED") is not None:
        env_overrides["cftc_enabled"] = _env_bool(f"{env_prefix}CFTC_ENABLED", defaults.cftc_enabled)

    # legacy envs
    if env("BASE_URL") is not None:
        env_overrides["base_url"] = env("BASE_URL")
    if env("TRADING_API_KEY") is not None:
        env_overrides["trading_api_key"] = env("TRADING_API_KEY")

    merged = _deep_merge(defaults.__dict__, dict(y_settings or {}))
    merged = _deep_merge(merged, env_overrides)

    known_fields = set(Settings.__dataclass_fields__.keys())
    extra = {k: v for k, v in merged.items() if k not in known_fields}

    return Settings(
        mode=str(merged.get("mode", defaults.mode)),
        log_level=str(merged.get("log_level", defaults.log_level)),
        strategy_yaml=str(merged.get("strategy_yaml", effective_yaml_path)),

        openrouter_api_key=str(merged.get("openrouter_api_key", defaults.openrouter_api_key) or ""),
        openrouter_model=str(merged.get("openrouter_model", defaults.openrouter_model)),
        openrouter_base_url=str(merged.get("openrouter_base_url", defaults.openrouter_base_url)),

        redis_enabled=bool(merged.get("redis_enabled", defaults.redis_enabled)),
        redis_url=str(merged.get("redis_url", defaults.redis_url)),

        mt5_enabled=bool(merged.get("mt5_enabled", defaults.mt5_enabled)),
        ccxt_enabled=bool(merged.get("ccxt_enabled", defaults.ccxt_enabled)),
        yfinance_enabled=bool(merged.get("yfinance_enabled", defaults.yfinance_enabled)),
        cftc_enabled=bool(merged.get("cftc_enabled", defaults.cftc_enabled)),

        base_url=str(merged.get("base_url", defaults.base_url)),
        trading_api_key=str(merged.get("trading_api_key", defaults.trading_api_key)),

        extra=extra,
    )


# single source of truth
settings = load_settings()

# backwards compatible constants
BASE_URL = settings.base_url
TRADING_API_KEY = settings.trading_api_key
