"""
# PSEUDO / DESIGN NOTES
#
# This module is the HTTP entrypoint for the project.
#
# Responsibilities (v0):
# - Create and configure the Flask app instance
# - Provide a minimal health endpoint
# - Expose non-secret config inspection endpoints (system.ini + strategy.yaml)
# - Expose secret *presence* flags only (never return secret values)
#
# Security notes:
# - Never return secret values via HTTP.
# - Avoid debug=True in production. Debug can be enabled via env var.
#
# Related docs:
# - PRD.md sections 7.1 and 7.4
"""

from __future__ import annotations

import os
from typing import Any, Dict

from flask import Flask, jsonify

from config import Settings
from core.contracts import SecretFlags

app = Flask(__name__)

def _load_settings() -> Settings:
    settings = Settings()
    # Backwards compatible if config.py only has load_settings for now.
    # We'll extend config.py later to also load runtime + secrets.
    settings.load_settings()
    return settings

def _secret_presence_flags() -> SecretFlags:
    # Presence only; values never returned.
    flags: Dict[str, Any] = {
        "openrouter_api_key": bool(os.getenv("OPENROUTER_API_KEY")),
        "binance_api_key": bool(os.getenv("BINANCE_API_KEY")),
        "mt5_credentials": bool(os.getenv("MT5_LOGIN") and os.getenv("MT5_PASSWORD")),
    }
    return SecretFlags.validate_flags(flags)

@app.get("/")
def home() -> str:
    return "Trading Server Running!"

@app.get("/health")
def health():
    return jsonify(status="ok")

@app.get("/config/system")
def config_system():
    settings = _load_settings()
    # Non-secret: return full parsed INI dictionary
    return jsonify(settings.system)

@app.get("/config/runtime")
def config_runtime():
    # Placeholder until config.py loads strategy.yaml into settings.runtime
    # Return empty mapping so callers can rely on endpoint shape.
    return jsonify({})

@app.get("/config/secrets")
def config_secrets():
    flags = _secret_presence_flags()
    return jsonify(flags.model_dump())

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes", "on"}

    app.run(host=host, port=port, debug=debug)