"""Flask server for the trading research system.

Endpoints
---------
GET  /                   – HTML dashboard UI
GET  /health             – JSON health + mode
GET  /config/system      – Non-secret system.ini config
GET  /config/runtime     – strategy.yaml runtime config
GET  /config/secrets     – Boolean secret presence flags only
GET  /epochs             – List all epoch artifact directories
GET  /epochs/<epoch_id>  – Manifest + evaluation for a specific epoch
POST /run                – Trigger a multi-epoch run (JSON body: {num_epochs: N})
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

from config import settings

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import re

_EPOCH_ID_RE = re.compile(r"^\d{4}$")


def _epochs_dir() -> Path:
    return Path(settings.get_epochs_dir())


def _read_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Trading Research Dashboard</title>
  <style>
    body { font-family: sans-serif; margin: 2rem; background: #f8f9fa; color: #212529; }
    h1   { color: #343a40; }
    h2   { margin-top: 2rem; color: #495057; }
    table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
    th, td { border: 1px solid #dee2e6; padding: 0.5rem 0.75rem; text-align: left; }
    th { background: #e9ecef; }
    tr:nth-child(even) { background: #f8f9fa; }
    .badge-pass { color: #155724; background: #d4edda; padding: 2px 6px; border-radius: 4px; }
    .badge-fail { color: #721c24; background: #f8d7da; padding: 2px 6px; border-radius: 4px; }
    .btn { display: inline-block; padding: 0.4rem 1rem; border: none; border-radius: 4px;
           cursor: pointer; font-size: 0.9rem; }
    .btn-primary { background: #007bff; color: #fff; }
    .btn-primary:hover { background: #0056b3; }
    #run-status { margin-top: 0.5rem; color: #495057; }
    pre { background: #e9ecef; padding: 1rem; border-radius: 4px; overflow: auto; }
  </style>
</head>
<body>
  <h1>&#x1F4C8; Trading Research Dashboard</h1>
  <p>Server is running. Use the controls below to manage epoch runs.</p>

  <h2>Run Epochs</h2>
  <label>Number of epochs:
    <input id="num-epochs" type="number" value="3" min="1" max="100"
           style="width:4rem; margin-left:0.5rem;" />
  </label>
  <button class="btn btn-primary" onclick="triggerRun()" style="margin-left:0.5rem;">
    &#x25B6; Run
  </button>
  <div id="run-status"></div>

  <h2>Epoch History</h2>
  <div id="epoch-table"><em>Loading&hellip;</em></div>

  <h2>API Quick Links</h2>
  <ul>
    <li><a href="/health">/health</a></li>
    <li><a href="/config/system">/config/system</a></li>
    <li><a href="/config/runtime">/config/runtime</a></li>
    <li><a href="/epochs">/epochs</a></li>
  </ul>

  <script>
    async function loadEpochs() {
      const r = await fetch('/epochs');
      const data = await r.json();
      const epochs = data.epochs || [];
      if (!epochs.length) {
        document.getElementById('epoch-table').innerHTML =
          '<em>No epochs recorded yet.</em>';
        return;
      }
      let html = '<table><thead><tr><th>Epoch</th><th>Mode</th>' +
        '<th>Total Return</th><th>Sharpe</th><th>Drawdown</th>' +
        '<th>Eval Score</th><th>Passed</th></tr></thead><tbody>';
      for (const e of epochs) {
        const m = e.metrics || {};
        const passed = e.eval_passed === true;
        const badge = passed
          ? '<span class="badge-pass">&#x2713; pass</span>'
          : '<span class="badge-fail">&#x2717; fail</span>';
        html += `<tr>
          <td>${e.epoch_num}</td>
          <td>${e.mode || '—'}</td>
          <td>${fmt(m.total_return)}</td>
          <td>${fmt(m.sharpe_ratio)}</td>
          <td>${fmt(m.max_drawdown)}</td>
          <td>${e.eval_score !== undefined ? e.eval_score.toFixed(2) : '—'}</td>
          <td>${badge}</td>
        </tr>`;
      }
      html += '</tbody></table>';
      document.getElementById('epoch-table').innerHTML = html;
    }

    function fmt(v) {
      return v !== undefined ? Number(v).toFixed(4) : '—';
    }

    async function triggerRun() {
      const n = parseInt(document.getElementById('num-epochs').value, 10) || 3;
      document.getElementById('run-status').textContent = 'Starting run…';
      const r = await fetch('/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_epochs: n }),
      });
      const data = await r.json();
      document.getElementById('run-status').textContent =
        data.message || JSON.stringify(data);
      loadEpochs();
    }

    loadEpochs();
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard() -> Any:
    return _DASHBOARD_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/health")
def health() -> Any:
    return jsonify({"status": "ok", "mode": "research"})


@app.route("/config/system")
def config_system() -> Any:
    return jsonify(settings.system)


@app.route("/config/runtime")
def config_runtime() -> Any:
    return jsonify(settings.runtime)


@app.route("/config/secrets")
def config_secrets() -> Any:
    """Return only boolean presence flags – never the actual secret values."""
    return jsonify(
        {
            "openrouter_api_key": bool(settings.openrouter_api_key),
            "openai_api_key": bool(settings.openai_api_key),
        }
    )


@app.route("/epochs")
def list_epochs() -> Any:
    """List all epoch artifact directories with their manifest + evaluation."""
    ed = _epochs_dir()
    if not ed.exists():
        return jsonify({"epochs": []})

    epochs = []
    for epoch_dir in sorted(ed.iterdir()):
        if not epoch_dir.is_dir():
            continue
        manifest = _read_json(epoch_dir / "manifest.json") or {}
        bt = _read_json(epoch_dir / "backtest_result.json") or {}
        ev = _read_json(epoch_dir / "evaluation.json") or {}
        epochs.append(
            {
                "epoch_num": manifest.get("epoch_num", epoch_dir.name),
                "timestamp": manifest.get("timestamp"),
                "mode": manifest.get("mode"),
                "metrics": bt.get("metrics", {}),
                "eval_passed": ev.get("passed"),
                "eval_score": ev.get("score"),
                "eval_feedback": ev.get("feedback"),
                "artifact_dir": str(epoch_dir),
            }
        )
    return jsonify({"epochs": epochs})


@app.route("/epochs/<epoch_id>")
def get_epoch(epoch_id: str) -> Any:
    """Return full artifact data for a specific epoch."""
    # Validate epoch_id to prevent path traversal
    if not _EPOCH_ID_RE.match(epoch_id):
        return jsonify({"error": "Invalid epoch id"}), 400
    epoch_dir = _epochs_dir() / epoch_id
    if not epoch_dir.exists():
        return jsonify({"error": f"Epoch {epoch_id!r} not found"}), 404

    return jsonify(
        {
            "manifest": _read_json(epoch_dir / "manifest.json"),
            "backtest_result": _read_json(epoch_dir / "backtest_result.json"),
            "evaluation": _read_json(epoch_dir / "evaluation.json"),
            "evaluator_state": _read_json(epoch_dir / "evaluator_state.json"),
        }
    )


@app.route("/run", methods=["POST"])
def run_epochs() -> Any:
    """Trigger a multi-epoch run in a background thread."""
    body: Dict[str, Any] = request.get_json(silent=True) or {}
    num_epochs = int(body.get("num_epochs", 3))

    def _do_run() -> None:
        try:
            from agents.epocher import Epocher
            from agents.evaluator import Evaluator
            from agents.memory import AgentMemory
            from agents.mcp_hub import MCPHub
            from agents.overseer import Overseer
            from backtest.runner import BacktestRunner
            from epoch_runner import EpochRunner

            sys_cfg = settings.system
            redis_url = sys_cfg.get("memory", {}).get("redis_url") or None

            hub = MCPHub(system_config=sys_cfg)
            mem = AgentMemory(namespace="server_run", redis_url=redis_url, mcp_hub=hub)
            mem.load_raptor_state()

            epoch_cfg = sys_cfg.get("epochs", {})
            overseer = Overseer(mcp_hub=hub, memory=mem, config=epoch_cfg)
            epocher = Epocher(mcp_hub=hub, memory=mem, config=epoch_cfg)
            evaluator = Evaluator(mcp_hub=hub, memory=mem)
            bt_runner = BacktestRunner(config=epoch_cfg)

            runner = EpochRunner(
                overseer=overseer,
                epocher=epocher,
                evaluator=evaluator,
                memory=mem,
                backtest_runner=bt_runner,
                io_dir=settings.get_io_dir(),
            )
            runner.run(num_epochs=num_epochs)
        except Exception as exc:
            logger.error("Background run failed: %s", exc, exc_info=True)

    thread = threading.Thread(target=_do_run, daemon=True)
    thread.start()

    return jsonify(
        {
            "message": f"Started {num_epochs} epoch(s) in background.",
            "num_epochs": num_epochs,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

