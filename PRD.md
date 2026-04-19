# Product Requirements Document (PRD)

## 1. Summary
This repository provides a Gymnasium-compatible trading environment, a backtesting workflow, and an optional LLM-driven assistant layer. The project emphasizes reproducibility, safe separation of secrets, and a clear configuration model.

## 2. Goals
- Provide a **Gymnasium trading environment** suitable for training and evaluating agents.
- Provide a **backtesting** path that is reproducible and configurable.
- Provide a **system-wide configuration layer** for non-secret defaults (LLM model/prompt, IO directories, concurrency) via `system.ini`.
- Keep **secrets out of source control** and out of runtime logs.
- Offer a minimal server surface for health/config inspection and future APIs.

## 3. Non-Goals (for now)
- No guarantee of broker/exchange execution correctness for live trading.
- No portfolio optimization suite beyond basic risk controls.
- No UI dashboard requirements.

## 4. Personas
- **Solo Researcher (Primary)**: Runs gym/backtests locally, iterates quickly, values reproducibility.
- **Quant Engineer (Secondary)**: Integrates with external datasets/providers and CI.
- **Operator (Future)**: Runs paper/live mode with strict safety controls.

## 5. Configuration Model (Source of Truth)
### 5.1 Secrets / Environment (Pydantic)
- Stored in `.secrets.env` and/or `.env` (not committed).
- Examples: `OPENROUTER_API_KEY`.
- **Rule**: secrets must never be written to logs or returned via API.

### 5.2 System-wide non-secret configuration: `system.ini`
Purpose: stable, system-wide defaults that are not part of a specific backtest/strategy run.
- LLM provider + model + base_url + base prompt
- IO folder conventions
- Concurrency / behavior defaults

### 5.3 Gym/backtest runtime configuration: `strategy.yaml`
Purpose: run-specific configuration for training/backtesting.
- strategy name, symbols, timeframe
- execution assumptions (paper trading, fees, slippage)
- risk constraints
- data provider toggles

## 6. User Stories
- As a researcher, I can run a backtest with a single `strategy.yaml` file and get deterministic outputs.
- As a researcher, I can change LLM model/prompt without touching YAML or secrets.
- As a developer, I can inspect loaded config at runtime without exposing secrets.

## 7. Functional Requirements
### 7.1 Config loading
- Load `system.ini` into `settings.system`.
- Load `strategy.yaml` into `settings.runtime`.
- Load env secrets into Settings fields.
- Provide a single `settings` object importable across modules.

### 7.2 IO paths
- Create or expect a root IO directory (default `./io`).
- Subdirectories: `./io/logs`, `./io/cache`.

### 7.3 Logging
- Central log level controlled by env (`log_level`) and/or system behavior defaults.
- Redaction rule: never log secrets.

### 7.4 Server API (initial)
- `/` returns a simple status string.
- `/health` returns basic health and mode.
- `/config/system` returns non-secret INI config.
- `/config/runtime` returns YAML runtime config.
- `/config/secrets` returns only boolean presence flags.

## 8. Security & Privacy Requirements
- `.secrets.env` must be gitignored.
- Do not return secret values via HTTP.
- Avoid writing secret values to disk in IO outputs.

## 9. Milestones
### v0 (Now)
- Establish config model (env + `system.ini` + `strategy.yaml`).
- Minimal server endpoints for inspection.
- Document usage.

### v1
- Implement core gym environment + backtest runner pipeline.
- Standardize output artifacts (metrics, trades, plots) under `io_dir`.
- Add tests for config loading and redaction.

### v2
- Add provider integrations and caching strategy.
- Add a pluggable LLM “assistant” layer for analysis/report generation.
- Add CI checks and linting.

---

## Appendix: Repository Files
- `system.ini`: system-wide defaults
- `strategy.yaml`: run-specific backtest/gym config
- `.secrets.env`: secrets (not committed)
- `config.py`: config loader / settings singleton
- `server.py`: minimal HTTP surface
