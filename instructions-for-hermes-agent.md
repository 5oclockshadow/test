# Instructions for Hermes Agent: create and bootstrap `mission-executor`

These are the exact steps to create a new private GitHub repository and populate it with a non-mock, ready-to-run starter structure.

## Goal

Create a new private repo:

- Owner: `5oclockshadow`
- Repo: `mission-executor`

Then add:

- `README.md`
- `docs/architecture.md`
- `docs/implementation-order.md`
- `src/mission_executor/__init__.py`
- `src/mission_executor/models.py`
- `src/mission_executor/executor.py`
- `pyproject.toml`

## Auth requirements

### If using a classic PAT
Use a token with:

- `repo`

### If using GitHub CLI
Authenticate first if needed:

```bash
gh auth login
```

Or if a PAT is already available:

```bash
export GITHUB_TOKEN='YOUR_PAT'
gh auth status
```

---

## Step 1: create the repository

Run:

```bash
gh repo create 5oclockshadow/mission-executor \
  --private \
  --description "Production-oriented mission runtime for autonomous agent workflows" \
  --disable-wiki \
  --disable-issues \
  --clone=false
```

If you prefer interactive mode:

```bash
gh repo create
```

---

## Step 2: clone the repo locally

```bash
git clone https://github.com/5oclockshadow/mission-executor.git
cd mission-executor
```

---

## Step 3: create the directory structure

```bash
mkdir -p docs
mkdir -p src/mission_executor
```

---

## Step 4: write the files

### `README.md`

```md
# Mission Executor

A production-oriented mission runtime for autonomous agent workflows, inspired by the execution discipline discussed for Hermes + osiiso.

## Recommendation order

If your goal is autonomy + lean code, do this in order:

1. **Refactor `batch_runner.py` first**
   - easiest non-interactive proving ground
   - high leverage
   - minimal UI complexity
2. **Refactor cron mission mode second**
   - make schedules create/resume missions
   - not raw prompt invocations
3. **Hook both into persistent mission state third**
   - likely in or near `hermes_state.py`

## What this repo contains

This repo is intended to hold a non-mock, ready-to-run mission execution layer:

- `README.md` — this file
- `docs/architecture.md` — architecture and rollout plan
- `docs/implementation-order.md` — concrete phased plan

## Core design goal

Turn one-shot agent runs into durable missions with:

- queue-backed execution
- retries and backoff
- resumable scheduled runs
- persistent mission state
- clean separation between orchestration and agent logic

## Proposed core abstractions

- **Mission**
  - objective
  - phase
  - priority
  - schedule
  - retry policy
- **MissionStep**
  - executable unit of work
  - status
  - timeout
  - dependency metadata
- **MissionExecutor**
  - submit
  - resume
  - cancel
  - retry
  - group/fan-out
- **MissionStateStore**
  - queued / running / blocked / failed / complete
  - next wake-up
  - last error
  - artifacts

## Why batch first

`batch_runner.py` is the best proving ground because it already maps well to a mission runtime:

- worker parallelism
- checkpointing
- resumability
- result aggregation
- non-interactive operation

## Why cron second

Cron is the path from “tool-capable agent” to “autonomous operator.”
Instead of running raw prompts on a schedule, schedules should create or resume mission records.

## Why persistence third

Once batch and cron share the same runtime, mission state becomes the durable backbone that makes autonomy reliable.
```

---

### `docs/architecture.md`

```md
# Architecture

## Purpose

Mission Executor is a small, production-oriented runtime for autonomous workflows.

It is designed to support:

- mission submission
- mission resumption
- grouped execution
- retries and backoff
- scheduled wakeups
- persistent state

## Core abstractions

### Mission
A durable top-level unit of work.

Fields:
- id
- objective
- status
- metadata
- steps

### MissionStep
A single executable step within a mission.

Fields:
- id
- name
- payload
- status
- retry_count

### MissionExecutor
Execution control plane.

Responsibilities:
- submit mission
- start mission
- complete mission
- fail mission
- later: retry, cancel, fan-out, schedule

### MissionStateStore
Persistence interface for mission lifecycle state.

Future backends:
- SQLite
- Postgres
- file-backed JSON for local testing

## Recommended rollout

1. Prove the model in non-interactive batch execution
2. Reuse the same runtime for cron-driven scheduled missions
3. Persist mission state so runs can resume across process restarts
```

---

### `docs/implementation-order.md`

```md
# Implementation Order

## 1. Batch first

Refactor a `batch_runner.py`-style execution path first.

Why:
- easiest proving ground
- no UI complexity
- already maps to queue + retry + result aggregation

## 2. Cron second

Move scheduled work from “run raw prompt” to “create/resume mission”.

Why:
- makes autonomy durable
- separates wakeup from execution
- enables deferred retries and resumptions

## 3. Persistence third

Add durable mission state.

Why:
- enables resume after failure
- supports blocked/retry states
- makes scheduled autonomy reliable

## 4. Expand executor semantics

After the basic runtime works:
- cancellation
- deadlines
- grouped steps
- priority handling
- retry policy per step
```

---

### `src/mission_executor/__init__.py`

```python
from .models import Mission, MissionStep, MissionStatus
from .executor import MissionExecutor

__all__ = [
    "Mission",
    "MissionStep",
    "MissionStatus",
    "MissionExecutor",
]
```

---

### `src/mission_executor/models.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MissionStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class MissionStep:
    id: str
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: MissionStatus = MissionStatus.QUEUED
    retry_count: int = 0


@dataclass
class Mission:
    id: str
    objective: str
    steps: list[MissionStep] = field(default_factory=list)
    status: MissionStatus = MissionStatus.QUEUED
    metadata: dict[str, Any] = field(default_factory=dict)
```

---

### `src/mission_executor/executor.py`

```python
from __future__ import annotations

from .models import Mission, MissionStatus


class MissionExecutor:
    def submit(self, mission: Mission) -> Mission:
        mission.status = MissionStatus.QUEUED
        return mission

    def start(self, mission: Mission) -> Mission:
        mission.status = MissionStatus.RUNNING
        return mission

    def complete(self, mission: Mission) -> Mission:
        mission.status = MissionStatus.COMPLETED
        return mission

    def fail(self, mission: Mission) -> Mission:
        mission.status = MissionStatus.FAILED
        return mission
```

---

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mission-executor"
version = "0.1.0"
description = "Production-oriented mission runtime for autonomous agent workflows"
readme = "README.md"
requires-python = ">=3.11"
dependencies = []

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["mission_executor*"]
```

---

## Step 5: commit and push

```bash
git add .
git commit -m "Bootstrap mission-executor runtime"
git push origin HEAD
```

---

## Success criteria

After completion, the repo should:

- exist as `5oclockshadow/mission-executor`
- be private
- contain the listed files
- install as a Python package with `pip install -e .`
