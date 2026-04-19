"""Multi-epoch training/backtesting orchestrator.

Coordinates Overseer → Epocher → Evaluator in a loop and writes canonical
artifacts to ``io/epochs/{epoch_num:04d}/`` after each epoch.

Typical usage
-------------
>>> from agents.overseer import Overseer
>>> from agents.epocher import Epocher
>>> from agents.evaluator import Evaluator
>>> from epoch_runner import EpochRunner
>>>
>>> runner = EpochRunner(
...     overseer=Overseer(),
...     epocher=Epocher(),
...     evaluator=Evaluator(),
... )
>>> summaries = runner.run(num_epochs=3)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EpochRunner:
    """Orchestrates the multi-epoch agent loop.

    Flow per epoch:
        1. ``Overseer.decide_epoch_policy()``
        2. ``Epocher.run_epoch()``
        3. ``Evaluator.evaluate()``
        4. Write canonical artifacts to ``io/epochs/{epoch_num:04d}/``
        5. Store a plain-text summary in AgentMemory for future epoch context

    Parameters
    ----------
    overseer:
        Overseer agent instance.
    epocher:
        Epocher agent instance.
    evaluator:
        Evaluator agent instance.
    memory:
        AgentMemory instance (optional).
    env:
        Gymnasium environment (optional, used when policy data_policy="live").
    backtest_runner:
        BacktestRunner instance (optional).
    io_dir:
        Root IO directory (default: ``./io``).
    """

    def __init__(
        self,
        overseer: Any,
        epocher: Any,
        evaluator: Any,
        memory: Optional[Any] = None,
        env: Optional[Any] = None,
        backtest_runner: Optional[Any] = None,
        io_dir: str = "./io",
    ) -> None:
        self._overseer = overseer
        self._epocher = epocher
        self._evaluator = evaluator
        self._memory = memory
        self._env = env
        self._backtest_runner = backtest_runner
        self._io_dir = Path(io_dir)
        self._epochs_dir = self._io_dir / "epochs"
        self._epochs_dir.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, num_epochs: int = 3) -> List[Dict[str, Any]]:
        """Run the full multi-epoch loop.

        Parameters
        ----------
        num_epochs:
            Number of epochs to execute.

        Returns
        -------
        list of epoch summary dicts.
        """
        summaries: List[Dict[str, Any]] = []
        for epoch_num in range(num_epochs):
            logger.info("=== Epoch %d / %d ===", epoch_num + 1, num_epochs)
            summary = self._run_single_epoch(epoch_num)
            summaries.append(summary)
            self._history.append(summary)

            if self._memory is not None:
                self._memory.add_memory_text(
                    f"Epoch {epoch_num}: mode={summary['mode']}, "
                    f"metrics={summary.get('metrics', {})}, "
                    f"eval_score={summary.get('eval_score', 'N/A')}"
                )

        logger.info("All %d epochs complete.", num_epochs)
        return summaries

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _run_single_epoch(self, epoch_num: int) -> Dict[str, Any]:
        """Execute a single epoch and write its artifacts."""
        start_ts = time.time()

        # 1. Overseer decides policy
        policy = self._overseer.decide_epoch_policy(epoch_num, self._history)
        logger.info(
            "Epoch %d policy: data_policy=%s tone=%s",
            epoch_num,
            policy.data_policy,
            policy.tone,
        )

        # 2. Epocher executes
        epoch_result = self._epocher.run_epoch(
            policy,
            env=self._env,
            backtest_runner=self._backtest_runner,
        )
        logger.info(
            "Epoch %d result: mode=%s metrics=%s error=%s",
            epoch_num,
            epoch_result.mode,
            epoch_result.metrics,
            epoch_result.error,
        )

        # 3. Evaluator assesses
        evaluation = self._evaluator.evaluate(epoch_result, policy)
        logger.info(
            "Epoch %d evaluation: passed=%s score=%.2f",
            epoch_num,
            evaluation.passed,
            evaluation.score,
        )

        elapsed = time.time() - start_ts

        # 4. Write canonical artifacts
        artifact_dir = self._write_artifacts(
            epoch_num, policy, epoch_result, evaluation, elapsed
        )
        logger.info("Epoch %d artifacts: %s", epoch_num, artifact_dir)

        return {
            "epoch_num": epoch_num,
            "mode": epoch_result.mode,
            "metrics": epoch_result.metrics,
            "eval_passed": evaluation.passed,
            "eval_score": evaluation.score,
            "eval_feedback": evaluation.feedback,
            "artifact_dir": str(artifact_dir),
            "elapsed_s": round(elapsed, 3),
        }

    # ------------------------------------------------------------------
    # Artifact writing
    # ------------------------------------------------------------------

    def _write_artifacts(
        self,
        epoch_num: int,
        policy: Any,
        epoch_result: Any,
        evaluation: Any,
        elapsed: float,
    ) -> Path:
        """Write canonical artifacts to ``io/epochs/{epoch_num:04d}/``."""
        epoch_dir = self._epochs_dir / f"{epoch_num:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # manifest.json
        manifest: Dict[str, Any] = {
            "epoch_num": epoch_num,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(elapsed, 3),
            "policy": policy.to_dict(),
            "mode": epoch_result.mode,
            "error": epoch_result.error,
        }
        self._write_json(epoch_dir / "manifest.json", manifest)

        # observations.jsonl (live mode)
        if epoch_result.observations:
            with open(epoch_dir / "observations.jsonl", "w") as fh:
                for obs in epoch_result.observations:
                    fh.write(json.dumps(obs) + "\n")

        # code.py (generated backtest code)
        if epoch_result.generated_code:
            (epoch_dir / "code.py").write_text(
                epoch_result.generated_code, encoding="utf-8"
            )

        # backtest_result.json
        backtest_data: Dict[str, Any] = {
            "metrics": epoch_result.metrics,
            "deap_best_params": epoch_result.deap_best_params,
        }
        if epoch_result.code_output is not None:
            backtest_data["code_output"] = epoch_result.code_output
        self._write_json(epoch_dir / "backtest_result.json", backtest_data)

        # evaluation.json
        self._write_json(epoch_dir / "evaluation.json", evaluation.to_dict())

        # evaluator_state.json
        self._write_json(
            epoch_dir / "evaluator_state.json",
            self._evaluator.get_state(),
        )

        return epoch_dir

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
