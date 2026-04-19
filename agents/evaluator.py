"""Evaluator agent (DSPy-inspired, self-managing).

Evaluates epoch results against the Overseer's policy and updates
its own rubric/state based on accumulated feedback.  State is persisted
via AgentMemory so the evaluator survives process restarts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Output of the Evaluator for a single epoch."""

    epoch_num: int
    passed: bool
    score: float  # 0.0 – 1.0
    metric_scores: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    rubric_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Evaluator:
    """Self-managing evaluator that persists its rubric/state across epochs.

    Inspired by DSPy's self-optimising prompt management:

    - Maintains a rubric string used as evaluation instructions.
    - Scores epoch results and records pass/fail history.
    - Automatically rewrites the rubric every *update_every* evaluations
      using the MCPHub when available.
    - Persists full state to AgentMemory.

    Parameters
    ----------
    mcp_hub:
        MCPHub instance for LLM evaluation and rubric self-update (optional).
    memory:
        AgentMemory instance for state persistence (optional).
    update_every:
        Number of evaluations between automatic rubric updates.
    """

    _STATE_KEY = "evaluator:state"
    _DEFAULT_RUBRIC = (
        "Evaluate the epoch result against the target metrics. "
        "Score 0.0–1.0. Consider: return vs target, drawdown, Sharpe ratio. "
        "Be strict but fair. Provide concise feedback."
    )

    def __init__(
        self,
        mcp_hub: Optional[Any] = None,
        memory: Optional[Any] = None,
        update_every: int = 5,
    ) -> None:
        self._hub = mcp_hub
        self._memory = memory
        self._update_every = update_every
        self._rubric: str = self._DEFAULT_RUBRIC
        self._rubric_version: int = 1
        self._history: List[Dict[str, Any]] = []
        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, epoch_result: Any, policy: Any) -> EvaluationResult:
        """Evaluate an epoch result against its policy.

        Parameters
        ----------
        epoch_result:
            EpochResult from the Epocher.
        policy:
            EpochPolicy from the Overseer.

        Returns
        -------
        EvaluationResult
        """
        if self._hub is not None:
            result = self._llm_evaluate(epoch_result, policy)
        else:
            result = self._heuristic_evaluate(epoch_result, policy)

        self._history.append(result.to_dict())
        self._maybe_update_rubric()
        self._save_state()
        return result

    def get_state(self) -> Dict[str, Any]:
        """Return the evaluator's current state dict."""
        return {
            "rubric": self._rubric,
            "rubric_version": self._rubric_version,
            "history_length": len(self._history),
        }

    # ------------------------------------------------------------------
    # Evaluation implementations
    # ------------------------------------------------------------------

    def _heuristic_evaluate(
        self, epoch_result: Any, policy: Any
    ) -> EvaluationResult:
        """Evaluate using simple metric threshold comparisons."""
        target = policy.target_metrics
        metrics: Dict[str, float] = getattr(epoch_result, "metrics", {}) or {}
        scores: Dict[str, float] = {}

        for metric, target_val in target.items():
            actual = metrics.get(metric)
            if actual is None:
                scores[metric] = 0.5  # unknown → neutral
            elif metric in ("max_drawdown",):
                # Lower is better
                scores[metric] = min(
                    1.0, target_val / max(abs(actual), 1e-9)
                )
            else:
                # Higher is better
                scores[metric] = (
                    min(1.0, actual / target_val) if target_val != 0 else 1.0
                )

        overall = sum(scores.values()) / max(len(scores), 1)
        passed = overall >= 0.6
        feedback = (
            f"Heuristic evaluation: score={overall:.2f}. "
            + ("Passed." if passed else "Failed – metrics below threshold.")
        )
        return EvaluationResult(
            epoch_num=epoch_result.epoch_num,
            passed=passed,
            score=round(overall, 4),
            metric_scores=scores,
            feedback=feedback,
            rubric_version=self._rubric_version,
        )

    def _llm_evaluate(
        self, epoch_result: Any, policy: Any
    ) -> EvaluationResult:
        """Evaluate using LLM with the current rubric."""
        prompt = (
            f"Rubric: {self._rubric}\n\n"
            f"Policy target metrics: {json.dumps(policy.target_metrics)}\n"
            f"Epoch result metrics: "
            f"{json.dumps(getattr(epoch_result, 'metrics', {}))}\n"
            f"Mode: {getattr(epoch_result, 'mode', 'unknown')}\n"
            f"Error: {getattr(epoch_result, 'error', None)}\n\n"
            "Return JSON with keys: passed (bool), score (float 0-1), "
            "metric_scores (dict), feedback (string)."
        )
        raw = self._hub.call_json([{"role": "user", "content": prompt}])
        # If only "raw" key is present, JSON parsing failed → use heuristic
        if set(raw.keys()) == {"raw"}:
            logger.debug("Evaluator LLM returned non-JSON; using heuristic.")
            return self._heuristic_evaluate(epoch_result, policy)
        try:
            return EvaluationResult(
                epoch_num=epoch_result.epoch_num,
                passed=bool(raw.get("passed", False)),
                score=float(raw.get("score", 0.0)),
                metric_scores=dict(raw.get("metric_scores", {})),
                feedback=str(raw.get("feedback", "")),
                rubric_version=self._rubric_version,
            )
        except Exception as exc:
            logger.warning("Evaluator LLM parse failed (%s); using heuristic.", exc)
            return self._heuristic_evaluate(epoch_result, policy)

    # ------------------------------------------------------------------
    # Rubric self-update
    # ------------------------------------------------------------------

    def _maybe_update_rubric(self) -> None:
        """Update rubric every *update_every* evaluations (LLM required)."""
        if (
            self._hub is None
            or len(self._history) == 0
            or len(self._history) % self._update_every != 0
        ):
            return

        recent = self._history[-self._update_every :]
        summary = "\n".join(
            f"Epoch {h['epoch_num']}: passed={h['passed']}, "
            f"score={h['score']:.2f}, feedback={h['feedback'][:80]}"
            for h in recent
        )
        update_prompt = (
            "You are an evaluator improving your own rubric.\n"
            f"Current rubric:\n{self._rubric}\n\n"
            f"Recent evaluations:\n{summary}\n\n"
            "Rewrite the rubric to be more accurate and useful. "
            "Return only the new rubric text, no prose."
        )
        try:
            new_rubric = self._hub.call(
                [{"role": "user", "content": update_prompt}],
                system_prompt="You are a self-improving evaluator. Be concise.",
            )
            if new_rubric and not new_rubric.startswith("[MCPHub error"):
                self._rubric = new_rubric.strip()
                self._rubric_version += 1
                logger.info(
                    "Evaluator rubric updated to v%d.", self._rubric_version
                )
        except Exception as exc:
            logger.warning("Rubric self-update failed: %s", exc)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        if self._memory is None:
            return
        state = {
            "rubric": self._rubric,
            "rubric_version": self._rubric_version,
            "history": self._history[-20:],  # keep last 20 for brevity
        }
        self._memory.store(self._STATE_KEY, state)

    def _load_state(self) -> None:
        if self._memory is None:
            return
        state = self._memory.retrieve(self._STATE_KEY)
        if state:
            self._rubric = state.get("rubric", self._DEFAULT_RUBRIC)
            self._rubric_version = state.get("rubric_version", 1)
            self._history = state.get("history", [])
            logger.info(
                "Evaluator loaded state: rubric v%d, %d history items.",
                self._rubric_version,
                len(self._history),
            )
