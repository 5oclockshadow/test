"""Overseer agent.

Decides each epoch's data policy, tone, rules, todos, and target metrics.
Can use an MCPHub for LLM-driven decisions or operate heuristically when
no hub is available.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EpochPolicy:
    """Policy decided by the Overseer for a single epoch."""

    epoch_num: int
    data_policy: str = "cached"  # "live" | "cached"
    tone: str = "analytical"
    rules: List[str] = field(default_factory=list)
    todos: List[str] = field(default_factory=list)
    target_metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Overseer:
    """Overseer agent that decides epoch policies.

    Maintains awareness of past epoch results to inform future decisions.
    Uses MCPHub for LLM-driven policy generation when available; otherwise
    falls back to a deterministic heuristic.

    Parameters
    ----------
    mcp_hub:
        MCPHub instance for LLM-driven decisions (optional).
    memory:
        AgentMemory instance for context retrieval (optional).
    config:
        Dict of runtime/system configuration.
    """

    def __init__(
        self,
        mcp_hub: Optional[Any] = None,
        memory: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._hub = mcp_hub
        self._memory = memory
        self._config = config or {}

    def decide_epoch_policy(
        self,
        epoch_num: int,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> EpochPolicy:
        """Decide policy for the given epoch number.

        Parameters
        ----------
        epoch_num:
            0-indexed epoch number.
        history:
            List of previous epoch summary dicts.

        Returns
        -------
        EpochPolicy
        """
        history = history or []
        if self._hub is not None:
            return self._llm_decide(epoch_num, history)
        return self._heuristic_decide(epoch_num, history)

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _heuristic_decide(
        self, epoch_num: int, history: List[Dict[str, Any]]
    ) -> EpochPolicy:
        """Deterministic heuristic policy (no LLM required)."""
        data_policy = "live" if epoch_num % 2 == 0 else "cached"
        target_return = round(0.02 * (epoch_num + 1), 4)
        return EpochPolicy(
            epoch_num=epoch_num,
            data_policy=data_policy,
            tone="analytical",
            rules=["minimize_drawdown", "respect_position_limits"],
            todos=[f"tune_epoch_{epoch_num}_parameters"],
            target_metrics={
                "min_return": target_return,
                "max_drawdown": 0.05,
                "sharpe_ratio": 1.0,
            },
            notes=f"Heuristic policy for epoch {epoch_num}",
        )

    # ------------------------------------------------------------------
    # LLM-driven decision
    # ------------------------------------------------------------------

    def _llm_decide(
        self, epoch_num: int, history: List[Dict[str, Any]]
    ) -> EpochPolicy:
        """LLM-driven policy decision via MCPHub."""
        history_summary = (
            "\n".join(
                f"Epoch {h.get('epoch_num', '?')}: "
                f"return={h.get('metrics', {}).get('total_return', 'N/A')}, "
                f"sharpe={h.get('metrics', {}).get('sharpe_ratio', 'N/A')}"
                for h in history[-5:]
            )
            or "No prior history."
        )
        prompt = (
            f"You are the Overseer of a multi-epoch trading research loop.\n"
            f"Epoch number: {epoch_num}\n"
            f"Recent history:\n{history_summary}\n\n"
            f"Decide the policy for this epoch. Return JSON with keys:\n"
            f'  data_policy ("live" or "cached"),\n'
            f"  tone (string),\n"
            f"  rules (list of strings),\n"
            f"  todos (list of strings),\n"
            f"  target_metrics (dict of metric_name -> float),\n"
            f"  notes (string).\n"
            f"Respond with valid JSON only."
        )
        result = self._hub.call_json([{"role": "user", "content": prompt}])
        # If only "raw" key present, JSON parsing failed → use heuristic
        if set(result.keys()) == {"raw"}:
            logger.debug("Overseer LLM returned non-JSON; using heuristic.")
            return self._heuristic_decide(epoch_num, history)
        try:
            return EpochPolicy(
                epoch_num=epoch_num,
                data_policy=str(result.get("data_policy", "cached")),
                tone=str(result.get("tone", "analytical")),
                rules=list(result.get("rules", [])),
                todos=list(result.get("todos", [])),
                target_metrics=dict(result.get("target_metrics", {})),
                notes=str(result.get("notes", "")),
            )
        except Exception as exc:
            logger.warning("Overseer LLM parse failed (%s); using heuristic.", exc)
            return self._heuristic_decide(epoch_num, history)
