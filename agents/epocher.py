"""Epocher agent.

Executes a single epoch with two modes:
  - live:     step through a Gymnasium trading environment.
  - backtest: run (or generate-then-execute) a backtest workflow.

DEAP-based parameter enrichment is performed before each epoch when the
``deap`` package is available.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import gymnasium  # type: ignore  # noqa: F401
    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GYM_AVAILABLE = False

try:
    from deap import algorithms as _deap_algos  # type: ignore
    from deap import base as _deap_base  # type: ignore
    from deap import creator as _deap_creator  # type: ignore
    from deap import tools as _deap_tools  # type: ignore
    _DEAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DEAP_AVAILABLE = False


@dataclass
class EpochResult:
    """Result produced by the Epocher for a single epoch."""

    epoch_num: int
    mode: str  # "live" | "backtest"
    observations: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    generated_code: Optional[str] = None
    code_output: Optional[str] = None
    deap_best_params: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Epocher:
    """Executes epochs in either live (Gym env) or backtest (code/runner) mode.

    Parameters
    ----------
    mcp_hub:
        MCPHub instance for LLM-generated backtest code (optional).
    memory:
        AgentMemory instance (optional).
    config:
        Runtime configuration dict.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_epoch(
        self,
        policy: Any,
        env: Optional[Any] = None,
        backtest_runner: Optional[Any] = None,
    ) -> EpochResult:
        """Run one epoch according to the given EpochPolicy.

        Parameters
        ----------
        policy:
            EpochPolicy from the Overseer.
        env:
            Optional Gymnasium environment instance (used in live mode).
        backtest_runner:
            Optional BacktestRunner instance (used in backtest mode).

        Returns
        -------
        EpochResult
        """
        # DEAP enrichment: evolve strategy parameters before running
        best_params = self._deap_enrich(policy)

        mode = policy.data_policy  # "live" or "cached"
        if mode == "live" and env is not None and _GYM_AVAILABLE:
            return self._run_live(policy, env, best_params)
        return self._run_backtest(policy, backtest_runner, best_params)

    # ------------------------------------------------------------------
    # Live mode
    # ------------------------------------------------------------------

    def _run_live(
        self,
        policy: Any,
        env: Any,
        best_params: Optional[Dict[str, Any]],
    ) -> EpochResult:
        """Step through the Gymnasium environment for one episode."""
        observations: List[Dict[str, Any]] = []
        total_reward = 0.0
        steps = 0
        max_steps = int(self._config.get("max_steps_per_epoch", 500))

        try:
            obs, _info = env.reset()
            terminated = truncated = False
            while not (terminated or truncated) and steps < max_steps:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                observations.append(
                    {
                        "step": steps,
                        "obs": obs.tolist() if hasattr(obs, "tolist") else obs,
                        "reward": float(reward),
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )

            metrics: Dict[str, float] = {
                "total_reward": round(total_reward, 6),
                "steps": float(steps),
                "avg_reward": round(total_reward / max(steps, 1), 6),
            }
            return EpochResult(
                epoch_num=policy.epoch_num,
                mode="live",
                observations=observations,
                metrics=metrics,
                deap_best_params=best_params,
            )
        except Exception as exc:
            logger.error("Epocher._run_live error: %s", exc)
            return EpochResult(
                epoch_num=policy.epoch_num,
                mode="live",
                error=str(exc),
                deap_best_params=best_params,
            )

    # ------------------------------------------------------------------
    # Backtest mode
    # ------------------------------------------------------------------

    def _run_backtest(
        self,
        policy: Any,
        backtest_runner: Optional[Any],
        best_params: Optional[Dict[str, Any]],
    ) -> EpochResult:
        """Run a backtest workflow, with optional LLM-generated code."""
        if backtest_runner is not None:
            try:
                result = backtest_runner.run(
                    policy=policy.to_dict(), params=best_params
                )
                return EpochResult(
                    epoch_num=policy.epoch_num,
                    mode="backtest",
                    metrics=result.get("metrics", {}),
                    deap_best_params=best_params,
                )
            except Exception as exc:
                logger.error("BacktestRunner.run error: %s", exc)
                return EpochResult(
                    epoch_num=policy.epoch_num,
                    mode="backtest",
                    error=str(exc),
                    deap_best_params=best_params,
                )

        # Generate and execute backtest code via LLM
        if self._hub is not None:
            return self._run_generated_backtest(policy, best_params)

        # Fallback: synthetic placeholder metrics
        return EpochResult(
            epoch_num=policy.epoch_num,
            mode="backtest",
            metrics=self._synthetic_metrics(),
            deap_best_params=best_params,
        )

    def _run_generated_backtest(
        self,
        policy: Any,
        best_params: Optional[Dict[str, Any]],
    ) -> EpochResult:
        """Generate backtest code via LLM and execute it in a subprocess."""
        code = self._generate_backtest_code(policy, best_params)
        output, error = self._execute_code_safely(code)
        metrics = self._parse_metrics_from_output(output)
        return EpochResult(
            epoch_num=policy.epoch_num,
            mode="backtest",
            metrics=metrics,
            generated_code=code,
            code_output=output,
            error=error,
            deap_best_params=best_params,
        )

    def _generate_backtest_code(
        self,
        policy: Any,
        best_params: Optional[Dict[str, Any]],
    ) -> str:
        """Ask the LLM to generate a self-contained backtest script."""
        prompt = (
            "Write a self-contained Python backtest script for these settings:\n"
            f"Policy: {policy.to_dict()}\n"
            f"Params: {best_params or {}}\n"
            "The script must:\n"
            "  1. Generate or simulate price data (no network calls).\n"
            "  2. Run a simple buy/hold or threshold strategy.\n"
            '  3. Print metrics as JSON on the last line: METRICS: {"total_return": 0.05}\n'
            "Use only the standard library plus numpy if available."
        )
        if self._hub is not None:
            code = self._hub.call(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are a quant developer. Return only Python code, no prose."
                ),
            )
            code = code.strip()
            for fence in ("```python", "```"):
                if code.startswith(fence):
                    code = code[len(fence):]
            if code.endswith("```"):
                code = code[:-3]
            return code.strip()

        # Minimal fallback script
        return textwrap.dedent(
            """\
            import json, random, math
            rng = random.Random(42)
            prices = [100.0]
            for _ in range(251):
                prices.append(prices[-1] * math.exp(rng.gauss(0.0002, 0.015)))
            returns = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
            total_return = prices[-1] / prices[0] - 1
            avg_r = sum(returns) / len(returns)
            std_r = math.sqrt(sum((r - avg_r) ** 2 for r in returns) / len(returns)) or 1e-9
            sharpe = (avg_r / std_r) * math.sqrt(252)
            print("METRICS:", json.dumps({
                "total_return": round(total_return, 6),
                "sharpe_ratio": round(sharpe, 4),
            }))
            """
        )

    def _execute_code_safely(self, code: str) -> Tuple[str, Optional[str]]:
        """Execute Python code in a subprocess with a 30 s timeout."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = proc.stdout.strip()
            err = proc.stderr.strip() if proc.returncode != 0 else None
            return output, err
        except subprocess.TimeoutExpired:
            return "", "Code execution timed out (30 s)"
        except Exception as exc:
            return "", str(exc)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _parse_metrics_from_output(output: str) -> Dict[str, float]:
        """Parse ``METRICS: {...}`` from subprocess stdout."""
        import json as _json

        for line in reversed(output.splitlines()):
            if line.startswith("METRICS:"):
                try:
                    return _json.loads(line[len("METRICS:"):].strip())
                except Exception as exc:
                    logger.debug(
                        "Failed to parse METRICS line %r: %s", line[:120], exc
                    )
        return {}

    @staticmethod
    def _synthetic_metrics() -> Dict[str, float]:
        """Return placeholder metrics when no runner/LLM is available."""
        import random as _random

        rng = _random.Random()
        return {
            "total_return": round(rng.uniform(-0.05, 0.10), 6),
            "sharpe_ratio": round(rng.uniform(0.5, 2.0), 4),
            "max_drawdown": round(rng.uniform(0.01, 0.10), 6),
        }

    # ------------------------------------------------------------------
    # DEAP enrichment
    # ------------------------------------------------------------------

    def _deap_enrich(self, policy: Any) -> Optional[Dict[str, Any]]:
        """Use DEAP to evolve strategy parameters if available."""
        if not _DEAP_AVAILABLE:
            return None
        try:
            return self._run_deap(policy)
        except Exception as exc:
            logger.warning("DEAP enrichment failed: %s", exc)
            return None

    @staticmethod
    def _run_deap(policy: Any) -> Dict[str, Any]:
        """Run a simple genetic algorithm to find good strategy params."""
        import random as _random

        # Avoid duplicate creator registration across multiple epoch runs
        if not hasattr(_deap_creator, "FitnessMax_Epocher"):
            _deap_creator.create(
                "FitnessMax_Epocher", _deap_base.Fitness, weights=(1.0,)
            )
        if not hasattr(_deap_creator, "Individual_Epocher"):
            _deap_creator.create(
                "Individual_Epocher",
                list,
                fitness=_deap_creator.FitnessMax_Epocher,
            )

        toolbox = _deap_base.Toolbox()

        # Generate thresholds in realistic momentum range [0.001, 0.02]
        # and position_size in [0.5, 1.0]
        def _rand_threshold() -> float:
            return _random.uniform(0.001, 0.02)

        def _rand_position() -> float:
            return _random.uniform(0.5, 1.0)

        def _make_individual() -> Any:
            return _deap_creator.Individual_Epocher(
                [_rand_threshold(), _rand_threshold(), _rand_position()]
            )

        toolbox.register("individual", _make_individual)
        toolbox.register(
            "population", _deap_tools.initRepeat, list, toolbox.individual
        )

        def evaluate(ind: list) -> Tuple[float, ...]:
            buy_threshold, sell_threshold, position_size = ind
            score = (buy_threshold - sell_threshold) * max(position_size, 0.0)
            return (score,)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", _deap_tools.cxTwoPoint)
        toolbox.register(
            "mutate", _deap_tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2
        )
        toolbox.register("select", _deap_tools.selTournament, tournsize=3)

        pop = toolbox.population(n=20)
        _deap_algos.eaSimple(
            pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, verbose=False
        )
        best = _deap_tools.selBest(pop, k=1)[0]
        return {
            "buy_threshold": round(best[0], 4),
            "sell_threshold": round(best[1], 4),
            "position_size": round(best[2], 4),
        }
