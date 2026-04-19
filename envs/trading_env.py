"""Gymnasium-compatible trading environment.

Simulates a single-asset trading scenario using Geometric Brownian Motion
price data.

Observation space (4 float32 values):
    - position:      normalised current position in [-1, 1]
    - cash_ratio:    fraction of initial capital held as cash in [0, 1]
    - price_change:  most recent log-return (clipped to [-1, 1])
    - volatility:    rolling std of the last 10 log-returns (clipped to [0, 1])

Action space:
    - 0: hold
    - 1: buy  (use ~95 % of available cash at current price)
    - 2: sell (liquidate entire position at current price)
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GYM_AVAILABLE = False
    gym = None  # type: ignore
    spaces = None  # type: ignore


if _GYM_AVAILABLE:

    class TradingEnv(gym.Env):
        """Minimal Gymnasium trading environment.

        Parameters
        ----------
        initial_price:
            Starting asset price.
        initial_cash:
            Starting cash balance.
        fee_bps:
            Round-trip trading fee in basis points.
        max_steps:
            Maximum episode length in steps.
        seed:
            Optional RNG seed for reproducibility.
        """

        metadata = {"render_modes": ["human"]}

        def __init__(
            self,
            initial_price: float = 100.0,
            initial_cash: float = 10_000.0,
            fee_bps: float = 10.0,
            max_steps: int = 252,
            seed: Optional[int] = None,
        ) -> None:
            super().__init__()
            self.initial_price = initial_price
            self.initial_cash = initial_cash
            self.fee_bps = fee_bps
            self.max_steps = max_steps

            self.observation_space = spaces.Box(
                low=np.array([-1.0, 0.0, -1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(3)

            self._rng = random.Random(seed)
            self._price: float = initial_price
            self._cash: float = initial_cash
            self._position: float = 0.0
            self._step_count: int = 0
            self._log_returns: list = [0.0] * 10

        # ------------------------------------------------------------------
        # Gymnasium interface
        # ------------------------------------------------------------------

        def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
            super().reset(seed=seed)
            if seed is not None:
                self._rng = random.Random(seed)
            self._price = self.initial_price
            self._cash = self.initial_cash
            self._position = 0.0
            self._step_count = 0
            self._log_returns = [0.0] * 10
            return self._get_obs(), {}

        def step(
            self, action: int
        ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
            prev_value = self._portfolio_value()

            # GBM price step
            log_ret = 0.0001 + 0.01 * self._rng.gauss(0, 1)
            self._price *= math.exp(log_ret)
            self._log_returns.append(log_ret)
            if len(self._log_returns) > 10:
                self._log_returns.pop(0)

            fee = self.fee_bps / 10_000.0
            if action == 1 and self._cash > 0:  # buy
                shares = (self._cash * 0.95) / (self._price * (1 + fee))
                cost = shares * self._price * (1 + fee)
                if cost <= self._cash:
                    self._position += shares
                    self._cash -= cost
            elif action == 2 and self._position > 0:  # sell
                proceeds = self._position * self._price * (1 - fee)
                self._cash += proceeds
                self._position = 0.0

            self._step_count += 1
            new_value = self._portfolio_value()
            reward = (new_value - prev_value) / max(prev_value, 1e-9)
            terminated = self._step_count >= self.max_steps

            info: Dict[str, Any] = {
                "price": self._price,
                "portfolio_value": new_value,
                "step": self._step_count,
            }
            return self._get_obs(), reward, terminated, False, info

        def render(self) -> None:
            logger.info(
                "Step %d | Price %.2f | Portfolio %.2f | Cash %.2f | Pos %.4f",
                self._step_count,
                self._price,
                self._portfolio_value(),
                self._cash,
                self._position,
            )

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _portfolio_value(self) -> float:
            return self._cash + self._position * self._price

        def _get_obs(self) -> np.ndarray:
            max_pos = max(
                self.initial_cash / max(self.initial_price, 1e-9), 1e-9
            )
            position_norm = float(
                np.clip(self._position / max_pos, -1.0, 1.0)
            )
            cash_ratio = float(
                np.clip(self._cash / self.initial_cash, 0.0, 1.0)
            )
            last_ret = float(
                np.clip(
                    self._log_returns[-1] if self._log_returns else 0.0,
                    -1.0,
                    1.0,
                )
            )
            volatility = float(
                np.clip(
                    math.sqrt(
                        sum(r ** 2 for r in self._log_returns)
                        / max(len(self._log_returns), 1)
                    ),
                    0.0,
                    1.0,
                )
            )
            return np.array(
                [position_norm, cash_ratio, last_ret, volatility],
                dtype=np.float32,
            )

else:

    class TradingEnv:  # type: ignore
        """Stub TradingEnv when gymnasium is not installed."""

        observation_space: Any = None
        action_space: Any = None

        def __init__(self, **kwargs: Any) -> None:
            logger.warning(
                "gymnasium is not installed; TradingEnv is a no-op stub."
            )

        def reset(self) -> Tuple[None, Dict[str, Any]]:
            return None, {}

        def step(
            self, action: Any
        ) -> Tuple[None, float, bool, bool, Dict[str, Any]]:
            return None, 0.0, True, False, {}

        def render(self) -> None:
            pass
