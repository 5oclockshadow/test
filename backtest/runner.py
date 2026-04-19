"""Canonical backtest runner.

Executes a configurable backtest workflow on historical or synthetic data.
Returns a dict with 'metrics' and 'trades' keys.

Dependencies: standard library only (numpy is used if available for
rolling statistics but is not required).
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Run a configurable single-asset backtest.

    Parameters
    ----------
    config:
        Runtime configuration dict (from system.ini [backtest] or strategy.yaml).

    Usage
    -----
    >>> runner = BacktestRunner()
    >>> result = runner.run()
    >>> result["metrics"]
    {'total_return': ..., 'sharpe_ratio': ..., ...}
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        policy: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        prices: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run the backtest and return metrics + trade log.

        Parameters
        ----------
        policy:
            Epoch policy dict from the Overseer.
        params:
            Strategy parameters (e.g. from DEAP enrichment).
        prices:
            Optional pre-computed price series.  If ``None``, a synthetic GBM
            series is generated.

        Returns
        -------
        dict
            Keys: ``metrics`` (dict of floats) and ``trades`` (list of dicts).
        """
        policy = policy or {}
        params = params or {}
        n_steps = int(self._config.get("backtest_steps", 252))

        if prices is None:
            prices = self._generate_prices(n_steps)

        buy_threshold = float(params.get("buy_threshold", 0.005))
        sell_threshold = float(params.get("sell_threshold", 0.003))
        position_size = float(params.get("position_size", 0.95))
        fee_bps = float(self._config.get("fee_bps", 10.0))

        trades, portfolio_values = self._simulate(
            prices, buy_threshold, sell_threshold, position_size, fee_bps
        )
        metrics = self._compute_metrics(portfolio_values, trades)
        return {"metrics": metrics, "trades": trades}

    # ------------------------------------------------------------------
    # Strategy simulation
    # ------------------------------------------------------------------

    def _simulate(
        self,
        prices: List[float],
        buy_threshold: float,
        sell_threshold: float,
        position_size: float,
        fee_bps: float,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Threshold momentum strategy simulation."""
        fee = fee_bps / 10_000.0
        cash = 10_000.0
        position = 0.0
        portfolio_values: List[float] = [cash]
        trades: List[Dict[str, Any]] = []
        window = 5

        for i in range(1, len(prices)):
            lookback_price = prices[max(0, i - window)]
            momentum = (prices[i] - lookback_price) / max(lookback_price, 1e-9)

            if momentum > buy_threshold and cash > 0:
                shares = (cash * position_size) / (prices[i] * (1 + fee))
                cost = shares * prices[i] * (1 + fee)
                if cost <= cash:
                    position += shares
                    cash -= cost
                    trades.append(
                        {
                            "step": i,
                            "action": "buy",
                            "price": round(prices[i], 4),
                            "shares": round(shares, 6),
                        }
                    )
            elif momentum < -sell_threshold and position > 0:
                proceeds = position * prices[i] * (1 - fee)
                trades.append(
                    {
                        "step": i,
                        "action": "sell",
                        "price": round(prices[i], 4),
                        "shares": round(position, 6),
                    }
                )
                cash += proceeds
                position = 0.0

            portfolio_values.append(cash + position * prices[i])

        return trades, portfolio_values

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        portfolio_values: List[float],
        trades: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute standard backtest performance metrics."""
        if len(portfolio_values) < 2:
            return {}

        returns = [
            portfolio_values[i] / portfolio_values[i - 1] - 1
            for i in range(1, len(portfolio_values))
        ]
        total_return = portfolio_values[-1] / portfolio_values[0] - 1
        n = len(returns)
        avg_r = sum(returns) / max(n, 1)
        if n >= 2:
            variance = sum((r - avg_r) ** 2 for r in returns) / (n - 1)
            std_r = math.sqrt(variance) if variance > 0 else 1e-9
        else:
            std_r = 1e-9
        sharpe = (avg_r / std_r) * math.sqrt(252)

        peak = portfolio_values[0]
        max_dd = 0.0
        for v in portfolio_values:
            if v > peak:
                peak = v
            dd = (peak - v) / max(peak, 1e-9)
            max_dd = max(max_dd, dd)

        return {
            "total_return": round(total_return, 6),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 6),
            "num_trades": float(len(trades)),
            "final_portfolio": round(portfolio_values[-1], 2),
        }

    # ------------------------------------------------------------------
    # Synthetic data generation
    # ------------------------------------------------------------------

    def _generate_prices(self, n: int) -> List[float]:
        """Generate a synthetic GBM price series of length *n*."""
        seed = int(self._config.get("seed", 42))
        rng = random.Random(seed)
        mu = 0.0002
        sigma = 0.015
        price = 100.0
        prices: List[float] = [price]
        for _ in range(n - 1):
            price *= math.exp(mu + sigma * rng.gauss(0, 1))
            prices.append(price)
        return prices
