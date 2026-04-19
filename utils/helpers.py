"""Utility helpers for trading research.

Notes:
- aiomql (MetaTrader 5 async wrapper) is optional; we import it if installed.
- PyPortfolioOpt is optional; we import it if installed.
- Keep this module import-safe even when optional deps are missing.
"""

from __future__ import annotations

from typing import Any, Dict

# Optional dependency imports
try:
    import aiomql  # type: ignore
except Exception:  # pragma: no cover
    aiomql = None  # noqa: N816

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns  # type: ignore
except Exception:  # pragma: no cover
    EfficientFrontier = None  # type: ignore
    risk_models = None  # type: ignore
    expected_returns = None  # type: ignore

def calculate_indicators(data: Any) -> Dict[str, Any]:
    """Placeholder indicator calculation.

    This repo currently does not define a canonical market-data format.
    For now we simply return a dict describing what's available.

    Parameters
    ----------
    data:
        Market data container (e.g., pandas DataFrame or list of candles).

    Returns
    -------
    Dict[str, Any]
        A dict of computed indicators (currently empty) plus basic metadata.
    """

    return {
        "indicators": {},
        "data_type": type(data).__name__,
        "aiomql_available": aiomql is not None,
        "pyportfolioopt_available": EfficientFrontier is not None,
    }