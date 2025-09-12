from __future__ import annotations
from typing import Dict, Iterable
import numpy as np

Year = int

def npv(cashflows: Dict[Year, float], r: float, t0: Year | None = None) -> float:
    """Present value of time-indexed cashflows at rate r.
    cashflows: {year: amount}; r as decimal (0.06 = 6%)."""
    if not cashflows:
        return 0.0
    t0 = t0 if t0 is not None else min(cashflows)
    return sum(cf / ((1 + r) ** (year - t0)) for year, cf in cashflows.items())

def irr(cashflows: Dict[Year, float], guess: float = 0.1) -> float:
    """Internal Rate of Return for unordered, sparse {year: cf}."""
    # Convert to ordered series starting at t0
    if not cashflows:
        raise ValueError("Empty cashflows")
    t0 = min(cashflows)
    years = sorted(cashflows)
    series = [cashflows[y] for y in years]
    times = np.array([y - t0 for y in years], dtype=float)

    # Newton–Raphson on NPV(r)=0
    r = guess
    for _ in range(100):
        denom = (1 + r) ** times
        f = (series / denom).sum()
        df = (-(times * series) / ((1 + r) ** (times + 1))).sum()
        if abs(df) < 1e-12: break
        r_new = r - f / df
        if abs(r_new - r) < 1e-9:
            r = r_new; break
        r = r_new
    return r

def payback_period(cashflows: Dict[Year, float]) -> int | None:
    """Years from first cashflow to reach non-negative cumulative value."""
    if not cashflows:
        return None
    t0 = min(cashflows)
    cum = 0.0
    for y in sorted(cashflows):
        cum += cashflows[y]
        if cum >= 0:
            return y - t0
    return None
