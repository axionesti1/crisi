from __future__ import annotations
from typing import Dict

Year = int


def interpolate_path(path: Dict[Year, float]) -> Dict[Year, float]:
    """Fill missing years linearly (e.g., carbon price path)."""
    yrs = sorted(path)
    out: Dict[Year, float] = {}
    for i, y in enumerate(yrs):
        out[y] = path[y]
        if i < len(yrs) - 1:
            y2 = yrs[i+1]
            dy = y2 - y
            inc = (path[y2] - path[y]) / dy
            for k in range(1, dy):
                out[y + k] = path[y] + inc * k
    return out


def carbon_cashflow(tco2_by_year: Dict[Year, float],
                    price_path: Dict[Year, float]) -> Dict[Year, float]:
    """Monetised GHG externality cashflow: negative cost for emissions,
    positive for reductions (tCO2e negative)."""
    p = interpolate_path(price_path)
    return {y: -tco2_by_year.get(y, 0.0) * p.get(y, list(p.values())[-1]) for y in tco2_by_year}


def add_externalities(fin_cf: Dict[Year, float],
                      ext_cf: Dict[Year, float]) -> Dict[Year, float]:
    """Sum financial cf with externalities (economic cashflow)."""
    years = set(fin_cf) | set(ext_cf)
    return {y: fin_cf.get(y, 0.0) + ext_cf.get(y, 0.0) for y in years}
