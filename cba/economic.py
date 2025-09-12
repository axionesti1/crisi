from __future__ import annotations
from typing import Dict
from .financial import npv, irr

# Type alias for years


Year = int

def apply_shadow_prices(fin_cf: Dict[Year, float],
                        factors: Dict[str, float] | None = None) -> Dict[Year, float]:
    """Apply shadow price conversion factors to financial cash flows.
    Currently applies a single non_traded factor to all cash flows.
    Extend this function to handle separate factors for traded, labour, etc.
    """
    k = (factors or {}).get("non_traded", 1.0)
    return {y: cf * k for y, cf in fin_cf.items()}


def enpv(econ_cf: Dict[Year, float], sdr: float) -> float:
    """Compute the economic net present value of a cashflow series using a social discount rate."""
    return npv(econ_cf, sdr)


def eirr(econ_cf: Dict[Year, float]) -> float:
    """Compute the economic internal rate of return from an economic cashflow series."""
    return irr(econ_cf)
