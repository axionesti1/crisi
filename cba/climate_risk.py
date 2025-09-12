from __future__ import annotations
from typing import Dict

# Alias for year type
Year = int

def expected_losses(hazards: Dict[str, Dict], exposure: Dict[str, float],
                    vuln_params: Dict[str, float], years: range) -> Dict[Year, float]:
    """Estimate expected losses from climate hazards.
    Sums over hazards: probability * damage_rate * exposed value * vulnerability modifier."""
    out: Dict[Year, float] = {y: 0.0 for y in years}
    for hz_name, spec in hazards.items():
        probs = spec.get('prob_by_year', {})
        dmg_rate = float(spec.get('damage_rate', 0.0))
        target = spec.get('target', 'revenue')
        base_val = float(exposure.get(target, 0.0))
        vuln_mult = float(vuln_params.get(hz_name, 1.0))
        for y in years:
            p = probs.get(y, probs.get(max(probs) if probs else y, 0.0))
            out[y] += p * dmg_rate * base_val * vuln_mult
    return out

def adjust_for_risk(fin_cf: Dict[Year, float], exp_loss_cf: Dict[Year, float]) -> Dict[Year, float]:
    """Adjust financial cash flows by subtracting expected losses (negative cashflow).
    Returns new cash flow dictionary combining keys from both."""
    years = set(fin_cf) | set(exp_loss_cf)
    return {y: fin_cf.get(y, 0.0) - exp_loss_cf.get(y, 0.0) for y in years}
