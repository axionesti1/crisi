from __future__ import annotations
from typing import Dict, List

from .climate_risk import expected_losses

Year = int

def evaluate_adaptation(hazards: Dict, exposure: Dict[str, float], vuln_params: Dict[str, float], years: range,
                        measures: List[Dict]) -> Dict[str, Dict[Year, float] | float]:
    """Apply adaptation measures as % reduction of damage_rate or vulnerability per hazard.
    Each measure: {'name', 'cost_by_year': {year: cost}, 'effects': {'flood': {'vuln_mult': 0.7}} }
    Returns a dict with baseline_losses, adapted_losses, avoided_losses, adaptation_cost, and adaptation_roi."""
    baseline_losses = expected_losses(hazards, exposure, vuln_params, years)
    mod_vuln = vuln_params.copy()
    total_cost: Dict[Year, float] = {y: 0.0 for y in years}
    for m in measures:
        for y, c in (m.get('cost_by_year') or {}).items():
            total_cost[y] = total_cost.get(y, 0.0) + c
        for hz, eff in (m.get('effects') or {}).items():
            if 'vuln_mult' in eff:
                mod_vuln[hz] = mod_vuln.get(hz, 1.0) * eff['vuln_mult']
    adapted_losses = expected_losses(hazards, exposure, mod_vuln, years)
    avoided_losses = {y: baseline_losses[y] - adapted_losses[y] for y in years}
    pv_avoided = sum(avoided_losses.values())
    pv_cost = sum(total_cost.values())
    adaptation_roi = (pv_avoided - pv_cost) / pv_cost if pv_cost else float('inf')
    return {
        "baseline_losses": baseline_losses,
        "adapted_losses": adapted_losses,
        "avoided_losses": avoided_losses,
        "adaptation_cost": total_cost,
        "adaptation_roi": adaptation_roi,
    }
