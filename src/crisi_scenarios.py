"""
Integrated foresight scenarios for the CRISI model.

This module defines YearlyAssumption and Scenario dataclasses and constructs
five scenarios over the 2025–2055 horizon. Each scenario merges climate
trajectories (RCPs) with socio-economic narratives (SSPs) and provides
annual assumptions for tourism growth, climate policy strength, adaptation
investment, and technological advancement indices.

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

START_YEAR: int = 2025
END_YEAR: int = 2055

@dataclass
class YearlyAssumption:
    year: int
    tourism_growth_rate: float
    climate_policy_index: float
    adaptation_investment_index: float
    technology_advancement_index: float
    notes: Optional[str] = ""

@dataclass
class Scenario:
    name: str
    climate_trajectory: str
    socio_economic_pathway: str
    institutional_sources: List[str]
    description: str
    assumptions_by_year: List[YearlyAssumption] = field(default_factory=list)

    def to_dataframe(self) -> 'pandas.DataFrame':
        import pandas as pd  # type: ignore
        return pd.DataFrame({
            "year": [ya.year for ya in self.assumptions_by_year],
            "tourism_growth_rate": [ya.tourism_growth_rate for ya in self.assumptions_by_year],
            "climate_policy_index": [ya.climate_policy_index for ya in self.assumptions_by_year],
            "adaptation_investment_index": [ya.adaptation_investment_index for ya in self.assumptions_by_year],
            "technology_advancement_index": [ya.technology_advancement_index for ya in self.assumptions_by_year],
            "notes": [ya.notes for ya in self.assumptions_by_year],
        })

def _linspace(start: float, stop: float, num_points: int) -> List[float]:
    if num_points <= 1:
        return [stop]
    step = (stop - start) / (num_points - 1)
    return [start + step * i for i in range(num_points)]

def _create_green_global_resilience() -> Scenario:
    """Optimistic scenario: RCP4.5 with SSP1 (sustainability)."""
    years = list(range(START_YEAR, END_YEAR + 1))
    n = len(years)
    assumptions = []
    for i, year in enumerate(years):
        cpi = 0.7 + (0.9 - 0.7) * i / (n - 1)
        ai  = 0.7 + (0.9 - 0.7) * i / (n - 1)
        ti  = 0.7 + (0.9 - 0.7) * i / (n - 1)
        assumptions.append(YearlyAssumption(
            year=year,
            tourism_growth_rate=0.03,
            climate_policy_index=cpi,
            adaptation_investment_index=ai,
            technology_advancement_index=ti,
            notes=""
        ))
    return Scenario(
        name="Green Global Resilience",
        climate_trajectory="RCP4.5 (stabilising ~2°C warming)",
        socio_economic_pathway="SSP1 – Sustainability",
        institutional_sources=[
            "IPCC RCP4.5/SSP1",
            "EU Strategic Foresight",
            "UNWTO Tourism Towards 2030",
            "OECD Megatrends"
        ],
        description="Optimistic pathway with strong climate policy, high adaptation investments and steady tourism growth.",
        assumptions_by_year=assumptions
    )

def _create_business_as_usual_drift() -> Scenario:
    """Moderate scenario: RCP6.0 with SSP2 (middle of the road)."""
    years = list(range(START_YEAR, END_YEAR + 1))
    n = len(years)
    assumptions = []
    for i, year in enumerate(years):
        tg = 0.025 - (0.005) * i / (n - 1)  # from 2.5% to 2.0%
        cpi = 0.4 + (0.1) * i / (n - 1)     # 0.4 to 0.5
        ai  = 0.35 + (0.15) * i / (n - 1)   # 0.35 to 0.5
        ti  = 0.5 + (0.1) * i / (n - 1)     # 0.5 to 0.6
        assumptions.append(YearlyAssumption(
            year=year,
            tourism_growth_rate=tg,
            climate_policy_index=cpi,
            adaptation_investment_index=ai,
            technology_advancement_index=ti,
            notes=""
        ))
    return Scenario(
        name="Business‑as‑Usual Drift",
        climate_trajectory="RCP6.0 (intermediate‑high warming)",        socio_economic_pathway="SSP2 – Middle of the Road",
        institutional_sources=[
            "IPCC RCP6.0/SSP2",
            "UNWTO baseline projections",
            "OECD Environmental Outlook"
        ],
        description="Continuation of current trends with moderate policy and adaptation efforts; tourism growth slows over time.",
        assumptions_by_year=assumptions
    )

def _create_divided_disparity() -> Scenario:
    """Inequality scenario: SSP4 with medium‑high emissions (like RCP6)."""
    years = list(range(START_YEAR, END_YEAR + 1))
    assumptions = []
    for year in years:
        tg = 0.02 if year <= 2035 else 0.01
        assumptions.append(YearlyAssumption(
            year=year,
            tourism_growth_rate=tg,
            climate_policy_index=0.3,
            adaptation_investment_index=0.35,
            technology_advancement_index=0.75,
            notes=""
        ))
    return Scenario(
        name="Divided Disparity",
        climate_trajectory="RCP6.0‑like",
        socio_economic_pathway="SSP4 – Inequality",
        institutional_sources=[
            "IPCC SSP4",
            "NIC Separate Silos",
            "OECD Megatrends"
        ],
        description="A world of widening inequality; high technology for elites but low adaptation elsewhere; two‑tier tourism system.",
        assumptions_by_year=assumptions
    )

def _create_techno_optimism_hot_planet() -> Scenario:
    """Tech‑driven high‑emissions scenario: RCP8.5 with SSP5."""
    years = list(range(START_YEAR, END_YEAR + 1))
    assumptions = []
    for year in years:
        tg = 0.04 if year <= 2040 else 0.02
        assumptions.append(YearlyAssumption(
            year=year,
            tourism_growth_rate=tg,
            climate_policy_index=0.2,
            adaptation_investment_index=0.3,
            technology_advancement_index=0.85,
            notes=""
        ))
    return Scenario(
        name="Techno‑Optimism on a Hot Planet",
        climate_trajectory="RCP8.5 (very high warming)",
        socio_economic_pathway="SSP5 – Fossil‑fueled Development",
        institutional_sources=[
            "IPCC RCP8.5/SSP5",
            "NIC Competitive Coexistence",
            "UNWTO/WEF tech futures"
        ],
        description="High‑growth world powered by fossil fuels and advanced tech; minimal mitigation triggers severe warming.",
        assumptions_by_year=assumptions
    )

def _create_regional_fortress_world() -> Scenario:
    """Fragmented scenario: SSP3 with high emissions (approx. RCP7)."""
    years = list(range(START_YEAR, END_YEAR + 1))
    assumptions = []
    for year in years:
        tg = 0.015 if year <= 2035 else -0.01
        assumptions.append(YearlyAssumption(
            year=year,
            tourism_growth_rate=tg,
            climate_policy_index=0.2,
            adaptation_investment_index=0.4,
            technology_advancement_index=0.4,
            notes=""
        ))
    return Scenario(
        name="Regional Fortress World",
        climate_trajectory="RCP7.0 (between RCP6.0 and RCP8.5)",
        socio_economic_pathway="SSP3 – Regional Rivalry",
        institutional_sources=[
            "IPCC SSP3",
            "NIC Separate Silos",
            "OECD & EU foresight"
        ],
        description="World fragments into self‑reliant blocs; little cooperation; tourism declines after the mid‑2030s.",
        assumptions_by_year=assumptions
    )

_SCENARIO_BUILDERS = [
    _create_green_global_resilience,
    _create_business_as_usual_drift,
    _create_divided_disparity,
    _create_techno_optimism_hot_planet,
    _create_regional_fortress_world,
]

def list_scenarios() -> List[Scenario]:
    return [builder() for builder in _SCENARIO_BUILDERS]

def get_scenario(name: str) -> Scenario:
    for builder in _SCENARIO_BUILDERS:
        scenario = builder()
        if scenario.name == name:
            return scenario
    raise KeyError(f"No scenario named '{name}' is defined.")

__all__ = [
    "YearlyAssumption",
    "Scenario",
    "list_scenarios",
    "get_scenario",
]
