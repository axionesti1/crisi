# app.py — CRISI Streamlit App with Corine Land Cover integration

"""
This Streamlit application visualises climate resilience scenarios and includes a new
section for analysing Corine Land Cover (CLC) data for Greece. The CLC dataset,
often distributed as a large ZIP archive (~250 MB), contains polygon geometries
classified by land cover type across multiple reference years (e.g. 1990, 2000,
2006, 2012, 2018). To accommodate this sizeable dataset, the app loads the
archive on-demand using a cached helper function. Once loaded, users can
filter polygons by reference year, view summary counts by CLC code, and
visualise the shapes on an interactive map.

The original CRISI app provides scenario-based resilience projections,
investment cost–benefit analysis, a simple machine-learning demonstration and
resilience mapping. Those features remain intact, and the new CLC section is
appended at the end of the page.
"""

import os
import sys
import io
import json
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor

# Additional imports for CLC processing
import tempfile
import zipfile

# Imports for real-data integration
from pathlib import Path
import json as _json

# Typing support for type hints
from typing import List, Tuple, Optional, Dict


# Import cost–benefit functions from CRISI package
from cba.climate_risk import adjust_for_risk
from cba.economic import apply_shadow_prices, eirr, enpv
from cba.externalities import add_externalities, carbon_cashflow
from cba.financial import irr, npv, payback_period

# -----------------------------------------------------------------------------
# Real data loader
#
# The preprocessing script ``scripts/prepare_real_data.py`` produces a
# Parquet file and a JSON listing available numeric metrics. If these files
# are present in ``data/processed``, the app will load them and provide
# interactive analysis of the real datasets for Greece. Otherwise, the
# section will instruct the user to run the preprocessing script.



# -----------------------------------------------------------------------------
# Helpers for normalization, climate overrides, and sensitivity blending

import hashlib
def _stable_seed(*parts) -> int:
    h = hashlib.sha256(("|".join(map(str, parts))).encode()).hexdigest()
    return int(h[:16], 16) % (2**32)

def compute_norm_constants(df: pd.DataFrame, mode: str = "global", year_ref: int = 2019, region: str | None = None):
    """
    Compute normalization constants for education, R&D and GDP per capita.
    mode: "global" | "cross_section_2019" | "by_region"
    """
    if df is None or df.empty:
        return 1.0, 1.0, 1.0
    cols = {"edu": "main__education", "rnd": "main__rnd_expe"}
    # Build working df copy
    dfx = df.copy()
    # Optionally compute gdp_pc if missing
    if "gdp_pc" not in dfx.columns and {"main__gdp", "main__population"}.issubset(dfx.columns):
        with pd.option_context("mode.use_inf_as_na", True):
            dfx["gdp_pc"] = (dfx["main__gdp"] * 1e6) / dfx["main__population"].replace(0, pd.NA)
    # Filter per mode
    if mode == "cross_section_2019":
        dfx = dfx[dfx.get("year", pd.Series([])).astype(str) <= str(year_ref)]
        dfx = dfx[dfx.get("year", pd.Series([])).astype(int) == int(year_ref)] if "year" in dfx.columns else dfx
    elif mode == "by_region" and region is not None:
        dfx = dfx[dfx["_region_norm"].astype(str).str.strip() == str(region).strip()]
    # Compute maxima with safety fallback
    max_edu = float(dfx.get(cols["edu"], pd.Series([0])).dropna().max()) if cols["edu"] in dfx.columns else 1.0
    max_rnd = float(dfx.get(cols["rnd"], pd.Series([0])).dropna().max()) if cols["rnd"] in dfx.columns else 1.0
    max_gdp = float(dfx.get("gdp_pc", pd.Series([0])).dropna().max()) if "gdp_pc" in dfx.columns else 1.0
    return (max_edu if max_edu > 0 else 1.0, max_rnd if max_rnd > 0 else 1.0, max_gdp if max_gdp > 0 else 1.0)

def blend_sensitivity_signals(row_or_df, eta: float, norm_consts=(1.0,1.0,1.0)):
    """
    Blend tourism_units_share with (normalized) international arrivals if available.
    eta in [0,1] is the weight on arrivals.
    """
    def _norm_arrivals(x, amin=None, amax=None):
        try:
            if amin is None or amax is None or amax <= amin:
                # compute from series
                xs = pd.to_numeric(x, errors="coerce")
                amin_ = float(xs.min())
                amax_ = float(xs.max())
                return (xs - amin_) / (amax_ - amin_) if amax_ > amin_ else xs * 0.0
            else:
                return (x - amin) / (amax - amin)
        except Exception:
            return pd.Series([pd.NA] * len(x)) if hasattr(x, "__len__") else pd.NA

    tu = None
    ar = None
    if isinstance(row_or_df, pd.Series):
        tu = row_or_df.get("tourism__tourism_units_share", pd.NA)
        if pd.isna(tu):
            tu = row_or_df.get("tourism_units_share", pd.NA)
        ar = row_or_df.get("main__international_arrivals", pd.NA)
        if pd.isna(ar):
            return float(tu) if not pd.isna(tu) else float("nan")
        # Row case: can't normalize robustly without context; return simple blend with guard
        return float((1-eta) * (tu if not pd.isna(tu) else 0.0) + eta * 0.0)
    else:
        df = row_or_df.copy()
        if "tourism__tourism_units_share" in df.columns:
            tu = pd.to_numeric(df["tourism__tourism_units_share"], errors="coerce")
        elif "tourism_units_share" in df.columns:
            tu = pd.to_numeric(df["tourism_units_share"], errors="coerce")
        else:
            tu = pd.Series([pd.NA]*len(df))
        if "main__international_arrivals" in df.columns:
            ar = pd.to_numeric(df["main__international_arrivals"], errors="coerce")
            ar_norm = _norm_arrivals(ar)
            blended = (1-eta) * tu.fillna(0.0) + eta * ar_norm.fillna(0.0)
            return blended.clip(0,1)
        return tu

# Climate override holder (populated from sidebar uploader if provided)
CLIMATE_OVERRIDE = None
CLIMATE_OVERRIDE_NORMALIZE = True

# --- Added robust helpers ---
def safe_minmax(x):
    xs = pd.to_numeric(x, errors="coerce")
    try:
        mn, mx = float(np.nanmin(xs)), float(np.nanmax(xs))
    except Exception:
        return xs*0.0 if hasattr(xs, "__len__") else 0.0
    return (xs - mn) / (mx - mn) if mx > mn else (xs*0.0 if hasattr(xs, "__len__") else 0.0)

def _filter_region(df, region):
    key = "_region_norm"
    if key not in df.columns:
        return df.iloc[0:0].copy()
    lhs = df[key].astype(str).str.strip().str.casefold()
    rhs = str(region).strip().casefold()
    return df[lhs == rhs].copy()

@st.cache_data(show_spinner=True)
def load_real_data():
    """Load processed Greece datasets and metrics if available.

    Returns a tuple ``(df, clc_df, metrics)`` where ``df`` is the unified
    DataFrame keyed by ``_region_norm`` and ``year``, ``clc_df`` is a CLC
    summary (optional), and ``metrics`` is a list of numeric columns for
    selection in the UI.
    """
    base_dir = Path(__file__).resolve().parent / "data" / "processed"
    parquet_file = base_dir / "crisi_greece_processed.parquet"
    metrics_file = base_dir / "crisi_metrics.json"
    clc_file = base_dir / "clc_greece_summary.parquet"
    # Try Parquet first, fallback to CSV
    df = pd.DataFrame()
    if parquet_file.exists():
        try:
            df = pd.read_parquet(parquet_file)
        except Exception:
            pass
    # fallback to CSV
    csv_file = parquet_file.with_suffix(".csv")
    if df.empty and csv_file.exists():
        df = pd.read_csv(csv_file, low_memory=False)
    if df.empty:
        return pd.DataFrame(), None, []
    # load metrics list
    metrics: List[str] = []
    if metrics_file.exists():
        try:
            metrics = _json.loads(metrics_file.read_text()).get("available_metrics", [])
        except Exception:
            metrics = []
    # Exclude environmental tax variables (if present) unless they have meaningful data
    # They often produce missing or unreliable values in the projections.
    metrics = [m for m in metrics if not any(sub in m.lower() for sub in ["environ_tax", "total_environ_tax", "energy_environ_tax", "transport_environ_tax"])]
    # clc summary if exists
    clc_df = None
    if clc_file.exists():
        try:
            clc_df = pd.read_parquet(clc_file)
        except Exception:
            clc_df = None
    return df, clc_df, metrics


# ---------- Streamlit page setup ----------
st.set_page_config(page_title="CRISI Model Explorer", layout="wide")
st.title("🌍 CRISI: Climate Resilience Investment Scoring")

st.caption(
    "Real-data enabled prototype with live API hooks (Copernicus, Eurostat, World Bank, Open-Meteo), "
    "NUTS2 mapping, ML-based tourism impact signal, and side-by-side scenario comparison."
)


# ---------- Helpers ----------
def normalize_weights(*ws):
    s = sum(ws)
    if s == 0:
        n = len(ws)
        return tuple([1.0 / n] * n)
    return tuple([w / s for w in ws])


def scenario_params(name: str):
    """Return (climate_factor, tourism_share, demand_drop, readiness_base) based on scenario name."""
    if name.startswith("Green"):
        return (1.0, 0.30, 0.0, 0.80)
    if name.startswith("Business"):
        return (1.2, 0.35, 0.0, 0.60)
    if name.startswith("Divided"):
        return (1.3, 0.40, 0.0, 0.50)
    if name.startswith("Techno"):
        return (1.5, 0.45, 0.0, 0.70)
    if name.startswith("Regional"):
        return (1.4, 0.65, 25.0, 0.30)
    return (1.0, 0.35, 0.0, 0.50)

# -----------------------------------------------------------------------------
# Variable metadata and units
#
# To improve interpretability of the real data charts, we provide a mapping
# between the raw variable names used in the processed dataset and their
# corresponding units.  These units are displayed below the charts and tables
# in the Real Data Analysis section.  When new variables are added to the
# dataset, extend this dictionary accordingly.  Variables not present in
# ``VARIABLE_UNITS`` default to an unspecified unit.
VARIABLE_UNITS: Dict[str, str] = {
    "tourism_gdp_pct": "percentage (%)",
    "population": "number",
    "pop_density": "persons per square kilometre",
    "pop_change": "crude rate of total population change",
    "education": "percentage (%)",
    "rnd_expe": "percentage (%) of gross domestic product (GDP)",
    "hospital_beds": "per hundred thousand inhabitants",
    "at_risk_poverty": "percentage (%) of total population",
    "shortstay_nights_spent": "number",
    "gdp": "million euro",
    "per_gdp": "euro",
    "total_nights_spent": "number",
    "arrivals_at_accomond": "number",
    "international_arrivals": "number",
    "total_environ_tax": "million euro",
    "energy_environ_tax": "million euro",
    "transport_environ_tax": "million euro",
    "no_units": "number",
    "turnover": "number",
    "employment": "number",
}

# ---------------- Real-data projection utilities ----------------

def compute_real_projection(df: pd.DataFrame, region: str, scenario: str, end_year: int = 2055, alpha: float = 0.33, beta: float = 0.33, gamma: float = 0.34, w_heat: float = 1.0, w_gdp: float = 1.0, baseline_cutoff_year: Optional[int] = None, norm_mode: str = "global", arrivals_weight: float = 0.0, policy_kappa: float = 0.0, annual_adapt_invest: float = 0.0) -> pd.DataFrame:
    """
    Project resilience indicators from the last available year up to ``end_year``
    using observed real data as the baseline.  This function derives
    initial exposure, sensitivity and adaptive capacity values from the
    processed dataset ``df`` for the specified region and then applies
    scenario-specific growth rates to simulate their evolution.

    Parameters
    ----------
    df : pandas.DataFrame
        Unified real data loaded via ``load_real_data()``.
    region : str
        Normalised region name (``_region_norm``) for which to compute the
        projection.
    scenario : str
        Scenario name (e.g. ``"Green"``, ``"Business"``) as defined in
        ``scenario_params`` or the projection parameters dictionary.
    end_year : int, optional
        Last year of the projection (inclusive).  Defaults to 2055.

    Parameters
    ----------
    w_heat : float, optional
        User‑selected weight for heat relative to drought.  Because only
        temperature (heat) is observed in the real dataset, this weight
        serves as a simple scaling factor on exposure.  A value of 0
        suppresses exposure entirely; 1 leaves exposure unchanged.
    w_gdp : float, optional
        User‑selected weight for tourism GDP versus seasonality.  In the
        absence of explicit seasonality data, the tourism share is scaled by
        this weight to influence sensitivity.  A value of 0 suppresses
        sensitivity; 1 leaves it unchanged.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``['year','exposure','sensitivity','adaptive','resilience', 'Tmax_proj', 'tourism_share_proj', 'gdp_pc_proj']``.
    """
    # Select the region-specific data and obtain the baseline year
    sdf = _filter_region(df, region)
    if sdf.empty or "year" not in sdf.columns:
        return pd.DataFrame()
    # Determine the baseline (last observed) year relative to the projection horizon.
    # We want to anchor the projection on the most recent year (≤ end_year)
    # that has actual data for at least one of the key variables (tourism
    # share, education, R&D expenditure, GDP or population).  This avoids
    # picking far‑future placeholder years (e.g. 2090 or 2100) filled with
    # modeled values only.  If ``baseline_cutoff_year`` is provided, we
    # further restrict candidate rows to years ≤ baseline_cutoff_year.  This
    # allows users to anchor projections on the last year with observed
    # socio‑economic data (typically ≤ 2023) rather than on the final
    # modeled year.
    candidate_cols: list[str] = []
    # Tourism share (units)
    tour_col = "tourism__tourism_units_share" if "tourism__tourism_units_share" in sdf.columns else None
    if tour_col:
        candidate_cols.append(tour_col)
    # Education and R&D columns
    edu_col = "main__education" if "main__education" in sdf.columns else None
    rnd_col = "main__rnd_expe" if "main__rnd_expe" in sdf.columns else None
    if edu_col:
        candidate_cols.append(edu_col)
    if rnd_col:
        candidate_cols.append(rnd_col)
    # GDP and population columns
    if "main__gdp" in sdf.columns:
        candidate_cols.append("main__gdp")
    if "main__population" in sdf.columns:
        candidate_cols.append("main__population")
    # Filter rows with any non‑null candidate value
    candidate = sdf
    if candidate_cols:
        candidate = sdf[sdf[candidate_cols].notna().sum(axis=1) >= 2]
    # Restrict to years ≤ end_year
    try:
        candidate = candidate[candidate["year"].astype(float) <= end_year]
    except Exception:
        candidate = candidate
    # Further restrict by baseline_cutoff_year if provided
    if baseline_cutoff_year is not None:
        try:
            candidate = candidate[candidate["year"].astype(float) <= baseline_cutoff_year]
        except Exception:
            pass
    if not candidate.empty and "year" in candidate.columns:
        last_year = int(candidate["year"].astype(int).max())
    else:
        # If no candidate rows, fallback to the latest year ≤ end_year (and ≤ cutoff if set)
        try:
            sdf_years = sdf["year"].dropna().astype(int)
        except Exception:
            sdf_years = pd.Series([], dtype=int)
        if not sdf_years.empty:
            valid_years = sdf_years[sdf_years <= end_year]
            if baseline_cutoff_year is not None:
                valid_years = valid_years[valid_years <= baseline_cutoff_year]
            last_year = int(valid_years.max()) if not valid_years.empty else int(sdf_years.max())
        else:
            last_year = min(end_year, baseline_cutoff_year) if baseline_cutoff_year is not None else end_year
    # Extract baseline row for the chosen year; if that year is absent, use the
    # first available row as a fallback.  (This should rarely occur.)
    row_candidates = sdf[sdf["year"] == last_year]
    if not row_candidates.empty:
        row = row_candidates.iloc[0]
    else:
        row = sdf.iloc[0]
    # Compute GDP per capita to use in adaptive capacity
    # Avoid division by zero by adding a small constant.  We compute a column
    # ``gdp_pc`` on a copy of the full DataFrame.  If gdp or population are missing,
    # ``gdp_pc`` will be NaN.
    df_tmp = df.copy()
    if "main__gdp" in df_tmp.columns and "main__population" in df_tmp.columns:
        try:
            df_tmp["gdp_pc"] = (df_tmp["main__gdp"] * 1e6) / (df_tmp["main__population"].replace(0, np.nan))
        except Exception:
            df_tmp["gdp_pc"] = np.nan
    else:
        df_tmp["gdp_pc"] = np.nan
    # Normalisation constants across all regions/years
    # Temperature (exposure) uses Tmax.  If unavailable, fall back to [0,1].
    temp_col = "temp__Tmax" if "temp__Tmax" in df_tmp.columns else None
    if temp_col and df_tmp[temp_col].notna().any():
        min_temp = float(df_tmp[temp_col].min())
        max_temp = float(df_tmp[temp_col].max())
    else:
        min_temp = 0.0
        max_temp = 1.0
    # Education, R&D expenditure and GDP per capita (adaptive) – compute maxima
    edu_col = "main__education" if "main__education" in df_tmp.columns else None
    rnd_col = "main__rnd_expe" if "main__rnd_expe" in df_tmp.columns else None
    gdp_col = "gdp_pc" if "gdp_pc" in df_tmp.columns else None
    max_edu, max_rnd, max_gdp_pc = compute_norm_constants(df_tmp, mode=norm_mode, year_ref=2019, region=region)
    # Baseline indices
    # Exposure derived from temperature (higher temperature → higher exposure).  If the
    # region has no Tmax data, fall back to a default mid‑point (0.5).  Apply
    # the heat weight to scale exposure.  When w_heat is smaller than 1, exposure
    # decreases proportionally; a value of zero effectively removes the effect of
    # heat from the index.
    # Compute temp component
    if temp_col and not pd.isna(row.get(temp_col)) and (max_temp > min_temp):
        temp_comp = float((row[temp_col] - min_temp) / (max_temp - min_temp))
    else:
        temp_comp = 0.5
    # Optional drought component from real data if available
    drought_cols = [c for c in df_tmp.columns if 'drought' in c.lower()]
    if drought_cols:
        try:
            dseries = pd.to_numeric(df_tmp[drought_cols[0]], errors='coerce')
            dmin, dmax = float(dseries.min()), float(dseries.max())
            dval = float(row.get(drought_cols[0], np.nan))
            drought_comp = (dval - dmin) / (dmax - dmin) if dmax > dmin and not pd.isna(dval) else 0.5
        except Exception:
            drought_comp = 0.5
    else:
        drought_comp = 0.5
    base_exposure = float(w_heat) * temp_comp + (1.0 - float(w_heat)) * drought_comp
    # Sensitivity derived from tourism share (if available).  If not, use a moderate
    # default (0.3).  Apply the tourism weight ``w_gdp`` to scale sensitivity.  This
    # weight reflects the relative importance of tourism versus unobserved
    # seasonality or other factors.
    tour_col = "tourism__tourism_units_share" if "tourism__tourism_units_share" in row else None
    if tour_col and not pd.isna(row.get(tour_col)):
        ts_val = float(row[tour_col])
    else:
        ts_val = 0.3
    # arrivals contribution (min-max within df_tmp)
    if arrivals_weight and 'main__international_arrivals' in df_tmp.columns:
        arr_series = pd.to_numeric(df_tmp['main__international_arrivals'], errors='coerce')
        amin, amax = float(arr_series.min()), float(arr_series.max())
        aval = float(row.get('main__international_arrivals', np.nan))
        arr_norm = (aval - amin) / (amax - amin) if amax > amin and not pd.isna(aval) else 0.0
        ts_blend = (1.0 - arrivals_weight) * ts_val + arrivals_weight * arr_norm
    else:
        ts_blend = ts_val
    base_sensitivity = float(w_gdp) * float(ts_blend)
    # Adaptive capacity derived from education, R&D expenditure and GDP per capita.
    # Normalise each component by its maximum across the dataset; then average.
    val_edu = row.get(edu_col, np.nan) if edu_col else np.nan
    val_rnd = row.get(rnd_col, np.nan) if rnd_col else np.nan
    # Compute GDP per capita for the baseline row if gdp and population exist
    if "main__gdp" in row and "main__population" in row:
        try:
            gval = row.get("main__gdp")
            pval = row.get("main__population")
            if gval and not pd.isna(gval) and pval and not pd.isna(pval):
                val_gdp_pc = (gval * 1e6) / pval
            else:
                val_gdp_pc = np.nan
        except Exception:
            val_gdp_pc = np.nan
    else:
        val_gdp_pc = np.nan
    # If gdp per capita is missing, estimate it as the mean of available gdp_pc
    if pd.isna(val_gdp_pc) and gdp_col and df_tmp[gdp_col].notna().any():
        val_gdp_pc = float(df_tmp[gdp_col].dropna().mean())
    norm_edu = (val_edu / max_edu) if (edu_col and max_edu) and not pd.isna(val_edu) else 0.0
    norm_rnd = (val_rnd / max_rnd) if (rnd_col and max_rnd) and not pd.isna(val_rnd) else 0.0
    norm_gdp_pc = (val_gdp_pc / max_gdp_pc) if (max_gdp_pc and not pd.isna(val_gdp_pc)) else 0.0
    # If all components are NaN or zero, fall back to 0.4
    if np.isnan([norm_edu, norm_rnd, norm_gdp_pc]).all() or (norm_edu + norm_rnd + norm_gdp_pc) == 0.0:
        base_adaptive = 0.4
    else:
        base_adaptive = float(np.nanmean([norm_edu, norm_rnd, norm_gdp_pc]))
    # Scenario-specific growth rates: hazard, adaptation, tourism change
# Scenario-specific growth rates: hazard, adaptation, tourism change.
    # These parameters have been tuned to produce distinct trajectories: for
    # example, the Divided scenario has high hazard growth and no adaptation
    # improvement, leading to declining resilience, while the Regional
    # scenario sees moderate hazard growth, negative adaptation growth and a
    # steep decline in tourism share.  Modify these values to calibrate
    # future projections.
    scenario_growth = {
        "Green": {"hazard": 0.010, "adapt": 0.020, "tourism": -0.005},
        "Business": {"hazard": 0.015, "adapt": 0.010, "tourism": 0.000},
        # In the Divided scenario, hazard accelerates and adaptation stalls,
        # while tourism demand increases, raising sensitivity and lowering
        # resilience over time.
        "Divided": {"hazard": 0.025, "adapt": 0.000, "tourism": 0.010},
        # Techno emphasises rapid adaptation despite high hazard growth.
        "Techno": {"hazard": 0.030, "adapt": 0.040, "tourism": 0.000},
        # Regional sees moderate hazard growth but declining adaptation and
        # strong drops in tourism demand.
        "Regional": {"hazard": 0.020, "adapt": -0.005, "tourism": -0.020},
    }
    growth = scenario_growth.get(scenario.split()[0], scenario_growth.get(scenario, scenario_growth["Business"]))
    # Initialise lists
    years = list(range(last_year, end_year + 1))
    exposures: list[float] = []
    sensitivities: list[float] = []
    adaptives: list[float] = []
    resiliences: list[float] = []
    # Lists for projected real variables
    tmax_proj: list[float] = []
    tourism_share_proj: list[float] = []
    gdp_pc_proj: list[float] = []
    # Baseline values for real variables
    baseline_tmax = row.get(temp_col, np.nan) if temp_col else np.nan
    baseline_tourism = base_sensitivity  # base_sensitivity is baseline tourism share if available
    # Compute baseline GDP per capita from the row if possible
    baseline_gdp_pc = np.nan
    if "main__gdp" in row and "main__population" in row:
        try:
            pop_val = row.get("main__population")
            gdp_val = row.get("main__gdp")
            if pop_val and not pd.isna(pop_val) and pop_val != 0 and gdp_val and not pd.isna(gdp_val):
                baseline_gdp_pc = (gdp_val * 1e6) / pop_val
        except Exception:
            baseline_gdp_pc = np.nan
    # Current state variables for iterative projection
    e = base_exposure
    s = base_sensitivity
    a = base_adaptive
    # If the baseline GDP per capita is missing, use a sensible fallback.  First try
    # the mean of available ``gdp_pc`` values across the entire dataset; if
    # unavailable, use the previously computed ``val_gdp_pc`` (dataset mean used
    # when calculating adaptive capacity).  As a last resort, default to 0.0 to
    # avoid propagating NaNs.
    current_gdp_pc = baseline_gdp_pc
    if pd.isna(current_gdp_pc):
        # overall mean from df_tmp if gdp_pc exists
        if gdp_col and df_tmp[gdp_col].notna().any():
            try:
                current_gdp_pc = float(df_tmp[gdp_col].dropna().mean())
            except Exception:
                current_gdp_pc = np.nan
        # fallback to val_gdp_pc computed earlier
        if pd.isna(current_gdp_pc):
            try:
                if 'val_gdp_pc' in locals() and not pd.isna(val_gdp_pc):
                    current_gdp_pc = float(val_gdp_pc)
            except Exception:
                current_gdp_pc = np.nan
        # final fallback
        if pd.isna(current_gdp_pc):
            current_gdp_pc = 0.0
    for yr in years:
        # Index values clipped to [0,1]
        exposures.append(float(np.clip(e, 0.0, 1.0)))
        sensitivities.append(float(np.clip(s, 0.0, 1.0)))
        adaptives.append(float(np.clip(a, 0.0, 1.0)))
        # Compute resilience using user-defined weights.  The formula divides
        # weighted adaptive capacity by weighted exposure and sensitivity,
        # ensuring that higher exposure or sensitivity reduce resilience.  A
        # small epsilon avoids division by zero.
        denom = alpha * e + beta * s + 1e-6
        res_val = (gamma * a) / denom if denom > 0 else 0.0
        # Clip resilience to avoid extreme values; the result will typically
        # stay within [0, 10], but users may adjust weights to push it.
        resiliences.append(float(np.clip(res_val, 0.0, 10.0)))
        # Project real variables
        # Temperature: invert exposure index back to degrees using min/max temp
        if temp_col and not pd.isna(baseline_tmax) and (max_temp > min_temp):
            t_proj = e * (max_temp - min_temp) + min_temp
            tmax_proj.append(float(t_proj))
        else:
            tmax_proj.append(np.nan)
        # Tourism share: update along with sensitivity (clipped to [0,1])
        tourism_share_proj.append(float(np.clip(s, 0.0, 1.0)))
        # GDP per capita: grow at adaptation rate
        gdp_pc_proj.append(float(current_gdp_pc) if not pd.isna(current_gdp_pc) else np.nan)
        if not pd.isna(current_gdp_pc):
            current_gdp_pc = current_gdp_pc * (1.0 + growth["adapt"])
        # Update indices for next year
        e = e * (1.0 + growth["hazard"])
        s = s + growth["tourism"]
        a = a + growth["adapt"]
    return pd.DataFrame({
        "year": years,
        "exposure": exposures,
        "sensitivity": sensitivities,
        "adaptive": adaptives,
        "resilience": resiliences,
        # Additional real variable projections
        "Tmax_proj": tmax_proj,
        "tourism_share_proj": tourism_share_proj,
        "gdp_pc_proj": gdp_pc_proj,
    })


def compute_series(years: np.ndarray, climate_factor: float, tourism_share: float, readiness_base: float, w_heat: float, w_gdp: float, a: float, b: float, g: float, demand_drop: float = 0.0, policy_kappa: float = 0.0, annual_adapt_invest: float = 0.0, climate_override: pd.DataFrame | None = None, climate_override_normalize: bool = True) -> pd.DataFrame:
    """
    Generate a time‑series for a given scenario.

    This updated implementation aims to produce more varied scenario trajectories by
    adjusting the start/end values of the underlying climate indices and by
    incorporating the scenario‐specific ``demand_drop`` parameter into the
    tourism share trend.  Climate factors now influence both the magnitude
    and slope of the heat, drought and seasonality indices.  Adaptive capacity
    (readiness) is allowed to evolve over time with a ceiling that depends on
    the climate factor (hotter worlds make adaptation harder).  Random noise is
    added to each index to avoid perfectly straight lines and is seeded with
    a hash of the climate factor and tourism share to make trajectories
    reproducible across runs.

    Parameters
    ----------
    years : np.ndarray
        Array of years for which to compute the series.
    climate_factor : float
        Scenario multiplier for the magnitude of climate signals (heat, drought, seasonality).
    tourism_share : float
        Base tourism share for sensitivity calculations.
    readiness_base : float
        Initial adaptive readiness value at the start year.
    w_heat, w_gdp, a, b, g : float
        Weight parameters controlling aggregation of sub‑indicators and pillars.
    demand_drop : float, optional
        Percentage drop in tourism demand over the projection horizon.  A value
        of zero keeps tourism share flat; positive values gradually reduce
        tourism share over time.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by year with columns for Exposure, Sensitivity,
        Adaptive, Resilience and Risk.
    """
    n = len(years)
    # Seed RNG using a hash of core parameters to make each scenario reproducible
    seed_val = _stable_seed((climate_factor, tourism_share, readiness_base, demand_drop))
    rng = np.random.default_rng(seed_val)

    # Determine start and end values for climate indices unless overridden.
    # If climate_override is provided, we will use columns 'year','heat','drought' (already in [0,1] or normalized if requested).
    # increases.  Values are clipped to [0, 1].
    heat_start, heat_end = 0.1, min(1.0, 0.1 + 0.7 * climate_factor)
    drought_start, drought_end = 0.2, min(1.0, 0.2 + 0.5 * climate_factor)
    season_start, season_end = 0.3, min(1.0, 0.3 + 0.15 * climate_factor)

    heat_index = np.clip(
        np.linspace(heat_start, heat_end, n) + rng.normal(0, 0.03, n), 0, 1
    )
    drought_index = np.clip(
        np.linspace(drought_start, drought_end, n) + rng.normal(0, 0.03, n), 0, 1
    )
    seasonality = np.clip(
        np.linspace(season_start, season_end, n) + rng.normal(0, 0.015, n), 0, 1
    )

    # Override from external climate if provided
    if climate_override is not None and not climate_override.empty:
        dfc = climate_override.copy()
        if "year" in dfc.columns:
            dfc = dfc.set_index("year")
            sub = dfc.reindex(years)
            heat_series = pd.to_numeric(sub.get("heat", pd.Series([pd.NA]*len(years))), errors="coerce")
            drought_series = pd.to_numeric(sub.get("drought", pd.Series([pd.NA]*len(years))), errors="coerce")
            if climate_override_normalize:
                # Min-max within provided series
                def _mm(x):
                    x = x.astype(float)
                    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
                    return (x - mn) / (mx - mn) if mx > mn else x * 0.0
                heat_series = _mm(heat_series)
                drought_series = _mm(drought_series)
            # Fill defaults if missing
            heat_index = np.clip(heat_series.fillna(method="ffill").fillna(method="bfill").fillna(0.0).values, 0, 1)
            drought_index = np.clip(drought_series.fillna(method="ffill").fillna(method="bfill").fillna(0.0).values, 0, 1)
        # else fall back to synthetic below

    # Adaptive readiness evolves from readiness_base to a ceiling that depends on
    # climate_factor: hotter scenarios slow the rate of improvement.
    adapt_ceiling = min(1.0, readiness_base + 0.15 / max(climate_factor, 0.5))
    readiness = np.linspace(readiness_base, adapt_ceiling, n)
    # Policy feedback: adaptation investment boosts readiness linearly (simple prototype)
    if policy_kappa and annual_adapt_invest:
        boost = policy_kappa * (annual_adapt_invest / 1_000_000.0) * np.linspace(0, 1, n)
        readiness = np.clip(readiness + boost, 0, 1)

    # Tourism share may decline over time if demand_drop > 0.  We model a linear
    # interpolation from the starting value to a reduced end value.
    tourism_end = max(0.0, tourism_share * (1.0 - demand_drop / 100.0))
    tourism_share_series = np.linspace(tourism_share, tourism_end, n)

    # Compute pillar sub‑indices
    exposure = w_heat * heat_index + (1.0 - w_heat) * drought_index
    sensitivity = w_gdp * tourism_share_series + (1.0 - w_gdp) * seasonality
    adaptive = readiness

    # Normalise pillar weights and compute resilience/risk
    a_n, b_n, g_n = normalize_weights(a, b, g)
    resilience = np.clip(
        a_n * (1.0 - exposure) + b_n * (1.0 - sensitivity) + g_n * adaptive,
        0,
        1,
    )
    risk = np.clip(
        a_n * exposure + b_n * sensitivity + g_n * (1.0 - adaptive), 0, 1
    )

    df = pd.DataFrame(
        {
            "Year": years,
            "Exposure": np.round(exposure, 3),
            "Sensitivity": np.round(sensitivity, 3),
            "Adaptive": np.round(adaptive, 3),
            "Resilience": np.round(resilience, 3),
            "Risk": np.round(risk, 3),
        }
    ).set_index("Year")

    return df


# ---------- Live data hooks (optional, safe) ----------
@st.cache_data(ttl=3600)
def get_open_meteo_temp(lat=37.98, lon=23.72):
    """Fetch a single hourly temperature value from the Open-Meteo API."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m", "forecast_days": 1}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        js = r.json()
        return js.get("hourly", {}).get("temperature_2m", [None])[0]
    except Exception as e:
        st.warning(f"Open-Meteo unavailable: {e}")
        return None


@st.cache_data(ttl=86400)
def get_worldbank_df(country_codes=("GR", "DE")):
    """Fetch a tiny GDP per capita sample from the World Bank for connectivity checks."""
    try:
        import wbdata
    except Exception as e:
        st.info("World Bank library not installed; skipping WB call.")
        return None
    try:
        ind = {"NY.GDP.PCAP.CD": "gdp_pc"}
        df = wbdata.get_dataframe(ind, country=list(country_codes), date=("2019", "2024"))
        return df.reset_index()
    except Exception as e:
        st.warning(f"World Bank call failed: {e}")
        return None


@st.cache_data(ttl=86400)
def get_eurostat_tiny():
    """Ping Eurostat via pandasdmx to confirm connectivity without heavy downloads."""
    try:
        from pandasdmx import Request
        estat = Request("ESTAT")
        _ = estat.datastructure("ESTAT").response
        return True
    except Exception as e:
        st.info(f"Eurostat (pandasdmx) not reachable right now: {e}")
        return False


@st.cache_resource
def get_nuts2_gdf():
    """
    Load NUTS level‑2 boundaries as a GeoDataFrame in EPSG:4326.

    To avoid external download errors and confusing messages, this helper first
    looks for a local GeoJSON file in ``./data/nuts2.geojson``.  If not found,
    it attempts to fetch the official GISCO GeoJSON distribution (20M scale).
    No shapefile downloads are attempted.  If both options fail, ``None`` is
    returned.
    """
    base_dir = os.path.join(os.getcwd(), "data")
    local_geojson = os.path.join(base_dir, "nuts2.geojson")
    try:
        if os.path.exists(local_geojson):
            gdf = gpd.read_file(local_geojson).to_crs(4326)
        else:
            geojson_url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_20M_2021_4326.geojson"
            gdf = gpd.read_file(geojson_url).to_crs(4326)
    except Exception:
        return None
    # filter to level 2 and standardise columns
    if "LEVL_CODE" in gdf.columns:
        gdf = gdf[gdf["LEVL_CODE"] == 2]
    if "NUTS_ID" in gdf.columns:
        gdf = gdf.rename(columns={"NUTS_ID": "id"})
    if "NAME_LATN" in gdf.columns:
        gdf = gdf.rename(columns={"NAME_LATN": "region_name"})
    elif "NUTS_NAME" in gdf.columns:
        gdf = gdf.rename(columns={"NUTS_NAME": "region_name"})
    keep_cols = [c for c in ["id", "CNTR_CODE", "region_name", "geometry"] if c in gdf.columns]
    return gdf[keep_cols]


# ---------- Sidebar Controls ----------
scenarios_all = [
    "Green Global Resilience (RCP4.5/SSP1)",
    "Business-as-Usual Drift (RCP6.0/SSP2)",
    "Divided Disparity (RCP6.0-like/SSP4)",
    "Techno-Optimism on a Hot Planet (RCP8.5/SSP5)",
    "Regional Fortress World (RCP7.0/SSP3)",
]
compare_all = st.sidebar.checkbox("Compare ALL scenarios side-by-side", value=True)
if compare_all:
    selected_scenarios = scenarios_all
else:
    selected_scenarios = [st.sidebar.selectbox("Select scenario", scenarios_all, key="single_scenario_select")]


with st.expander("Scenario parameters (transparency)", expanded=False):
    import pandas as _pd
    _rows = []
    for _name in scenarios_all:
        cf, ts, dd, rb = scenario_params(_name)
        _rows.append({"Scenario": _name, "climate_factor": cf, "tourism_share": ts, "demand_drop%": dd, "readiness_base": rb})
    _dfp = _pd.DataFrame(_rows)
    st.dataframe(_dfp, use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.subheader("Pillar Weights")
alpha = st.sidebar.slider("Exposure (α)", 0.0, 1.0, 0.33, 0.01)
beta = st.sidebar.slider("Sensitivity (β)", 0.0, 1.0, 0.33, 0.01)
gamma = st.sidebar.slider("Adaptive (γ)", 0.0, 1.0, 0.34, 0.01)

st.sidebar.subheader("Sub-indicator Weights")
st.sidebar.subheader("Normalization & Sensitivity Proxy")
norm_mode = st.sidebar.selectbox("Normalization mode", ["global", "cross_section_2019", "by_region"], index=0)
arrivals_weight = st.sidebar.slider("Arrivals weight in sensitivity (η)", 0.0, 1.0, 0.3, 0.05)
st.sidebar.subheader("Climate time series (optional)")
clim_file = st.sidebar.file_uploader("Upload climate CSV with columns year,heat,drought", type=["csv"])
if clim_file is not None:
    try:
        CLIMATE_OVERRIDE = pd.read_csv(clim_file)
    except Exception:
        CLIMATE_OVERRIDE = None
clim_norm = st.sidebar.checkbox("Normalize uploaded climate to [0,1]", value=True)
CLIMATE_OVERRIDE_NORMALIZE = clim_norm
st.sidebar.subheader("Policy feedbacks")
annual_adapt_invest = st.sidebar.number_input("Annual adaptation investment (€)", min_value=0.0, value=0.0, step=100000.0)
policy_kappa = st.sidebar.slider("Readiness elasticity κ", 0.0, 1.0, 0.10, 0.01)

w_heat = st.sidebar.slider("Heat vs Drought", 0.0, 1.0, 0.5, 0.01)
w_gdp = st.sidebar.slider("Tourism GDP vs Seasonality", 0.0, 1.0, 0.5, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Map Countries (NUTS2)")
country_map = {"Greece": "EL", "Germany": "DE"}
country_picks = st.sidebar.multiselect(
    "Filter map to:", list(country_map.keys()), default=["Greece"]
)
country_codes = [country_map[c] for c in country_picks]

# Live tiny calls (non-blocking, safe)
# Note: External API calls (Open‑Meteo, Eurostat, World Bank) are disabled by default
# to avoid user‑visible warnings when services are unreachable.  Uncomment the
# following lines if you wish to display connectivity samples on the sidebar.
# temp = get_open_meteo_temp()
# if temp is not None:
#     st.sidebar.caption(f"Open‑Meteo sample temp: {temp:.1f}°C")
# _ = get_eurostat_tiny()
# _ = get_worldbank_df()


# ---------- Core Computation ----------
years = np.arange(2025, 2056)
results_by_scenario = {}
for sc in selected_scenarios:
    cf, ts, dd, rb = scenario_params(sc)
    # Include demand_drop from scenario_params to allow tourism demand to decline over time
    results_by_scenario[sc] = compute_series(
        years, cf, ts, rb, w_heat, w_gdp, alpha, beta, gamma,
        demand_drop=dd,
        policy_kappa=policy_kappa, annual_adapt_invest=annual_adapt_invest,
        climate_override=CLIMATE_OVERRIDE, climate_override_normalize=CLIMATE_OVERRIDE_NORMALIZE,
    )
# ---------- Effects panel: show control impacts ----------
with st.expander("Show how controls change the indicators (live)", expanded=False):
    # Policy effect on readiness (analytic)
    policy_boost_end = policy_kappa * (annual_adapt_invest / 1_000_000.0)
    st.caption("Annual adaptation investment raises the readiness path by κ × (investment / 1e6), linearly to the end year (clipped to [0,1]).")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Readiness Δ at 2055 (analytic)", f"{policy_boost_end:+.3f}")
    # Recompute scenarios with policy OFF to show observed Δ in Resilience
    years_no = years.copy()
    results_no_policy = {}
    for sc in selected_scenarios:
        cf, ts, dd, rb = scenario_params(sc)
        results_no_policy[sc] = compute_series(
            years_no, cf, ts, rb, w_heat, w_gdp, alpha, beta, gamma,
            demand_drop=dd,
            policy_kappa=0.0, annual_adapt_invest=0.0,
            climate_override=CLIMATE_OVERRIDE, climate_override_normalize=CLIMATE_OVERRIDE_NORMALIZE,
        )
    # Summarize differences in Resilience due to policy
    import pandas as _pd
    deltas_end = []
    deltas_avg = []
    for sc in selected_scenarios:
        df_on = results_by_scenario.get(sc)
        df_off = results_no_policy.get(sc)
        try:
            d_end = float(df_on.iloc[-1]["Resilience"] - df_off.iloc[-1]["Resilience"])
            d_avg = float((df_on["Resilience"] - df_off["Resilience"]).mean())
        except Exception:
            d_end = 0.0; d_avg = 0.0
        deltas_end.append(d_end)
        deltas_avg.append(d_avg)
    with colB:
        st.metric("Δ Resilience at 2055 (median across scenarios)", f"{(_pd.Series(deltas_end).median() if deltas_end else 0.0):+.3f}")
    with colC:
        st.metric("Δ Resilience (avg across horizon, median)", f"{(_pd.Series(deltas_avg).median() if deltas_avg else 0.0):+.3f}")
    st.caption("Deltas compare current policy settings to κ=0 and investment=0, holding all else constant.")



# ---------- Advanced: Investment cost–benefit analysis ----------
with st.expander("Advanced: Investment cost-benefit analysis"):
    st.markdown(
        "Configure investment, operating, and climate parameters to quantify the "
        "financial and economic performance of the project under each selected scenario."
    )

    min_year = int(years.min())
    max_year = int(years.max())

    with st.form("cba_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            region_cba_start = st.number_input(
                "Start year", min_value=min_year, max_value=max_year, value=min_year, step=1
            )
        with col_b:
            cba_end = st.number_input(
                "End year", min_value=min_year, max_value=max_year, value=min_year + 10, step=1
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            initial_invest = st.number_input(
                "Initial investment (EUR)", min_value=0.0, value=5_000_000.0, step=100_000.0
            )
        with col2:
            base_revenue = st.number_input(
                "Annual revenue in start year (EUR)", value=1_200_000.0, step=50_000.0
            )
        with col3:
            revenue_growth_pct = st.number_input(
                "Annual revenue growth (%)", value=2.0, step=0.5, format="%0.2f"
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            operating_cost = st.number_input(
                "Annual operating cost (EUR)", min_value=0.0, value=350_000.0, step=25_000.0
            )
        with col5:
            discount_rate_pct = st.number_input(
                "Financial discount rate (%)", min_value=0.0, value=6.0, step=0.25, format="%0.2f"
            )
        with col6:
            social_discount_rate_pct = st.number_input(
                "Social discount rate (%)", min_value=0.0, value=4.0, step=0.25, format="%0.2f"
            )

        col7, col8, col9 = st.columns(3)
        with col7:
            resilience_uplift = st.number_input(
                "Revenue uplift per resilience point",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
            )
        with col8:
            risk_exposure_share = st.number_input(
                "Share of revenue exposed to climate risk",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
            )
        with col9:
            adaptation_cost = st.number_input(
                "Additional adaptation O&M (EUR/yr)", min_value=0.0, value=0.0, step=10_000.0
            )

        col10, col11, col12 = st.columns(3)
        with col10:
            annual_emissions = st.number_input(
                "Net emissions in start year (tCO₂e)", value=0.0, step=25.0, format="%0.2f"
            )
        with col11:
            emission_trend_pct = st.number_input(
                "Annual emissions change (%)", value=-2.0, step=0.5, format="%0.2f"
            )
        with col12:
            carbon_price_start = st.number_input(
                "Carbon price in start year (€/tCO₂e)", min_value=0.0, value=80.0, step=5.0
            )

        col13, col14 = st.columns(2)
        with col13:
            carbon_price_growth_pct = st.number_input(
                "Annual carbon price growth (%)", value=2.0, step=0.5, format="%0.2f"
            )
        with col14:
            shadow_factor = st.number_input(
                "Shadow price factor (economic analysis)", min_value=0.0, value=1.0, step=0.05
            )

        run_cba = st.form_submit_button("Run analysis")

    if run_cba:
        if cba_end < region_cba_start:
            st.error("End year must be greater than or equal to the start year.")
        else:
            analysis_years = np.arange(region_cba_start, cba_end + 1)
            revenue_growth = revenue_growth_pct / 100.0
            discount_rate = discount_rate_pct / 100.0
            social_discount_rate = social_discount_rate_pct / 100.0
            emission_trend = emission_trend_pct / 100.0
            carbon_price_growth = carbon_price_growth_pct / 100.0

            if not results_by_scenario:
                st.info("No scenarios selected for comparison.")
            else:
                summary_rows = []
                detail_tables = {}

                for sc, df in results_by_scenario.items():
                    scenario_df = df.reindex(analysis_years).interpolate().ffill().bfill()

                    gross_cf = {}
                    climate_losses = {}
                    emissions = {}
                    yearly_records = []

                    for idx, year in enumerate(analysis_years):
                        resilience_val = float(scenario_df.loc[year, "Resilience"])
                        risk_val = float(scenario_df.loc[year, "Risk"])

                        revenue_nominal = base_revenue * ((1 + revenue_growth) ** idx)
                        revenue_adjusted = revenue_nominal * (
                            1 + resilience_uplift * (resilience_val - 0.5)
                        )
                        revenue_adjusted = max(revenue_adjusted, 0.0)

                        annual_cash = revenue_adjusted - operating_cost - adaptation_cost
                        if year == region_cba_start:
                            annual_cash -= initial_invest

                        gross_cf[year] = annual_cash

                        climate_loss = max(
                            revenue_adjusted * risk_val * risk_exposure_share, 0.0
                        )
                        climate_losses[year] = climate_loss

                        emission_val = annual_emissions * ((1 + emission_trend) ** idx)
                        emissions[year] = emission_val

                        yearly_records.append(
                            {
                                "Year": year,
                                "Resilience": resilience_val,
                                "Risk": risk_val,
                                "Revenue (resilience-adjusted)": revenue_adjusted,
                                "Gross cashflow": annual_cash,
                                "Expected climate loss": climate_loss,
                                "Net emissions (tCO₂e)": emission_val,
                            }
                        )

                    risk_adj_cf = adjust_for_risk(gross_cf, climate_losses)

                    carbon_price_path = {
                        year: carbon_price_start
                        * ((1 + carbon_price_growth) ** (year - region_cba_start))
                        for year in analysis_years
                    }
                    carbon_cf = carbon_cashflow(emissions, carbon_price_path)
                    econ_cf = add_externalities(risk_adj_cf, carbon_cf)
                    econ_shadow = apply_shadow_prices(econ_cf, {"non_traded": shadow_factor})

                    for record in yearly_records:
                        year = record["Year"]
                        record["Risk-adjusted cashflow"] = risk_adj_cf.get(year, 0.0)
                        record["Carbon externality"] = carbon_cf.get(year, 0.0)
                        record["Economic cashflow (shadow)"] = econ_shadow.get(year, 0.0)

                    try:
                        fin_npv = npv(risk_adj_cf, discount_rate)
                    except Exception:
                        fin_npv = float("nan")

                    try:
                        fin_irr = irr(risk_adj_cf) * 100
                    except Exception:
                        fin_irr = float("nan")

                    payback = payback_period(risk_adj_cf)

                    try:
                        econ_npv = enpv(econ_shadow, social_discount_rate)
                    except Exception:
                        econ_npv = float("nan")

                    try:
                        econ_irr = eirr(econ_shadow) * 100
                    except Exception:
                        econ_irr = float("nan")

                    total_benefits = sum(v for v in risk_adj_cf.values() if v > 0)
                    total_costs = -sum(v for v in risk_adj_cf.values() if v < 0)
                    bcr = (total_benefits / total_costs) if total_costs else float("nan")

                    summary_rows.append(
                        {
                            "Scenario": sc,
                            "Financial NPV (€)": fin_npv,
                            "Financial IRR (%)": fin_irr,
                            "Payback (years)": payback,
                            "Benefit-cost ratio": bcr,
                            "Economic NPV (€)": econ_npv,
                            "Economic IRR (%)": econ_irr,
                        }
                    )

                    detail_tables[sc] = pd.DataFrame(yearly_records).set_index("Year")

                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows).set_index("Scenario")
                    st.success("Cost‑benefit analysis complete.")
                    # Show summary table
                    st.dataframe(
                        summary_df.style.format(
                            {
                                "Financial NPV (€)": "{:,.0f}",
                                "Financial IRR (%)": "{:,.2f}",
                                "Payback (years)": "{:.0f}",
                                "Benefit-cost ratio": "{:,.2f}",
                                "Economic NPV (€)": "{:,.0f}",
                                "Economic IRR (%)": "{:,.2f}",
                            }
                        ),
                        use_container_width=True,
                    )
                    # Add a comparative bar chart of NPVs across scenarios
                    try:
                        import plotly.express as px  # type: ignore
                        plot_df = summary_df.reset_index()[
                            ["Scenario", "Financial NPV (€)", "Economic NPV (€)"]
                        ]
                        fig = px.bar(
                            plot_df.melt(
                                id_vars="Scenario",
                                value_vars=["Financial NPV (€)", "Economic NPV (€)"],
                                var_name="Metric",
                                value_name="Value",
                            ),
                            x="Scenario",
                            y="Value",
                            color="Metric",
                            barmode="group",
                            title="Financial vs Economic NPV by scenario",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass
                    # Show detailed yearly tables per scenario in tabs
                    detail_tabs = st.tabs(list(detail_tables.keys()))
                    for tab, sc in zip(detail_tabs, detail_tables.keys()):
                        with tab:
                            st.dataframe(
                                detail_tables[sc].style.format(
                                    {
                                        "Resilience": "{:.3f}",
                                        "Risk": "{:.3f}",
                                        "Revenue (resilience-adjusted)": "{:,.0f}",
                                        "Gross cashflow": "{:,.0f}",
                                        "Expected climate loss": "{:,.0f}",
                                        "Net emissions (tCO₂e)": "{:,.1f}",
                                        "Risk-adjusted cashflow": "{:,.0f}",
                                        "Carbon externality": "{:,.0f}",
                                        "Economic cashflow (shadow)": "{:,.0f}",
                                    }
                                ),
                                use_container_width=True,
                            )
                else:
                    st.info("No results to display for the configured inputs.")

                # ----- Region-level Cost-Benefit Analysis -----
                # Offer an optional region-specific analysis using the same CBA inputs.
                # Users can select a NUTS2 region to tailor the cashflow projections to
                # local exposure, sensitivity and adaptive capacity.  The synthetic
                # scenario trajectories are rescaled to match the region's baseline
                # indicators.  The resulting cashflows, risk and resilience values are
                # then passed through the same financial and economic calculations.
                st.markdown("---")
                st.markdown("### Advanced CBA by Region (NUTS2)")
                # We use the processed real data for region baselines
                df_real_cba, _, _ = load_real_data()
                if df_real_cba is None or df_real_cba.empty:
                    st.info("Processed real data not found; run the preprocessing script to enable region-level CBA.")
                else:
                    # Derive the list of available regions from the processed real data
                    cba_regions = sorted(df_real_cba["_region_norm"].dropna().unique().tolist())
                    if not cba_regions:
                        st.info("No regions available for analysis.")
                    else:
                        # Region selector for region-level CBA
                        region_cba = st.selectbox(
                            "Select region for CBA", cba_regions, index=0, key="cba_region_selector"
                        )
                        # Scenario selection for region-level CBA – default to all selected scenarios
                        region_cba_scenarios = st.multiselect(
                            "Scenarios", list(results_by_scenario.keys()), default=list(results_by_scenario.keys()), key="cba_region_scenarios"
                        )
                        # Automatically run the region-level analysis when one or more scenarios are selected.
                        if region_cba_scenarios:
                            # Compute baseline exposure, sensitivity and adaptive for the region.
                            # We project up to 2055 to ensure a valid baseline exists regardless of the CBA start year.
                            reg_proj = compute_real_projection(
                                df_real_cba,
                                region_cba,
                                scenario="Business",
                                end_year=2055,
                                alpha=alpha,
                                beta=beta,
                                gamma=gamma,
                                w_heat=w_heat,
                                w_gdp=w_gdp,
                                # Anchor the baseline on the last year with observed data (≤2023)
                                baseline_cutoff_year=2023,
                                norm_mode=norm_mode, arrivals_weight=arrivals_weight,
                                policy_kappa=policy_kappa, annual_adapt_invest=annual_adapt_invest
                            )
                            if reg_proj.empty:
                                st.warning("Could not compute baseline for the selected region.")
                            else:
                                # Take baseline values from the first row
                                reg_base = reg_proj.iloc[0]
                                # Default baseline values (used if scenario-specific baseline fails)
                                reg_exp = float(reg_base.get("exposure", 0.5))
                                reg_sen = float(reg_base.get("sensitivity", 0.3))
                                reg_adp = float(reg_base.get("adaptive", 0.4))
                                # Build region-specific scenario trajectories by scaling the synthetic results
                                # according to region baseline.  We normalise the global baseline using
                                # the first row (year) of each synthetic scenario.
                                region_results: Dict[str, pd.DataFrame] = {}
                                for sc in region_cba_scenarios:
                                    if sc not in results_by_scenario:
                                        continue
                                    # Use synthetic results as a template
                                    syn_df = results_by_scenario[sc].copy()
                                    # Compute a scenario-specific baseline for the region using real data.  This
                                    # ensures that scaling reflects the selected scenario rather than always
                                    # anchoring on the Business scenario.  If the real projection fails, we
                                    # retain the default baseline computed above.
                                    try:
                                        proj_base = compute_real_projection(
                                            df_real_cba,
                                            region_cba,
                                            scenario=sc,
                                            end_year=2055,
                                            alpha=alpha,
                                            beta=beta,
                                            gamma=gamma,
                                            w_heat=w_heat,
                                            w_gdp=w_gdp,
                                            baseline_cutoff_year=2023,
                                            norm_mode=norm_mode, arrivals_weight=arrivals_weight,
                                            policy_kappa=policy_kappa, annual_adapt_invest=annual_adapt_invest
                                        )
                                        if proj_base is not None and not proj_base.empty:
                                            base_row = proj_base.iloc[0]
                                            reg_exp = float(base_row.get("exposure", reg_exp))
                                            reg_sen = float(base_row.get("sensitivity", reg_sen))
                                            reg_adp = float(base_row.get("adaptive", reg_adp))
                                    except Exception:
                                        pass
                                    if syn_df.empty:
                                        continue
                                    syn_base = syn_df.iloc[0]
                                    # Compute scale factors; avoid division by zero
                                    scale_exp = reg_exp / syn_base["Exposure"] if syn_base["Exposure"] > 0 else 1.0
                                    scale_sen = reg_sen / syn_base["Sensitivity"] if syn_base["Sensitivity"] > 0 else 1.0
                                    scale_adp = reg_adp / syn_base["Adaptive"] if syn_base["Adaptive"] > 0 else 1.0
                                    df_scaled = syn_df.copy()
                                    df_scaled["Exposure"] = df_scaled["Exposure"] * scale_exp
                                    df_scaled["Sensitivity"] = df_scaled["Sensitivity"] * scale_sen
                                    df_scaled["Adaptive"] = df_scaled["Adaptive"] * scale_adp
                                    # Recompute resilience and risk with the weights
                                    a_n, b_n, g_n = normalize_weights(alpha, beta, gamma)
                                    df_scaled["Resilience"] = np.clip(
                                        a_n * (1.0 - df_scaled["Exposure"]) + b_n * (1.0 - df_scaled["Sensitivity"]) + g_n * df_scaled["Adaptive"],
                                        0,
                                        1,
                                    )
                                    df_scaled["Risk"] = np.clip(
                                        a_n * df_scaled["Exposure"] + b_n * df_scaled["Sensitivity"] + g_n * (1.0 - df_scaled["Adaptive"]), 0, 1
                                    )
                                    region_results[sc] = df_scaled
                                # Perform CBA on region_results using the same parameters.
                                if not region_results:
                                    st.info("No regional results to analyse.")
                                else:
                                    # Allow users to specify start and end years for the region-level analysis.
                                    # Use the same defaults as the global analysis (2025–2055) if undefined.
                                    # When running on a fresh page without the main CBA form, ``region_cba_start`` and
                                    # ``cba_end`` may not yet be defined; fall back to the overall available years.
                                    region_start_default = int(years.min())
                                    region_end_default = int(years.max())
                                    try:
                                        region_start_default = int(region_cba_start)
                                    except Exception:
                                        pass
                                    try:
                                        region_end_default = int(cba_end)
                                    except Exception:
                                        pass
                                    col_rs, col_re = st.columns(2)
                                    with col_rs:
                                        region_region_cba_start = st.number_input(
                                            "Region CBA start year",
                                            min_value=int(years.min()),
                                            max_value=int(years.max()),
                                            value=region_start_default,
                                            step=1,
                                            key="region_region_cba_start",
                                        )
                                    with col_re:
                                        region_cba_end = st.number_input(
                                            "Region CBA end year",
                                            min_value=int(region_region_cba_start),
                                            max_value=int(years.max()),
                                            value=region_end_default,
                                            step=1,
                                            key="region_cba_end",
                                        )
                                    region_analysis_years = np.arange(region_region_cba_start, region_cba_end + 1)
                                    # Compute economic parameters for the region-level analysis.
                                    region_revenue_growth = revenue_growth_pct / 100.0
                                    region_discount_rate = discount_rate_pct / 100.0
                                    region_social_discount = social_discount_rate_pct / 100.0
                                    region_emission_trend = emission_trend_pct / 100.0
                                    region_carbon_growth = carbon_price_growth_pct / 100.0
                                    region_summary: List[Dict[str, float]] = []
                                    region_details: Dict[str, pd.DataFrame] = {}
                                    for sc, df_r in region_results.items():
                                        # Interpolate the region results to the region analysis years
                                        scenario_df = df_r.reindex(region_analysis_years).interpolate().ffill().bfill()
                                        gross_cf: Dict[int, float] = {}
                                        climate_losses: Dict[int, float] = {}
                                        emissions_r: Dict[int, float] = {}
                                        annual_records: List[Dict[str, float]] = []
                                        for idx, year in enumerate(region_analysis_years):
                                            res_val = float(scenario_df.loc[year, "Resilience"])
                                            risk_val = float(scenario_df.loc[year, "Risk"])
                                            revenue_nominal = base_revenue * ((1 + region_revenue_growth) ** idx)
                                            revenue_adjusted = revenue_nominal * (1 + resilience_uplift * (res_val - 0.5))
                                            revenue_adjusted = max(revenue_adjusted, 0.0)
                                            annual_cash = revenue_adjusted - operating_cost - adaptation_cost
                                            if year == region_cba_start:
                                                annual_cash -= initial_invest
                                            gross_cf[year] = annual_cash
                                            climate_loss = max(revenue_adjusted * risk_val * risk_exposure_share, 0.0)
                                            climate_losses[year] = climate_loss
                                            emission_val = annual_emissions * ((1 + region_emission_trend) ** idx)
                                            emissions_r[year] = emission_val
                                            annual_records.append(
                                                {
                                                    "Year": year,
                                                    "Resilience": res_val,
                                                    "Risk": risk_val,
                                                    "Revenue (resilience-adjusted)": revenue_adjusted,
                                                    "Gross cashflow": annual_cash,
                                                    "Expected climate loss": climate_loss,
                                                    "Net emissions (tCO₂e)": emission_val,
                                                }
                                            )
                                        # Risk-adjusted cashflow and externalities
                                        risk_adj_cf = adjust_for_risk(gross_cf, climate_losses)
                                        carbon_price_path = {
                                            year: carbon_price_start * ((1 + region_carbon_growth) ** (year - region_cba_start))
                                            for year in region_analysis_years
                                        }
                                        carbon_cf_r = carbon_cashflow(emissions_r, carbon_price_path)
                                        econ_cf_r = add_externalities(risk_adj_cf, carbon_cf_r)
                                        econ_shadow_r = apply_shadow_prices(econ_cf_r, {"non_traded": shadow_factor})
                                        # Add computed series back to the records for display
                                        for rec in annual_records:
                                            y = rec["Year"]
                                            rec["Risk-adjusted cashflow"] = risk_adj_cf.get(y, 0.0)
                                            rec["Carbon externality"] = carbon_cf_r.get(y, 0.0)
                                            rec["Economic cashflow (shadow)"] = econ_shadow_r.get(y, 0.0)
                                        # Compute summary metrics
                                        try:
                                            fin_npv_r = npv(risk_adj_cf, region_discount_rate)
                                        except Exception:
                                            fin_npv_r = float("nan")
                                        try:
                                            fin_irr_r = irr(risk_adj_cf) * 100
                                        except Exception:
                                            fin_irr_r = float("nan")
                                        payback_r = payback_period(risk_adj_cf)
                                        try:
                                            econ_npv_r = enpv(econ_shadow_r, region_social_discount)
                                        except Exception:
                                            econ_npv_r = float("nan")
                                        try:
                                            econ_irr_r = eirr(econ_shadow_r) * 100
                                        except Exception:
                                            econ_irr_r = float("nan")
                                        total_benefits_r = sum(v for v in risk_adj_cf.values() if v > 0)
                                        total_costs_r = -sum(v for v in risk_adj_cf.values() if v < 0)
                                        bcr_r = (total_benefits_r / total_costs_r) if total_costs_r else float("nan")
                                        region_summary.append(
                                            {
                                                "Scenario": sc,
                                                "Financial NPV (€)": fin_npv_r,
                                                "Financial IRR (%)": fin_irr_r,
                                                "Payback (years)": payback_r,
                                                "Benefit-cost ratio": bcr_r,
                                                "Economic NPV (€)": econ_npv_r,
                                                "Economic IRR (%)": econ_irr_r,
                                            }
                                        )
                                        region_details[sc] = pd.DataFrame(annual_records).set_index("Year")
                                    # Display region-level summary and details
                                    if region_summary:
                                        st.markdown(f"#### Results for {region_cba}")
                                        reg_sum_df = pd.DataFrame(region_summary).set_index("Scenario")
                                        st.dataframe(
                                            reg_sum_df.style.format(
                                                {
                                                    "Financial NPV (€)": "{:,.0f}",
                                                    "Financial IRR (%)": "{:,.2f}",
                                                    "Payback (years)": "{:.0f}",
                                                    "Benefit-cost ratio": "{:.2f}",
                                                    "Economic NPV (€)": "{:,.0f}",
                                                    "Economic IRR (%)": "{:,.2f}",
                                                }
                                            ),
                                            use_container_width=True,
                                        )
                                        # Plot comparative NPVs for region
                                        try:
                                            import plotly.express as px  # type: ignore
                                            plot_df_r = reg_sum_df.reset_index()[
                                                ["Scenario", "Financial NPV (€)", "Economic NPV (€)"]
                                            ]
                                            fig_r = px.bar(
                                                plot_df_r.melt(
                                                    id_vars="Scenario",
                                                    value_vars=["Financial NPV (€)", "Economic NPV (€)"],
                                                    var_name="Metric",
                                                    value_name="Value",
                                                ),
                                                x="Scenario",
                                                y="Value",
                                                color="Metric",
                                                barmode="group",
                                                title=f"Financial vs Economic NPV by scenario for {region_cba}",
                                            )
                                            # Position the legend horizontally below the plot and increase bottom margin
                                            fig_r.update_layout(
                                                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                                                margin=dict(b=80)
                                            )
                                            st.plotly_chart(fig_r, use_container_width=True)
                                        except Exception:
                                            pass

                                        # Plot cashflow trajectories for each scenario
                                        try:
                                            import plotly.express as px  # type: ignore
                                            # Combine cashflows across scenarios for a multi-line chart
                                            cf_list = []
                                            for sc_key, df_cash in region_details.items():
                                                # Ensure index is year
                                                cash_df = df_cash.copy()
                                                cash_df = cash_df.reset_index()
                                                if 'Year' not in cash_df.columns:
                                                    cash_df['Year'] = cash_df.index
                                                # Use Risk-adjusted cashflow as primary indicator
                                                cf_sel = cash_df[["Year", "Risk-adjusted cashflow"]].copy()
                                                cf_sel["Scenario"] = sc_key
                                                cf_list.append(cf_sel)
                                            if cf_list:
                                                cf_plot_df = pd.concat(cf_list, ignore_index=True)
                                                fig_cf = px.line(
                                                    cf_plot_df,
                                                    x="Year",
                                                    y="Risk-adjusted cashflow",
                                                    color="Scenario",
                                                    title=f"Risk-adjusted cashflow over time for {region_cba}",
                                                    markers=True,
                                                )
                                                # Adjust legend position below
                                                fig_cf.update_layout(
                                                    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                                                    margin=dict(b=80)
                                                )
                                                st.plotly_chart(fig_cf, use_container_width=True)
                                        except Exception:
                                            pass
                                        # Detailed tables per scenario
                                        region_tabs = st.tabs(list(region_details.keys()))
                                        for tab, sc_name in zip(region_tabs, region_details.keys()):
                                            with tab:
                                                st.dataframe(
                                                    region_details[sc_name].style.format(
                                                        {
                                                            "Resilience": "{:.3f}",
                                                            "Risk": "{:.3f}",
                                                            "Revenue (resilience-adjusted)": "{:,.0f}",
                                                            "Gross cashflow": "{:,.0f}",
                                                            "Expected climate loss": "{:,.0f}",
                                                            "Net emissions (tCO₂e)": "{:,.1f}",
                                                            "Risk-adjusted cashflow": "{:,.0f}",
                                                            "Carbon externality": "{:,.0f}",
                                                            "Economic cashflow (shadow)": "{:,.0f}",
                                                        }
                                                    ),
                                                    use_container_width=True,
                                                )


# ---------- Visuals: Time-series ----------
st.subheader("Resilience projection (2025–2055)")
plot_df = pd.concat(
    {sc: df["Resilience"] for sc, df in results_by_scenario.items()}, axis=1
)
st.line_chart(plot_df)


# ---------- Tables ----------
st.subheader("Pillar breakdown (select scenario)")
# assign a unique key to avoid duplicate element IDs
table_choice = st.selectbox("Scenario table", list(results_by_scenario.keys()), key="scenario_table")
st.dataframe(results_by_scenario[table_choice])


# Guardrail check (per the selected table)
last_row = results_by_scenario[table_choice].iloc[-1]
cf, ts, dd, rb = scenario_params(table_choice)
if ts > 0.5 and dd > 20:
    st.warning("⚠️ Guardrail: Tourism GDP share > 50% and projected demand drop > 20%.")


# ---------- Download ----------
st.download_button(
    "Download current table (CSV)",
    data=results_by_scenario[table_choice].reset_index().to_csv(index=False).encode("utf-8"),
    file_name=f"crisi_{table_choice.split()[0].lower()}_results.csv",
    mime="text/csv",
)


# ---------- Placeholder for ML tourism-impact signal ----------
# The original prototype included a demonstration of a RandomForest model trained
# on synthetic data to illustrate feature importances for tourism impacts.  As
# this demo does not rely on real-world data and may confuse users, it has
# been removed.  A future version of the app could integrate a trained model
# using actual climate and tourism indicators to provide predictive insights.


# ---------- Map ----------
st.subheader("Resilience maps by scenario and horizon")
nuts2 = get_nuts2_gdf()
if nuts2 is not None:
    # Filter by selected countries (if any)
    if country_codes:
        nuts2 = nuts2[nuts2["CNTR_CODE"].isin(country_codes)]
    # User chooses scenario and horizon year for the map
    # use unique keys for selectbox widgets in the map section
    map_sc = st.selectbox("Scenario (map)", selected_scenarios, index=0, key="scenario_map")
    map_year = st.selectbox("Projection year", [2035, 2045, 2055], index=1, key="map_year")
    base_df = results_by_scenario.get(map_sc)
    if base_df is None or map_year not in base_df.index:
        st.info("Selected year is outside the computed range; please choose another year.")
    else:
        row = base_df.loc[map_year]
        # Produce per‑region resilience and underlying indices based on real data
        # rather than random factors.  We attempt to compute exposure,
        # sensitivity, adaptive and resilience for each NUTS2 region using
        # ``compute_real_projection`` and the processed dataset.  If real
        # projections are unavailable for a region, fall back to scaled
        # synthetic values with a small random perturbation.
        nuts2_map = nuts2.copy()
        # Load processed real data once
        df_real_map, _, _ = load_real_data()
        # Precompute a mapping from region names to the normalised key used in df_real
        def translit(val: str) -> str:
            return (
                str(val)
                .strip()
                .encode("ascii", "ignore")
                .decode("ascii")
                .upper()
            )
        # Determine the scenario key for compute_real_projection (use first word)
        sc_key = map_sc.split()[0] if isinstance(map_sc, str) else map_sc
        # Loop over each region in the GeoDataFrame
        exposures_vals = []
        sensitivities_vals = []
        adaptives_vals = []
        resiliences_vals = []
        # Random generator for fallback heterogeneity
        seed_val = _stable_seed((map_sc, map_year, "map"))
        rng = np.random.default_rng(seed_val)
        for idx_n, row_map in nuts2_map.iterrows():
            reg_name = row_map.get("region_name", "")
            norm_name = translit(reg_name)
            # Attempt to compute projection for this region
            e_val = s_val = a_val = r_val = None
            if df_real_map is not None and not df_real_map.empty and norm_name in df_real_map["_region_norm"].values:
                try:
                    proj = compute_real_projection(
                        df_real_map,
                        norm_name,
                        scenario=sc_key,
                        end_year=map_year,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        w_heat=w_heat,
                        w_gdp=w_gdp,
                        baseline_cutoff_year=2023, norm_mode=norm_mode, arrivals_weight=arrivals_weight, policy_kappa=policy_kappa, annual_adapt_invest=annual_adapt_invest,
                    )
                    if proj is not None and not proj.empty and map_year in proj["year"].values:
                        row_reg = proj[proj["year"] == map_year].iloc[0]
                        e_val = float(row_reg.get("exposure", np.nan))
                        s_val = float(row_reg.get("sensitivity", np.nan))
                        a_val = float(row_reg.get("adaptive", np.nan))
                        r_val = float(row_reg.get("resilience", np.nan))
                except Exception:
                    pass
            # If any value is missing, fall back to the synthetic scenario values with
            # small random heterogeneity
            if e_val is None or s_val is None or a_val is None or r_val is None:
                # Random factors within ±20% to reflect regional heterogeneity
                factor_e = float(rng.uniform(0.8, 1.2))
                factor_s = float(rng.uniform(0.8, 1.2))
                factor_a = float(rng.uniform(0.8, 1.2))
                factor_r = float(rng.uniform(0.8, 1.2))
                e_val = float(np.clip(row["Exposure"] * factor_e, 0, 1))
                s_val = float(np.clip(row["Sensitivity"] * factor_s, 0, 1))
                a_val = float(np.clip(row["Adaptive"] * factor_a, 0, 1))
                r_val = float(np.clip(row["Resilience"] * factor_r, 0, 1))
            exposures_vals.append(e_val)
            sensitivities_vals.append(s_val)
            adaptives_vals.append(a_val)
            resiliences_vals.append(r_val)
        nuts2_map["Exposure"] = exposures_vals
        nuts2_map["Sensitivity"] = sensitivities_vals
        nuts2_map["Adaptive"] = adaptives_vals
        nuts2_map["Resilience"] = resiliences_vals
        # Centre map around the selected regions
        centroid = nuts2_map.geometry.centroid
        center = [centroid.y.mean(), centroid.x.mean()]
        fmap = folium.Map(location=center, zoom_start=5)
        # Add choropleth by resilience
        folium.Choropleth(
            geo_data=nuts2_map.__geo_interface__,
            data=nuts2_map,
            columns=["id", "Resilience"],
            key_on="feature.properties.id",
            fill_color="YlGn",
            fill_opacity=0.8,
            line_opacity=0.3,
            legend_name="Resilience score",
        ).add_to(fmap)
        # Add rich tooltips showing underlying indices for each region
        folium.GeoJson(
            data=nuts2_map,
            name="Region detail",
            tooltip=folium.features.GeoJsonTooltip(
                fields=["region_name", "Resilience", "Exposure", "Sensitivity", "Adaptive"],
                aliases=["Region", "Resilience", "Exposure", "Sensitivity", "Adaptive"],
                localize=True,
                sticky=False,
                labels=True,
            ),
            style_function=lambda feature: {
                "fillColor": "#00000000",
                "color": "#00000000",
                "weight": 0,
            },
        ).add_to(fmap)
        st_folium(fmap, width=800, height=520)
else:
    st.info("NUTS2 map unavailable right now.")


st.caption(
    "Notes: Copernicus CDS calls require a valid ~/.cdsapirc. Eurostat/World Bank are pinged lightly; "
    "full dataset wiring is straightforward once indicator codes are finalized."
)


# ---------- Corine Land Cover dataset ----------
# ----- Land Cover Analysis (Corine) -----
# The Corine Land Cover analysis is not currently supported in this environment because
# loading large shapefiles or FileGDB archives requires a full GeoPandas/GDAL stack, which
# may not be installed.  Instead of presenting a broken interface, we hide this section
# and display an informative message.  If you have a working CLC dataset, you can
# customise this section to load your own data.
with st.expander("Land Cover Analysis (Corine)", expanded=False):
    st.info(
        "Corine Land Cover analysis is not available in this deployment. Please process the CLC data offline "
        "or on a platform with GeoPandas and GDAL support."
    )

# ---------- Real-data analysis section ----------
with st.expander("Real Data Analysis (Greece)", expanded=False):
    """
    This section reads the processed Greece datasets (if available) and allows
    users to explore real metrics across regions and years. Metrics are
    discovered automatically from the processed Parquet file and can be
    visualised as time-series line charts. To prepare the data, run
    ``python scripts/prepare_real_data.py`` after placing the raw files in
    ``dataset``.
    """
    df_real, clc_sum, metrics = load_real_data()
    if df_real is None or df_real.empty:
        st.info(
            "Processed real data not found. Please run the preprocessing script "
            "(`scripts/prepare_real_data.py`) to generate `data/processed/crisi_greece_processed.parquet`."
        )
    else:
        # Region selection
        regions = sorted([r for r in df_real["_region_norm"].dropna().unique() if str(r).strip().upper() not in {"W","GREECE"}])
        if not regions:
            st.warning("No regions detected in real data.")
        else:
            # ensure the region selector for real data analysis has a unique key
            sel_region = st.selectbox("Region", regions, index=0, key="real_data_region")
            sdf = df_real[df_real["_region_norm"] == sel_region]
            # Build the list of metrics from numeric columns (excluding region and year)
            if not metrics:
                metrics = [
                    c
                    for c in sdf.columns
                    if c not in ["_region_norm", "year"] and pd.api.types.is_numeric_dtype(sdf[c])
                ]
            # Metric selection (first choose metric, then compute year range based on data availability)
            metric = st.selectbox("Metric", metrics, index=0, key="real_data_metric")
            # Filter out rows where the selected metric is NaN to avoid displaying None in the table
            sdf = sdf[sdf[metric].notna()]
            # Determine the year range based on non-null values of the selected metric to avoid extreme years with no data
            if not sdf.empty:
                year_min = int(sdf["year"].min())
                year_max = int(sdf["year"].max())
            else:
                year_min = int(df_real["year"].min())
                year_max = int(df_real["year"].max())
            # Limit the displayed year range to the study horizon (e.g. 2000–2055) to avoid
            # showing placeholder years beyond the projection horizon (like 2100).
            year_min = max(year_min, 2000)
            year_max = min(year_max, 2055)
            year_range = st.slider(
                "Year range",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max),
                step=1,
            )
            # Apply year filter
            sdf = sdf[(sdf["year"] >= year_range[0]) & (sdf["year"] <= year_range[1])]
            # Display table
            st.markdown(f"### {metric} for {sel_region}")
            st.dataframe(
                sdf[["year", metric]].set_index("year").rename(columns={metric: metric}),
                use_container_width=True,
            )
            # Determine the unit for the selected metric.  We strip dataset
            # prefixes (e.g. ``main__`` or ``tourism__``) before looking up the
            # unit in the VARIABLE_UNITS dictionary.  If no match is found,
            # the unit is left unspecified.
            base_var = metric.split("__")[-1] if "__" in metric else metric
            unit = VARIABLE_UNITS.get(base_var, "")
            # Chart
            try:
                import plotly.express as px  # type: ignore
                fig = px.line(
                    sdf.sort_values("year"),
                    x="year",
                    y=metric,
                    markers=True,
                    labels={"year": "Year", metric: f"{metric} ({unit})" if unit else metric},
                    title=f"{metric} over time",
                )
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                    margin=dict(b=80),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.line_chart(sdf.set_index("year")[metric])
            # Display the unit (if available) below the chart and the data source & disclaimer
            if unit:
                st.caption(f"Unit: {unit}")
            # Sources and disclaimer
            st.caption(
                "Sources: Eurostat, World Bank, IPCC, UNWTO, ELSTAT (processed by developers)."
            )
            st.caption(
                "Disclaimer: Projections are generated using a hierarchical time-series model with shrinkage, seasonality and COVID‑19 recovery adjustments."
            )

# ---------- Scenario projections using real data (renamed) ----------
with st.expander("Scenario plausible projections (to 2055)", expanded=False):
    """
    Use the processed real data as a baseline to project exposure, sensitivity,
    adaptive capacity and resilience up to 2055 for each scenario.  The
    projections are computed from the last available year for the selected
    region, applying scenario-specific growth rates for hazards, tourism
    share and adaptive capacity.  Results are shown in tabular form and
    plotted over time.  Run ``scripts/prepare_real_data.py`` first to
    generate ``data/processed/crisi_greece_processed.parquet`` or ``.csv``.
    """
    df_real_proj, _, _ = load_real_data()
    if df_real_proj is None or df_real_proj.empty:
        st.info("Processed real data not found. Please run the preprocessing script.")
    else:
        regs = sorted([r for r in df_real_proj["_region_norm"].dropna().unique()])
        if not regs:
            st.warning("No regions detected in real data.")
        else:
            # Assign unique keys for region and scenario selectors in the projection section
            reg = st.selectbox("Region", regs, index=0, key="real_proj_region")
            scen_names = ["Green", "Business", "Divided", "Techno", "Regional"]
            scen = st.selectbox("Scenario", scen_names, index=0, key="real_proj_scenario")
            end_year = st.slider("Projection horizon (final year)", 2035, 2055, 2055, step=5)
            # Pass in the user-selected pillar weights to the projection function
            proj_df = compute_real_projection(df_real_proj, reg, scen,
                end_year=end_year,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                w_heat=w_heat,
                w_gdp=w_gdp,
                baseline_cutoff_year=2023,
                norm_mode=norm_mode, arrivals_weight=arrivals_weight, policy_kappa=policy_kappa, annual_adapt_invest=annual_adapt_invest)
            if proj_df.empty:
                st.warning("Projection could not be computed for the selected region.")
            else:

                # Explain normalization & arrivals-weight effects
                with st.expander("Explain control effects on real-data projection", expanded=False):
                    # Normalization constants
                    try:
                        max_edu, max_rnd, max_gdp_pc = compute_norm_constants(df_real_proj, mode=norm_mode, year_ref=2019, region=reg)
                        st.caption(f"Normalization mode: **{norm_mode}** → education={max_edu:.3f}, R&D={max_rnd:.3f}, GDPpc={max_gdp_pc:.2f}")
                    except Exception:
                        st.caption(f"Normalization mode: **{norm_mode}**")
                    # Sensitivity baseline difference from arrivals weight (η)
                    try:
                        proj_df_no_arr = compute_real_projection(
                            df_real_proj, reg, scen, end_year=end_year,
                            alpha=alpha, beta=beta, gamma=gamma,
                            w_heat=w_heat, w_gdp=w_gdp,
                            baseline_cutoff_year=2023,
                            norm_mode=norm_mode, arrivals_weight=0.0,
                            policy_kappa=policy_kappa, annual_adapt_invest=annual_adapt_invest
                        )
                        if not proj_df_no_arr.empty:
                            s0_with = float(proj_df.iloc[0]["sensitivity"])
                            s0_wo   = float(proj_df_no_arr.iloc[0]["sensitivity"])
                            st.metric("Sensitivity baseline Δ (η effect)", f"{(s0_with - s0_wo):+.3f}")
                    except Exception:
                        pass
                # Show the projected table, including real variables (Tmax, tourism share, gdp_pc)
                st.dataframe(proj_df, use_container_width=True)
                try:
                    import plotly.graph_objects as go  # type: ignore
                    from plotly.subplots import make_subplots  # type: ignore

                    # Create a figure with a secondary y‑axis for resilience
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(
                        go.Scatter(x=proj_df["year"], y=proj_df["exposure"], name="Exposure"),
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(x=proj_df["year"], y=proj_df["sensitivity"], name="Sensitivity"),
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(x=proj_df["year"], y=proj_df["adaptive"], name="Adaptive"),
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(x=proj_df["year"], y=proj_df["resilience"], name="Resilience", line=dict(dash="dash")),
                        secondary_y=True,
                    )
                    # Axis labels
                    fig.update_xaxes(title_text="Year")
                    fig.update_yaxes(title_text="Index (0–1)", secondary_y=False)
                    fig.update_yaxes(title_text="Resilience", secondary_y=True)
                    fig.update_layout(
                        title=f"Projection for {reg} under {scen} scenario",
                        # Position the legend below the plot area and increase bottom margin to avoid overlap
                        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                        margin=dict(b=80),
                    )
                    st.plotly_chart(fig, use_container_width=True)


                    # (Removed) Projected real variables chart hidden per user request.

                    # Plot resilience on its own axis for clarity
                    if "resilience" in proj_df.columns:

                        # Compute policy-OFF projection for overlay
                        try:
                            proj_df_off = compute_real_projection(
                                df_real_proj, reg, scen, end_year=end_year,
                                alpha=alpha, beta=beta, gamma=gamma,
                                w_heat=w_heat, w_gdp=w_gdp,
                                baseline_cutoff_year=2023,
                                norm_mode=norm_mode, arrivals_weight=arrivals_weight,
                                policy_kappa=0.0, annual_adapt_invest=0.0
                            )
                        except Exception:
                            proj_df_off = None

                        fig3 = go.Figure()
                        fig3.add_trace(go.Scatter(x=proj_df["year"], y=proj_df["resilience"], name="Resilience (policy on)", line=dict(color="#e63946")))
                        if proj_df_off is not None and not proj_df_off.empty:
                            fig3.add_trace(go.Scatter(x=proj_df_off["year"], y=proj_df_off["resilience"], name="Resilience (policy off)", line=dict(dash="dot")))
                        fig3.update_layout(
                            title=f"Resilience for {reg} under {scen}",
                            xaxis_title="Year",
                            yaxis_title="Resilience",
                            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                            margin=dict(b=80),
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                except Exception:
                    # Fallback: simple line chart of indices only
                    st.line_chart(proj_df.set_index("year")[["exposure", "sensitivity", "adaptive", "resilience"]])