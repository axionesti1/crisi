"""Scoring functions for the CRISI pilot.

This module implements a simple normalized scoring algorithm for
climate‑tourism risk and resilience. The design follows the
methodology laid out in the project specification: indicators are
grouped into pillars (exposure, sensitivity, adaptive capacity).
Each indicator is normalized and weighted. The risk score is a
linear combination of the pillars, and the resilience score is
defined as 1 − risk. An optional priority flag can be added
according to hybrid decision rules (e.g. when tourism dependency
and projected drop exceed thresholds).

The indicator definitions and weights are configured via a YAML
file. See ``config/indicators.yaml`` for an example.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from typing import Dict, Any


def _normalize(series: pd.Series, *, higher_is_worse: bool = True) -> pd.Series:
    """Normalize a numeric series to the interval [0, 1].

    The normalization trims extreme outliers by clipping to the 2nd
    and 98th percentiles before scaling. When ``higher_is_worse`` is
    True the series is scaled directly (higher values map to larger
    normalized scores). When False the scaling is inverted so that
    higher values map to lower normalized scores.

    Parameters
    ----------
    series : pandas.Series
        The series of numeric values to normalize.
    higher_is_worse : bool, default True
        Whether a larger raw value corresponds to a worse outcome.

    Returns
    -------
    pandas.Series
        A series of normalized values in [0, 1].
    """
    # Convert to float and handle missing values gracefully
    s = series.astype(float).copy()
    # Compute lower and upper bounds using robust percentiles
    lo = np.nanpercentile(s, 2)
    hi = np.nanpercentile(s, 98)
    # Clip outliers to avoid extreme effects
    s = s.clip(lo, hi)
    # Scale to [0,1]; add a small epsilon to avoid division by zero
    eps = 1e-9
    normalized = (s - lo) / (hi - lo + eps)
    if not higher_is_worse:
        normalized = 1.0 - normalized
    return normalized


def _pillar_score(df: pd.DataFrame, items: Dict[str, Dict[str, Any]]) -> pd.Series:
    """Compute the weighted normalized score for a pillar.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing raw indicator columns.
    items : dict
        Mapping of indicator names to metadata dicts. Each dict must
        contain keys ``direction`` ("higher_is_worse" or
        "lower_is_worse") and ``weight`` (numeric).

    Returns
    -------
    pandas.Series
        The pillar score for each row.
    """
    scores = []
    weights = []
    for col, meta in items.items():
        if col not in df.columns:
            # Skip missing columns gracefully
            continue
        direction = meta.get("direction", "higher_is_worse")
        higher_is_worse = direction == "higher_is_worse"
        weight = float(meta.get("weight", 1.0))
        normalized = _normalize(df[col], higher_is_worse=higher_is_worse)
        scores.append(normalized * weight)
        weights.append(weight)
    if not scores:
        # No indicators provided; return zeros
        return pd.Series(0.0, index=df.index)
    total_weight = np.sum(weights)
    stacked = np.vstack([s.values for s in scores])
    # Weighted average across indicators
    return pd.Series(np.sum(stacked, axis=0) / (total_weight + 1e-9), index=df.index)


def compute_scores(
    df: pd.DataFrame,
    *,
    config_path: str = "config/indicators.yaml",
) -> pd.DataFrame:
    """Compute risk and resilience scores for each row.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing raw indicator columns and identifying
        columns such as region_id and scenario.
    config_path : str, default "config/indicators.yaml"
        Path to a YAML file describing indicator weights, directions
        and pillar weights.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing identifying columns plus columns
        ``risk``, ``resilience_score`` and ``priority_flag``.
    """
    cfg: Dict[str, Any] = yaml.safe_load(open(config_path, "r"))

    # Compute pillar-level scores
    E = _pillar_score(df, cfg.get("exposure", {}))
    S = _pillar_score(df, cfg.get("sensitivity", {}))
    AC = _pillar_score(df, cfg.get("adaptive_capacity", {}))

    # Retrieve pillar weights
    pillars_cfg = cfg.get("pillars", {})
    alpha = float(pillars_cfg.get("alpha", 0.4))
    beta = float(pillars_cfg.get("beta", 0.4))
    gamma = float(pillars_cfg.get("gamma", 0.2))

    # Compute risk as weighted sum of pillars, invert adaptive capacity
    risk = alpha * E + beta * S - gamma * AC
    risk = risk.clip(0.0, 1.0)
    # Resilience score is 1 - risk scaled to 100
    resilience_score = (1.0 - risk) * 100.0

    # Compose output DataFrame
    out_cols = []
    for col in ["region_id", "region_name", "scenario"]:
        if col in df.columns:
            out_cols.append(col)
    out = df[out_cols].copy()
    out["risk"] = risk
    out["resilience_score"] = resilience_score

    # Apply hybrid rule for priority flag
    priority_cfg = cfg.get("priority_rule", {})
    if priority_cfg:
        tg_thr = float(priority_cfg.get("tourism_gdp_share_gt", 0.0))
        drop_thr = float(priority_cfg.get("projected_drop_gt", 0.0))
        tourism_gdp = df.get("tourism_gdp_share", pd.Series(0.0, index=df.index))
        projected_drop = df.get("projected_drop", pd.Series(0.0, index=df.index))
        flag = (tourism_gdp >= tg_thr) & (projected_drop >= drop_thr)
        out["priority_flag"] = flag
    else:
        out["priority_flag"] = False

    return out


__all__ = ["compute_scores"]