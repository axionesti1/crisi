"""ETL pipeline for CRISI tourism-climate analytics MVP.

The script orchestrates the following steps:
- Download and clean GISCO NUTS-2 boundaries.
- Retrieve Eurostat tourism and GDP indicators with pandaSDMX.
- Retrieve Copernicus CDS climate projections and aggregate to NUTS-2.
- Join and validate the canonical feature table, exporting GeoParquet.

Run with ``python -m flows.etl``. See README for CLI flags.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

# ✅ Use the non-deprecated Pandera import path (NumPy 2.0 compatible in recent versions)
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

HAS_PANDERA = True
try:
    import pandera.pandas as pa  # noqa: F401
    from pandera.pandas import Column, DataFrameSchema, Check
except Exception as _pandera_exc:
    HAS_PANDERA = False
    Column = DataFrameSchema = Check = object  # type: ignore
    import logging as _lg
    _lg.getLogger(__name__).warning("Pandera unavailable: %s — schema validation will be skipped.", _pandera_exc)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


DATA_DIR = project_root() / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = project_root() / "config"


@dataclass
class CliArgs:
    pull_cds: bool = False
    pull_eurostat: bool = False
    rebuild: bool = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    for folder in [RAW_DIR / "cds", RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    LOGGER.debug("Ensured data directories exist under %s", DATA_DIR)


def read_yaml(path: Path) -> dict:
    import yaml  # deferred import
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_geo_dataframe(df: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    LOGGER.info("Wrote %s rows to %s", len(df), path)


def _pick(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
    """Return the first matching column name by case-insensitive search."""
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


# ---------------------------------------------------------------------------
# GISCO NUTS-2 downloader
# ---------------------------------------------------------------------------

def download_nuts2(force: bool = False) -> gpd.GeoDataFrame:
    """Download GISCO NUTS-2 polygons and store locally.

    Uses the official GISCO distribution endpoint.
    https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/
    """
    target_path = INTERIM_DIR / "nuts2.parquet"
    if target_path.exists() and not force:
        LOGGER.info("Loading cached NUTS-2 dataset from %s", target_path)
        return gpd.read_parquet(target_path)

    # Prefer GeoJSON (streams well); fall back to download GPKG locally.
    geojson_url = (
        "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
        "NUTS_RG_01M_2024_4326.geojson"
    )
    LOGGER.info("Requesting NUTS-2 polygons from %s", geojson_url)

    try:
        gdf = gpd.read_file(geojson_url)
        LOGGER.info("Fetched %s rows from GISCO (GeoJSON)", len(gdf))
    except Exception as exc_json:
        LOGGER.warning("GeoJSON read failed (%s), trying GPKG download...", exc_json)
        try:
            import requests
            gpkg_url = (
                "https://gisco-services.ec.europa.eu/distribution/v2/nuts/gpkg/"
                "NUTS_RG_01M_2024_4326.gpkg"
            )
            r = requests.get(gpkg_url, timeout=60)
            r.raise_for_status()
            local = RAW_DIR / "NUTS_RG_01M_2024_4326.gpkg"
            local.write_bytes(r.content)
            gdf = gpd.read_file(local, layer="NUTS_RG_01M_2024_4326")
            LOGGER.info("Fetched %s rows from GISCO (GPKG)", len(gdf))
        except Exception as exc_gpkg:
            LOGGER.warning("Falling back to sample NUTS geometries due to: %s", exc_gpkg)
            from shapely.geometry import box
            gdf = gpd.GeoDataFrame(
                {
                    "NUTS_ID": ["ES62", "FR10"],
                    "NAME_LATN": ["Murcia", "Île-de-France"],
                    "CNTR_CODE": ["ES", "FR"],
                    "LEVL_CODE": [2, 2],
                    "geometry": [
                        box(-1.5, 37.5, -0.5, 38.5),
                        box(1.5, 48.0, 3.5, 49.0),
                    ],
                },
                crs="EPSG:4326",
            )

    if "LEVL_CODE" in gdf.columns:
        gdf = gdf[gdf["LEVL_CODE"].eq(2)].copy()

    keep_cols = [c for c in ["NUTS_ID", "NAME_LATN", "CNTR_CODE", "geometry"] if c in gdf.columns]
    nuts2 = gdf[keep_cols].copy()
    nuts2 = nuts2.rename(
        columns={
            "NUTS_ID": "region_id",
            "NAME_LATN": "region_name",
            "CNTR_CODE": "country_code",
        }
    )
    # Optional: compute area_km2 (use equal-area projection)
    try:
        nuts2_area = nuts2.to_crs(3035)
        nuts2["area_km2"] = nuts2_area.geometry.area / 1e6
    except Exception:
        nuts2["area_km2"] = np.nan

    write_geo_dataframe(nuts2, target_path)
    return nuts2


# ---------------------------------------------------------------------------
# Eurostat tourism + GDP retrieval via pandaSDMX
# ---------------------------------------------------------------------------

def _sdmx_to_df(series_like) -> pd.DataFrame:
    """Convert SDMX Series/MultiIndex to a tidy DataFrame with robust column naming."""
    df = series_like.reset_index()
    # Value column sometimes ends up as 0 if the object was a Series
    if "value" not in df.columns:
        # assume the last column is value if unnamed
        df = df.rename(columns={df.columns[-1]: "value"})
    # Normalize dimension names
    rename_map = {}
    for cand, target in [
        (("geo", "GEO"), "region_id"),
        (("time_period", "TIME_PERIOD", "time", "TIME"), "period"),
        (("freq", "FREQ"), "freq"),
        (("unit", "UNIT"), "unit"),
    ]:
        for c in cand:
            if c in df.columns:
                rename_map[c] = target
                break
    df = df.rename(columns=rename_map)
    return df


def fetch_eurostat_tourism(force: bool = False) -> pd.DataFrame:
    """Retrieve Eurostat tourism datasets with pandaSDMX.

    Uses datasets tour_occ_arn2/tour_occ_nin2 for arrivals/nights.
    """
    target_path = INTERIM_DIR / "eurostat_tourism.parquet"
    if target_path.exists() and not force:
        LOGGER.info("Loading cached Eurostat tourism table from %s", target_path)
        return pd.read_parquet(target_path)

    try:
        # Provider name can be "ESTAT" or "EUROSTAT" depending on pandasdmx version
        from pandasdmx import Request  # type: ignore

        try:
            client = Request("ESTAT")
        except Exception:
            client = Request("EUROSTAT")

        dataset_id = "tour_occ_nin2"  # nights
        LOGGER.info("Querying Eurostat dataset %s via pandaSDMX", dataset_id)
        # Request all; you may restrict with params for speed
        data_response = client.data(resource_id=dataset_id, key=".")
        df = _sdmx_to_df(data_response.to_pandas())

        # Parse period to year/month when possible
        period_col = _pick(df, ("period",))
        df["period"] = df[period_col].astype(str)
        # Common formats: "2020", "2020-01", "2020M01"
        def _split_period(s: str) -> Tuple[int, Optional[int]]:
            s = s.strip()
            if "-" in s:
                y, m = s.split("-", 1)
                return int(y), int(m)
            if "M" in s.upper():
                y, m = s.upper().split("M", 1)
                return int(y), int(m)
            return int(s), None

        ym = df["period"].map(_split_period)
        df["year"] = [t[0] for t in ym]
        df["month"] = [t[1] for t in ym]

        region_col = _pick(df, ("region_id",))
        df["region_id"] = df[region_col]

        # Compute seasonality index when monthly data exist: top 3 months share (%)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        if df["month"].notna().any():
            monthly = df.dropna(subset=["month"]).copy()
            seasonality = (
                monthly.groupby(["region_id", "year"], dropna=True)
                .apply(lambda x: x.nlargest(3, "value")["value"].sum() / x["value"].sum() * 100.0)
                .rename("seasonality_idx")
                .reset_index()
            )
        else:
            # If only annual totals, we cannot compute a true seasonality index
            seasonality = (
                df.groupby(["region_id", "year"], dropna=True)["value"].sum()
                .rename("seasonality_idx")
                .reset_index()
            )
            seasonality["seasonality_idx"] = np.nan

    except Exception as exc:
        LOGGER.warning("Falling back to synthetic tourism data due to: %s", exc)
        seasonality = pd.DataFrame(
            {
                "region_id": ["ES62", "ES62", "FR10", "FR10"],
                "year": [2030, 2050, 2030, 2050],
                "seasonality_idx": [np.nan, np.nan, np.nan, np.nan],
            }
        )

    seasonality.to_parquet(target_path, index=False)
    LOGGER.info("Wrote cleaned tourism data to %s", target_path)
    return seasonality


def fetch_eurostat_gdp(force: bool = False) -> pd.DataFrame:
    target_path = INTERIM_DIR / "eurostat_gdp.parquet"
    if target_path.exists() and not force:
        LOGGER.info("Loading cached Eurostat GDP table from %s", target_path)
        return pd.read_parquet(target_path)

    try:
        from pandasdmx import Request  # type: ignore

        try:
            client = Request("ESTAT")
        except Exception:
            client = Request("EUROSTAT")

        dataset_id = "nama_10r_2gdp"
        LOGGER.info("Querying Eurostat dataset %s via pandaSDMX", dataset_id)
        data_response = client.data(resource_id=dataset_id, key=".")
        df = _sdmx_to_df(data_response.to_pandas())

        region_col = _pick(df, ("region_id",))
        period_col = _pick(df, ("period",))
        df = df.rename(columns={region_col: "region_id", period_col: "year"})
        df["year"] = df["year"].astype(str).str.slice(0, 4).astype(int)
        df = df.rename(columns={"value": "gdp"})
        df["gdp"] = pd.to_numeric(df["gdp"], errors="coerce")

        # Keep a recent slice to shrink file size (optional)
        if df["year"].notna().any():
            recent_min = max(df["year"].min(), 2010)
            df = df[df["year"] >= recent_min].copy()

    except Exception as exc:
        LOGGER.warning("Falling back to synthetic GDP data due to: %s", exc)
        df = pd.DataFrame(
            {"region_id": ["ES62", "FR10"], "year": [2030, 2030], "gdp": [45_000.0, 350_000.0]}
        )

    df.to_parquet(target_path, index=False)
    LOGGER.info("Wrote cleaned GDP data to %s", target_path)
    return df


# ---------------------------------------------------------------------------
# CDS climate retrieval and zonal statistics
# ---------------------------------------------------------------------------

def load_cds_config() -> dict:
    cfg_path = CONFIG_DIR / "cds_requests.yml"
    if cfg_path.exists():
        return read_yaml(cfg_path)
    LOGGER.warning("CDS config missing at %s. Using default template.", cfg_path)
    return {
        "dataset": "sis-heat-and-cold-spells",
        "variables": ["heat_wave_frequency"],
        "scenarios": ["Baseline", "Conservative", "Moderate", "High", "Extreme"],
        "years": [2030, 2050],
        "baseline": {"years": [1981, 2010], "scenario": "Baseline"},
        "scenario_request_map": {
            "Baseline": "historical",
            "Conservative": "rcp26",
            "Moderate": "rcp45",
            "High": "rcp60",
            "Extreme": "rcp85",
        },
    }


def canonical_climate_column(variable_name: str) -> Optional[str]:
    """Map dataset-specific variable names to canonical CRISI columns."""
    name = variable_name.lower()
    if "tmax" in name or "temperature" in name:
        return "heat_tmax_delta"
    if "heat_wave" in name or "heatwave" in name:
        return "heatwave_days"
    if "drought" in name or "spei" in name:
        return "drought_spei"
    return None


def make_final_schema(allowed_scenarios: Iterable[str]) -> DataFrameSchema:
    scenario_values = sorted({str(s) for s in allowed_scenarios if s is not None})

    columns: Dict[str, Column] = {
        "region_id": Column(str, nullable=False),
        "region_name": Column(str, nullable=True),
        "year": Column(int, checks=[Check.ge(1950), Check.le(2100)], nullable=False),
        "heat_tmax_delta": Column(float, nullable=True, checks=[Check.ge(-5), Check.le(10)]),
        "heatwave_days": Column(float, nullable=True, checks=[Check.ge(0)]),
        "drought_spei": Column(float, nullable=True),
        "tourism_gdp_share": Column(float, nullable=True, checks=[Check.ge(0), Check.le(100)]),
        "seasonality_idx": Column(float, nullable=True, checks=[Check.ge(0), Check.le(100)]),
        "readiness_proxy": Column(float, nullable=True),
        "projected_drop": Column(float, nullable=True),
    }
    if scenario_values:
        columns["scenario"] = Column(str, nullable=False, checks=[Check.isin(scenario_values)])
    return DataFrameSchema(columns)


@dataclass
class ClimateRow:
    region_id: str
    scenario: str
    year: int
    heat_tmax_delta: Optional[float]
    heatwave_days: Optional[float]
    drought_spei: Optional[float]


def compute_zonal_stats_raster(raster, geometries: gpd.GeoDataFrame) -> pd.Series:
    """Compute mean zonal statistics for a raster over provided geometries."""
    # Ensure rioxarray accessor is registered
    import rioxarray  # noqa: F401
    stats: List[float] = []
    for _, row in geometries.iterrows():
        clipped = raster.rio.clip([row.geometry], geometries.crs, drop=False)
        stats.append(float(clipped.mean().item()))
    return pd.Series(stats, index=geometries.index)


def aggregate_cds_to_nuts(nuts_gdf: gpd.GeoDataFrame, force: bool = False) -> pd.DataFrame:
    """Download CDS NetCDFs and aggregate to NUTS-2."""
    target_path = INTERIM_DIR / "climate.parquet"
    if target_path.exists() and not force:
        LOGGER.info("Loading cached climate aggregates from %s", target_path)
        return pd.read_parquet(target_path)

    config = load_cds_config()
    dataset = config.get("dataset")
    variables = list(config.get("variables", []))
    scenarios = list(config.get("scenarios", []))
    years = list(config.get("years", []))
    baseline_cfg = config.get("baseline", {})
    baseline_years = baseline_cfg.get("years", [])
    scenario_request_map: Dict[str, Optional[str]] = config.get("scenario_request_map", {})

    rows: List[ClimateRow] = []
    raw_dir = RAW_DIR / "cds"
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from cdsapi import Client  # type: ignore
        import xarray as xr
        import rioxarray  # noqa: F401

        client = Client()  # uses ~/.cdsapirc

        baseline_means = {}
        if baseline_years:
            baseline_file = raw_dir / f"{dataset}_baseline.nc"
            if not baseline_file.exists() or force:
                baseline_scenario = baseline_cfg.get("scenario") or (scenarios[0] if scenarios else None)
                request_experiment = scenario_request_map.get(baseline_scenario, baseline_scenario)
                request = {
                    "format": "netcdf",
                    "experiment": request_experiment,
                    "variable": list(variables),
                    "year": [str(y) for y in baseline_years],
                }
                LOGGER.info("Retrieving CDS baseline dataset: %s", json.dumps(request))
                client.retrieve(dataset, request, str(baseline_file))
            ds_base = xr.open_dataset(baseline_file)
            for var in variables:
                if var in ds_base:
                    baseline_means[var] = float(ds_base[var].mean().item())

        # Reproject nuts to EPSG:4326 for consistent clipping if needed
        nuts_wgs = nuts_gdf.to_crs("EPSG:4326")

        for scenario in scenarios:
            for year in years:
                filename = raw_dir / f"{dataset}_{scenario}_{year}.nc"
                if not filename.exists() or force:
                    request_experiment = scenario_request_map.get(scenario, scenario)
                    request = {
                        "format": "netcdf",
                        "experiment": request_experiment,
                        "variable": list(variables),
                        "year": str(year),
                    }
                    LOGGER.info("Retrieving CDS file: %s", json.dumps(request))
                    client.retrieve(dataset, request, str(filename))

                import xarray as xr  # local
                ds = xr.open_dataset(filename)

                scenario_df = nuts_wgs[["region_id"]].copy()
                scenario_df["scenario"] = scenario
                scenario_df["year"] = int(year)
                scenario_df[["heat_tmax_delta", "heatwave_days", "drought_spei"]] = np.nan

                for var in variables:
                    if var not in ds:
                        LOGGER.debug("Variable %s not in dataset; skipping", var)
                        continue
                    data = ds[var]
                    # Ensure CRS on data for rioxarray
                    if not getattr(data, "rio", None) or not data.rio.crs:
                        data = data.rio.write_crs("EPSG:4326")
                    stats = compute_zonal_stats_raster(data, nuts_wgs)
                    baseline = baseline_means.get(var)
                    values = stats - baseline if baseline is not None else stats

                    column = canonical_climate_column(var)
                    if column is None:
                        LOGGER.debug("Ignoring climate variable %s (no canonical mapping)", var)
                        continue
                    scenario_df[column] = values.values

                rows.extend(
                    ClimateRow(
                        region_id=row.region_id,
                        scenario=row.scenario,
                        year=row.year,
                        heat_tmax_delta=(float(row.heat_tmax_delta) if pd.notna(row.heat_tmax_delta) else None),
                        heatwave_days=(float(row.heatwave_days) if pd.notna(row.heatwave_days) else None),
                        drought_spei=(float(row.drought_spei) if pd.notna(row.drought_spei) else None),
                    )
                    for row in scenario_df.itertuples(index=False)
                )
    except Exception as exc:
        LOGGER.warning("Falling back to synthetic climate data due to: %s", exc)
        rng = np.random.default_rng(42)
        fallback_scenarios = scenarios or ["Baseline", "Conservative", "Moderate", "High", "Extreme"]
        fallback_years = years or [2030, 2050]
        severity_lookup = {"Baseline": 0.5, "Conservative": 1.0, "Moderate": 1.5, "High": 2.0, "Extreme": 2.5}
        for scenario in fallback_scenarios:
            for year in fallback_years:
                for region_id in nuts_gdf["region_id"].tolist():
                    loc = severity_lookup.get(scenario, 1.5)
                    rows.append(
                        ClimateRow(
                            region_id=region_id,
                            scenario=scenario,
                            year=int(year),
                            heat_tmax_delta=float(rng.normal(loc=loc, scale=0.3)),
                            heatwave_days=float(rng.integers(5, 30)),
                            drought_spei=np.nan,
                        )
                    )

    climate_df = pd.DataFrame([row.__dict__ for row in rows])
    if not climate_df.empty:
        climate_df = (
            climate_df.groupby(["region_id", "scenario", "year"], as_index=False)
            .agg({"heat_tmax_delta": "mean", "heatwave_days": "mean", "drought_spei": "mean"})
        )
    climate_df.to_parquet(target_path, index=False)
    LOGGER.info("Wrote climate aggregates to %s", target_path)
    return climate_df


# ---------------------------------------------------------------------------
# Final assembly & validation
# ---------------------------------------------------------------------------

def load_settings() -> dict:
    cfg_path = CONFIG_DIR / "settings.yml"
    if cfg_path.exists():
        return read_yaml(cfg_path)
    LOGGER.warning("Settings config missing at %s; using defaults", cfg_path)
    return {
        "weights": {
            "exposure": {"heat_tmax_delta": 0.4, "heatwave_days": 0.4, "drought_spei": 0.2},
            "sensitivity": {"tourism_gdp_share": 0.6, "seasonality_idx": 0.4},
            "capacity": {"readiness_proxy": -0.2},
        },
        "thresholds": {"tourism_gdp_share_high": 15, "projected_drop_high": 10},
    }


FINAL_COLUMNS = [
    "region_id",
    "region_name",
    "geometry",
    "scenario",
    "year",
    "heat_tmax_delta",
    "heatwave_days",
    "drought_spei",
    "tourism_gdp_share",
    "seasonality_idx",
    "readiness_proxy",
    "projected_drop",
]


def build_final_features(
    nuts: gpd.GeoDataFrame,
    climate: pd.DataFrame,
    tourism: pd.DataFrame,
    gdp: pd.DataFrame,
) -> gpd.GeoDataFrame:
    LOGGER.info("Joining climate, tourism, and NUTS data")
    # Merge tourism metrics onto climate table (left join on region/year)
    features = climate.merge(tourism, on=["region_id", "year"], how="left")

    # Placeholder tourism_gdp_share (TODO when Eurostat tourism satellite data is integrated)
    features["tourism_gdp_share"] = np.nan

    # Optionally use GDP for future ratios. Currently unused but kept for completeness.
    features = features.merge(gdp, on=["region_id", "year"], how="left", suffixes=("", "_gdp"))

    # Join names/geometry
    keep_geo = [c for c in ["region_id", "region_name", "geometry"] if c in nuts.columns]
    features = features.merge(nuts[keep_geo], on="region_id", how="left")

    # Add empty columns required by schema
    for col in ["readiness_proxy", "projected_drop"]:
        if col not in features:
            features[col] = np.nan

    # Reorder and ensure geometry column is GeoSeries
    features = gpd.GeoDataFrame(features, geometry="geometry", crs=nuts.crs if "geometry" in nuts else "EPSG:4326")
    for col in FINAL_COLUMNS:
        if col not in features:
            features[col] = np.nan
    features = features[FINAL_COLUMNS]

    allowed_scenarios = climate["scenario"].dropna().unique().tolist()
    if not allowed_scenarios:
        allowed_scenarios = load_cds_config().get("scenarios", [])
    schema = make_final_schema(allowed_scenarios)
    schema.validate(features.drop(columns="geometry"))

    output_path = PROCESSED_DIR / "features_geo.parquet"
    write_geo_dataframe(features, output_path)
    return features


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Iterable[str]] = None) -> CliArgs:
    parser = argparse.ArgumentParser(description="CRISI ETL pipeline")
    parser.add_argument("--pull-cds", action="store_true", help="Download climate data from CDS")
    parser.add_argument("--pull-eurostat", action="store_true", help="Download Eurostat datasets")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild intermediate files")
    parsed = parser.parse_args(argv)
    return CliArgs(pull_cds=parsed.pull_cds, pull_eurostat=parsed.pull_eurostat, rebuild=parsed.rebuild)


def main(argv: Optional[Iterable[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    args = parse_args(argv)
    ensure_directories()

    nuts = download_nuts2(force=args.rebuild)

    tourism = fetch_eurostat_tourism(force=args.pull_eurostat or args.rebuild)
    gdp = fetch_eurostat_gdp(force=args.pull_eurostat or args.rebuild)

    climate = aggregate_cds_to_nuts(nuts, force=args.pull_cds or args.rebuild)

    build_final_features(nuts, climate, tourism, gdp)


if __name__ == "__main__":
    main(sys.argv[1:])
