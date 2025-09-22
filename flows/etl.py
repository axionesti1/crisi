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
from typing import Dict, Iterable, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from pandera import Column, DataFrameSchema, Check
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

# now this works on recent versions
Column(float, checks=Check.between(0, 100, inclusive="both"))

# External service clients are imported lazily in the functions that use them so
# that the pipeline can operate in environments where optional dependencies are
# unavailable or where network access is restricted.

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
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
    for folder in [RAW_DIR / "cds", INTERIM_DIR, PROCESSED_DIR]:
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


# ---------------------------------------------------------------------------
# GISCO NUTS-2 downloader
# ---------------------------------------------------------------------------


def download_nuts2(force: bool = False) -> gpd.GeoDataFrame:
    """Download GISCO NUTS-2 polygons and store them locally.

    The implementation follows the official GISCO distribution endpoint
    documented at https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/  # noqa: E501  [GISCO]
    """

    target_path = INTERIM_DIR / "nuts2.parquet"
    if target_path.exists() and not force:
        LOGGER.info("Loading cached NUTS-2 dataset from %s", target_path)
        return gpd.read_parquet(target_path)

    url = (
        "https://gisco-services.ec.europa.eu/distribution/v2/nuts/gpkg/"
        "NUTS_RG_01M_2024_4326.gpkg"
    )
    LOGGER.info("Requesting NUTS-2 polygons from %s", url)

    try:
        import requests

        session = requests.Session()
        retries = requests.adapters.Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        response = session.get(url, timeout=60)
        response.raise_for_status()

        temp_path = RAW_DIR / "nuts.gpkg"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(response.content)
        gdf = gpd.read_file(temp_path, layer="NUTS_RG_01M_2024_4326")
        LOGGER.info("Fetched %s rows from GISCO", len(gdf))
    except Exception as exc:  # broad catch to support offline execution
        LOGGER.warning("Falling back to sample NUTS geometries due to: %s", exc)
        from shapely.geometry import box

        gdf = gpd.GeoDataFrame(
            {
                "NUTS_ID": ["ES62", "FR10"],
                "NAME_LATN": ["Murcia", "Île-de-France"],
                "geometry": [
                    box(-1.5, 37.5, -0.5, 38.5),
                    box(1.5, 48.0, 3.5, 49.0),
                ],
            },
            crs="EPSG:4326",
        )

    nuts2 = gdf.loc[gdf["LEVL_CODE"].eq(2) if "LEVL_CODE" in gdf else slice(None), ["NUTS_ID", "NAME_LATN", "geometry"]].copy()
    nuts2 = nuts2.rename(columns={"NUTS_ID": "region_id", "NAME_LATN": "region_name"})
    write_geo_dataframe(nuts2, target_path)
    return nuts2


# ---------------------------------------------------------------------------
# Eurostat tourism + GDP retrieval via pandaSDMX
# ---------------------------------------------------------------------------


def fetch_eurostat_tourism(force: bool = False) -> pd.DataFrame:
    """Retrieve Eurostat tourism datasets with pandaSDMX.

    Uses the API documented at https://pandasdmx.readthedocs.io/en/v0.9/usage.html  # noqa: E501  [Eurostat/pandaSDMX]
    and datasets tour_occ_arn2/tour_occ_nin2 for arrivals/nights.
    """

    target_path = INTERIM_DIR / "eurostat_tourism.parquet"
    if target_path.exists() and not force:
        LOGGER.info("Loading cached Eurostat tourism table from %s", target_path)
        return pd.read_parquet(target_path)

    try:
        from pandasdmx import Request  # type: ignore

        client = Request("EUROSTAT")
        dataset_id = "tour_occ_nin2"
        LOGGER.info("Querying Eurostat dataset %s via pandaSDMX", dataset_id)
        # Request a recent slice of monthly data with GEO dimension at NUTS-2.
        data_response = client.data(resource_id=dataset_id, key=".")
        data_df = data_response.to_pandas()  # MultiIndex Series
        df = data_df.reset_index()
        df = df.rename(columns={"geo": "region_id", "time_period": "period"})
        # Extract year and month if available
        if df["period"].str.contains("-", na=False).any():
            period = df["period"].str.split("-", expand=True)
            df["year"] = period[0].astype(int)
            df["month"] = period[1].astype(int)
        else:
            df["year"] = df["period"].astype(int)
            df["month"] = np.nan
        df = df.rename(columns={0: "value"})
    except Exception as exc:
        LOGGER.warning("Falling back to synthetic tourism data due to: %s", exc)
        sample = {
            "region_id": ["ES62", "ES62", "FR10", "FR10"],
            "year": [2030, 2050, 2030, 2050],
            "month": [np.nan, np.nan, np.nan, np.nan],
            "value": [1.2e6, 1.4e6, 3.1e6, 3.3e6],
        }
        df = pd.DataFrame(sample)
        df["period"] = df["year"].astype(str)

    # Compute seasonality index: share of top 3 months of nights. If monthly
    # data available, calculate; otherwise fall back to NaN.
    if df["month"].notna().any():
        nights = df.dropna(subset=["month"]).copy()
        nights["value"] = pd.to_numeric(nights["value"], errors="coerce")
        seasonality = (
            nights.groupby(["region_id", "year"], dropna=True)
            .apply(lambda x: x.nlargest(3, "value")["value"].sum() / x["value"].sum() * 100)
            .rename("seasonality_idx")
            .reset_index()
        )
    else:
        seasonality = (
            df.groupby(["region_id", "year"], dropna=True)["value"].sum()
            .rename("seasonality_idx")
            .reset_index()
        )
        seasonality["seasonality_idx"] = np.nan

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

        client = Request("EUROSTAT")
        dataset_id = "nama_10r_2gdp"
        LOGGER.info("Querying Eurostat dataset %s via pandaSDMX", dataset_id)
        data_response = client.data(resource_id=dataset_id, key=".")
        data_df = data_response.to_pandas()
        df = data_df.reset_index().rename(columns={"geo": "region_id", "time_period": "year", 0: "gdp"})
        df["year"] = df["year"].astype(int)
    except Exception as exc:
        LOGGER.warning("Falling back to synthetic GDP data due to: %s", exc)
        df = pd.DataFrame(
            {
                "region_id": ["ES62", "FR10"],
                "year": [2030, 2030],
                "gdp": [45_000.0, 350_000.0],
            }
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

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
from typing import Iterable, Dict

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

    import rioxarray  # noqa: F401  # needed for rio accessor

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

        client = Client()  # Authentication handled via ~/.cdsapirc; see docs.  [CDS]

        baseline_means = {}
        if baseline_years:
            baseline_file = raw_dir / f"{dataset}_baseline.nc"
            if not baseline_file.exists() or force:
                baseline_scenario = baseline_cfg.get("scenario")
                if baseline_scenario is None and scenarios:
                    baseline_scenario = scenarios[0]
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
                baseline_means[var] = ds_base[var].mean().item()

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

                ds = xr.open_dataset(filename)
                scenario_df = nuts_gdf[["region_id"]].copy()
                scenario_df["scenario"] = scenario
                scenario_df["year"] = int(year)
                scenario_df[["heat_tmax_delta", "heatwave_days", "drought_spei"]] = np.nan

                for var in variables:
                    data = ds[var]
                    if not data.rio.crs:
                        data = data.rio.write_crs("EPSG:4326")
                    data = data.rio.reproject(nuts_gdf.crs)
                    stats = compute_zonal_stats_raster(data, nuts_gdf)
                    baseline = baseline_means.get(var)
                    values = stats - baseline if baseline is not None else stats

                    column = canonical_climate_column(var)
                    if column is None:
                        LOGGER.debug("Ignoring climate variable %s (no canonical mapping)", var)
                        continue

                    scenario_df[column] = values

                rows.extend(
                    ClimateRow(
                        region_id=row.region_id,
                        scenario=row.scenario,
                        year=row.year,
                        heat_tmax_delta=(
                            float(row.heat_tmax_delta)
                            if pd.notna(row.heat_tmax_delta)
                            else None
                        ),
                        heatwave_days=(
                            float(row.heatwave_days) if pd.notna(row.heatwave_days) else None
                        ),
                        drought_spei=(
                            float(row.drought_spei) if pd.notna(row.drought_spei) else None
                        ),
                    )
                    for row in scenario_df.itertuples(index=False)
                )
    except Exception as exc:
        LOGGER.warning("Falling back to synthetic climate data due to: %s", exc)
        rng = np.random.default_rng(42)
        fallback_scenarios = scenarios or ["Baseline", "Conservative", "Moderate", "High", "Extreme"]
        fallback_years = years or [2030, 2050]
        severity_lookup = {
            "Baseline": 0.5,
            "Conservative": 1.0,
            "Moderate": 1.5,
            "High": 2.0,
            "Extreme": 2.5,
        }

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
    climate_df = (
        climate_df.groupby(["region_id", "scenario", "year"], as_index=False)
        .agg({
            "heat_tmax_delta": "mean",
            "heatwave_days": "mean",
            "drought_spei": "mean",
        })
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

    features = features.merge(nuts[["region_id", "region_name", "geometry"]], on="region_id", how="left")

    # Add empty columns required by schema
    for col in ["readiness_proxy", "projected_drop"]:
        if col not in features:
            features[col] = np.nan

    # Reorder and ensure geometry column is GeoSeries
    features = gpd.GeoDataFrame(features, geometry="geometry", crs=nuts.crs)
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    args = parse_args(argv)
    ensure_directories()

    nuts = download_nuts2(force=args.rebuild)

    tourism = fetch_eurostat_tourism(force=args.pull_eurostat or args.rebuild)
    gdp = fetch_eurostat_gdp(force=args.pull_eurostat or args.rebuild)

    climate = aggregate_cds_to_nuts(nuts, force=args.pull_cds or args.rebuild)

    build_final_features(nuts, climate, tourism, gdp)


if __name__ == "__main__":
    main(sys.argv[1:])
