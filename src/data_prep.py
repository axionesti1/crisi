"""Data preparation script for CRISI pilot.

This script reads raw climate and tourism indicators and spatial
boundaries, merges them into a single GeoDataFrame and writes
GeoParquet output. It is intended as an example pipeline for the
pilot application. In practice, you would call this after
downloading or generating the raw datasets.

Example usage from the project root::

    python src/data_prep.py \
        --shapefile data_raw/regions.shp \
        --climate data_raw/climate_indicators.csv \
        --tourism data_raw/tourism_econ.csv \
        --output data_proc/features_geo.parquet
"""

import argparse
import sys
import pandas as pd
import geopandas as gpd
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Join raw indicators with spatial boundaries.")
    parser.add_argument("--shapefile", required=True, help="Path to a shapefile containing region geometries.")
    parser.add_argument("--climate", required=True, help="CSV with climate indicators (per region per scenario).")
    parser.add_argument("--tourism", required=True, help="CSV with tourism/economic indicators (per region).")
    parser.add_argument("--output", required=True, help="Output GeoParquet file.")
    args = parser.parse_args(argv)

    # Load shapefile; rename geometry and ensure CRS is WGS84 for consistency
    shape_path = Path(args.shapefile)
    gdf = gpd.read_file(shape_path)
    # Expect region_id and region_name columns
    if "region_id" not in gdf.columns:
        # Try to infer region_id from common NUTS naming conventions
        # E.g. NUTS_ID -> region_id, NAME_LATN -> region_name
        if "NUTS_ID" in gdf.columns:
            gdf = gdf.rename(columns={"NUTS_ID": "region_id"})
        else:
            raise KeyError("No region_id column found in shapefile")
    if "region_name" not in gdf.columns:
        # Use NAME_LATN or NUTS_NAME as fallback
        for cand in ["NAME_LATN", "NUTS_NAME", "name"]:
            if cand in gdf.columns:
                gdf = gdf.rename(columns={cand: "region_name"})
                break
        if "region_name" not in gdf.columns:
            # create dummy region_name equal to region_id
            gdf["region_name"] = gdf["region_id"]

    # Read indicator CSVs
    clim = pd.read_csv(args.climate)
    tour = pd.read_csv(args.tourism)

    # Merge climate and tourism on region_id
    # For climate we might have multiple scenarios; we will perform a left
    # merge to propagate tourism variables across scenarios.
    # This assumes tourism data does not vary by scenario in the pilot.
    feat = pd.merge(clim, tour, on="region_id", how="left")

    # Merge with geometries. We'll keep geometry for each region and
    # broadcast across scenarios by re-merging for each scenario.
    # To do this, repeat geometries for each scenario row per region.
    # We'll perform a merge on region_id.
    merged = gdf.merge(feat, on="region_id", how="inner")

    # Ensure output has a consistent CRS (WGS84)
    if merged.crs is None:
        # default to EPSG:4326 if not defined
        merged.set_crs(4326, inplace=True)
    else:
        merged = merged.to_crs(4326)

    # Save as GeoParquet
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"Written GeoParquet to {out_path}")


if __name__ == "__main__":
    main()