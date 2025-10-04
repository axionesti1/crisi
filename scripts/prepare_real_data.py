"""Prepare Greece-specific datasets for the CRISI Streamlit app.

This script reads raw CSVs and Corine Land Cover (CLC) shapefiles from the
``dataset`` folder of the repository, harmonises their schema (regions,
year columns, numeric fields), aggregates the CLC by region and year, and
merges everything into a single tidy Parquet file. The output is written
to ``data/processed`` with a companion JSON file listing which numeric
columns are available for analysis. It can be re-run anytime the raw
data changes.

Key features:

* **Region harmonisation** – Many datasets use different names or NUTS codes
  for Greek regions. The script normalises them to a consistent key
  ``_region_norm`` using simple transliteration and upper-casing and
  includes a mapping for common variants (e.g. ``ATTIKI`` → ``ATTICA``).
* **Year extraction** – A ``year`` column is extracted from date fields or
  existing year columns and converted to integer type for reliable joins.
* **CLC summarisation** – Each CLC ZIP (1990–2018) is extracted on the fly
  (requires GeoPandas) and clipped to Greece using country code fields or
  a bounding box. Areas (in km²) are summed by region, year and land cover
  code, then pivoted to wide format.
* **Merging datasets** – Temperature, sea-level-rise, flood fatalities and
  main datasets are merged on ``[_region_norm, year]``. Columns are
  prefixed with their dataset tag to avoid collisions.

Run this script from the repository root:

    python scripts/prepare_real_data.py

The final Parquet file can then be loaded in the Streamlit app for real
data analysis.
"""

import re
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

try:
    import geopandas as gpd  # type: ignore
    from shapely.geometry import box  # type: ignore
    HAS_GPD = True
except Exception:
    HAS_GPD = False

# Directories
REPO = Path(__file__).resolve().parents[1]
DATA_RAW = REPO / "dataset"
DATA_OUT = REPO / "data" / "processed"
DATA_OUT.mkdir(parents=True, exist_ok=True)

# Patterns for finding raw files. Adjust as needed.
PATTERNS = {
    "main": ["main_dataset*.csv", "main*dataset*csv.csv"],
    "temp": ["temperature_dataset*.csv", "temperature*dataset*csv.csv"],
    "slr": ["sea*level*rise*.csv", "sea_level_rise*dataset*.csv"],
    "flood": ["flood*fatalit*.csv", "flood*fatal*.csv"],
    "clc": ["clc*.zip", "*corine*.zip", "*CLC*.zip"],
    "vars": ["*.txt", "*readme*.md", "*variables*.txt"],
}


def _find_one(patterns: List[str]) -> Optional[Path]:
    """Return the first matching file from DATA_RAW for the given patterns."""
    for pat in patterns:
        for p in sorted(DATA_RAW.glob(pat)):
            if p.is_file():
                return p
    return None


def _find_many(patterns: List[str]) -> List[Path]:
    """Return all matching files from DATA_RAW for the given patterns (unique)."""
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(sorted(DATA_RAW.glob(pat)))
    # unique
    out: List[Path] = []
    seen: set[Path] = set()
    for p in hits:
        if p not in seen and p.is_file():
            out.append(p)
            seen.add(p)
    return out


def _guess_year_from_name(name: str) -> Optional[int]:
    """Extract a 4-digit year from a filename, if present."""
    m = re.search(r"(19|20)\d{2}", name)
    return int(m.group(0)) if m else None


def _normalize_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``_region_key`` and ``_region_norm`` columns based on the first region-like column."""
    if df.empty:
        return df
    cols = {c.lower(): c for c in df.columns}
    cand = [x for x in [
        "region",
        "periphery",
        "perifereia",
        "nuts_name",
        "περιοχή",
        "νομαρχία",
        "district",
        "region_name",
        "nuts 2",
    ] if x in cols]
    if not cand:
        return df
    rc = cols[cand[0]]
    # create key by stripping accents and uppercasing
    def translit(val: str) -> str:
        return (
            val
            .strip()
            .encode("ascii", "ignore")
            .decode("ascii")
            .upper()
        )
    df["_region_key"] = df[rc].astype(str).apply(translit)
    rep = {
        "ATTIKI": "ATTICA",
        "ATTIKH": "ATTICA",
        "CENTRAL MACEDONIA": "CENTRAL MACEDONIA",
        "KENTRIKH MAKEDONIA": "CENTRAL MACEDONIA",
        "ANATOLIKI MAKEDONIA KAI THRAKI": "EASTERN MACEDONIA & THRACE",
        "THRAKI": "EASTERN MACEDONIA & THRACE",
        "NOTIO AIGAIO": "SOUTH AEGEAN",
        "NORTH AEGEAN": "NORTH AEGEAN",
        "KRHTH": "CRETE",
        "PELOPONNISOS": "PELOPONNESE",
    }
    df["_region_norm"] = df["_region_key"].map(rep).fillna(df["_region_key"])
    return df


def load_csv(path: Optional[Path]) -> pd.DataFrame:
    """Load a CSV file with date parsing when possible."""
    if not path:
        return pd.DataFrame()
    try:
        # read header to guess date columns
        header = pd.read_csv(path, nrows=0)
        date_cols = [c for c in header.columns if c.lower() in [
            "date",
            "ημερομηνία",
            "time",
            "datetime",
        ]]
        df = pd.read_csv(path, low_memory=False, parse_dates=date_cols)
        return df
    except Exception:
        return pd.read_csv(path, low_memory=False)


def ensure_year(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there is an integer ``year`` column in the DataFrame."""
    if df.empty:
        return df
    cols = {c.lower(): c for c in df.columns}
    if "year" in cols:
        col = cols["year"]
        df["year"] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    elif "έτος" in cols:
        col = cols["έτος"]
        df["year"] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    elif "date" in cols:
        col = cols["date"]
        df["year"] = pd.to_datetime(df[col], errors="coerce").dt.year.astype("Int64")
    return df


def summarise_clc(zips: List[Path]) -> pd.DataFrame:
    """Summarise CLC polygons for Greece by region, year and class.

    Returns a DataFrame with columns ``['_region_norm', 'year', 'CLC_<code>']`` and area in km².
    """
    if not HAS_GPD:
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    # bounding box for Greece
    greece_bbox = box(19.0, 34.6, 29.7, 41.8)
    for zp in zips:
        try:
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(td)
                shp_files = list(Path(td).rglob("*.shp"))
                if not shp_files:
                    continue
                shp = shp_files[0]
                gdf = gpd.read_file(shp)
                # standardise CRS
                try:
                    gdf = gdf.to_crs(4326)
                except Exception:
                    pass
                # keep Greece by code or clip
                lower = {c.lower(): c for c in gdf.columns}
                mask = None
                for key in ["cntr_code", "country", "nuts_0", "iso3", "cntrname", "cntr-code"]:
                    if key in lower:
                        col = lower[key]
                        mask = gdf[col].astype(str).str.upper().isin(["EL", "GRC", "GREECE", "ΕΛΛΑΔΑ"])
                        if mask.any():
                            break
                if mask is not None and mask.any():
                    gdf = gdf[mask]
                else:
                    # clip to bounding box
                    gdf = gpd.clip(gdf, greece_bbox)
                # pick class column
                class_col = None
                for c in [
                    "code_18",
                    "code_12",
                    "code_06",
                    "code_00",
                    "code_90",
                    "clc_code",
                    "CODE",
                    "CLC_CODE",
                    "CODE_18",
                ]:
                    if c in gdf.columns:
                        class_col = c
                        break
                # year from filename or column
                y = _guess_year_from_name(zp.name)
                if y is None and "year" in gdf.columns:
                    y = int(gdf.iloc[0]["year"])
                gdf["year"] = y
                # compute area (km²)
                try:
                    gdf_proj = gdf.to_crs(3857)
                    gdf["area_km2"] = gdf_proj.geometry.area / 1e6
                except Exception:
                    gdf["area_km2"] = 1.0
                # region name
                name_col = None
                for key in ["nuts_name", "region", "perif", "adm_name", "name", "region_name", "nuts_name"]:
                    if key.lower() in lower:
                        name_col = lower[key.lower()]
                        break
                tmp = pd.DataFrame({"_region_norm": "GREECE"}, index=gdf.index)
                if name_col is not None:
                    def translit(val: str) -> str:
                        return (
                            str(val)
                            .strip()
                            .encode("ascii", "ignore")
                            .decode("ascii")
                            .upper()
                        )
                    tmp["_region_norm"] = gdf[name_col].apply(translit)
                agg_cols = ["_region_norm", "year"]
                if class_col:
                    agg_cols.append(class_col)
                gdf = gdf.join(tmp["_region_norm"])
                gb = gdf.groupby(agg_cols, dropna=False)["area_km2"].sum().reset_index()
                rows.append(gb)
        except Exception as e:
            print(f"CLC processing failed for {zp}: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    # pivot wide if class present
    class_candidates = [c for c in df.columns if c not in ["_region_norm", "year", "area_km2"]]
    if class_candidates:
        cls_col = class_candidates[-1]
        df_w = df.pivot_table(
            index=["_region_norm", "year"],
            columns=cls_col,
            values="area_km2",
            aggfunc="sum",
        ).fillna(0.0)
        df_w.columns = [f"CLC_{c}" for c in df_w.columns]
        df_w = df_w.reset_index()
        return df_w
    else:
        return df.groupby(["_region_norm", "year"])['area_km2'].sum().reset_index().rename(columns={"area_km2": "CLC_area_km2"})


def safe_merge(base: pd.DataFrame, other: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Merge two tables on ``[_region_norm, year]`` and prefix numeric columns."""
    if other.empty:
        return base
    if "_region_norm" not in other.columns or "year" not in other.columns:
        return base
    # select numeric columns
    num_cols = [c for c in other.select_dtypes("number").columns if c not in ["year"]]
    # also include non-numeric? might be counts; we only want numeric features
    df = other[["_region_norm", "year"] + num_cols]
    df = df.rename(columns={c: f"{tag}__{c}" for c in num_cols})
    return base.merge(df, on=["_region_norm", "year"], how="outer")


def main() -> None:
    # Find raw files
    main_csv = _find_one(PATTERNS["main"])
    temp_csv = _find_one(PATTERNS["temp"])
    slr_csv = _find_one(PATTERNS["slr"])
    flood_csv = _find_one(PATTERNS["flood"])
    clc_zips = _find_many(PATTERNS["clc"])

    df_main = load_csv(main_csv)
    df_temp = load_csv(temp_csv)
    df_slr = load_csv(slr_csv)
    df_flood = load_csv(flood_csv)

    # Normalise regions & year columns
    for df in [df_main, df_temp, df_slr, df_flood]:
        if not df.empty:
            _normalize_regions(df)
            ensure_year(df)

    # Summarise CLC
    df_clc = summarise_clc(clc_zips) if clc_zips else pd.DataFrame()

    # Build base DataFrame – choose whichever dataset has region+year first
    base = pd.DataFrame()
    for candidate in [df_temp, df_main, df_slr, df_flood]:
        if not candidate.empty and "_region_norm" in candidate.columns and "year" in candidate.columns:
            base = candidate[["_region_norm", "year"]].drop_duplicates().copy()
            break
    if base.empty:
        # No region-year table; create a minimal stub
        print("No base dataset found; cannot merge real data.")
        return
    # Merge with each dataset
    unified = base.copy()
    unified = safe_merge(unified, df_main, "main")
    unified = safe_merge(unified, df_temp, "temp")
    unified = safe_merge(unified, df_slr, "slr")
    unified = safe_merge(unified, df_flood, "flood")
    # merge CLC summary
    if not df_clc.empty and set(["_region_norm", "year"]).issubset(df_clc.columns):
        unified = unified.merge(df_clc, on=["_region_norm", "year"], how="left")
    elif not df_clc.empty and "_region_norm" in df_clc.columns:
        unified = unified.merge(df_clc, on=["_region_norm"], how="left")
    # Fill region if missing
    unified["_region_norm"] = unified["_region_norm"].fillna("GREECE")
    # Sort
    unified = unified.sort_values(["_region_norm", "year"]).reset_index(drop=True)
    # Write outputs: Parquet preferred (pyarrow) but fallback to CSV if unavailable
    out_main = DATA_OUT / "crisi_greece_processed.parquet"
    try:
        unified.to_parquet(out_main, index=False)
    except Exception:
        # Parquet engine missing; write CSV instead
        out_main = out_main.with_suffix(".csv")
        unified.to_csv(out_main, index=False)
    # Build metrics list
    numeric_cols = [c for c in unified.select_dtypes("number").columns if c.lower() != "year"]
    (DATA_OUT / "crisi_metrics.json").write_text(
        json.dumps({"available_metrics": numeric_cols}, ensure_ascii=False, indent=2)
    )
    # Write CLC summary separate if available
    if not df_clc.empty:
        out_clc = DATA_OUT / "clc_greece_summary.parquet"
        try:
            df_clc.to_parquet(out_clc, index=False)
        except Exception:
            out_clc = out_clc.with_suffix(".csv")
            df_clc.to_csv(out_clc, index=False)
    print(f"Wrote {out_main}")
    if not df_clc.empty:
        print(f"Wrote {out_clc}")


if __name__ == "__main__":
    main()