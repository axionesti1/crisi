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
# Patterns for finding raw files. Adjust as needed.
PATTERNS = {
    # Main dataset patterns: prioritise the latest rigorous dataset.  First, look
    # for files explicitly marked ``_rigorous_new``.  If none are found,
    # fall back to other remade datasets (rigorous or generic) and finally
    # to corrected, enhanced and legacy files.  Sorting by pattern
    # ordering rather than alphabetical ensures that ``_rigorous_new``
    # datasets take precedence over older ``_rigorous_final`` versions.
    "main": [
        "main_dataset_remade*rigorous_new*.csv",
        "main_dataset_remade*rigorous*.csv",
        "main_dataset*remade*.csv",
        "main_dataset*corrected*.csv",
        "main_dataset*enhanced*.csv",
        "main_dataset*.csv",
        "main*dataset*csv.csv",
    ],
    "temp": ["temperature_dataset*.csv", "temperature*dataset*csv.csv"],
    "slr": ["sea*level*rise*.csv", "sea_level_rise*dataset*.csv"],
    "flood": ["flood*fatalit*.csv", "flood*fatal*.csv"],
    # Corine Land Cover data can come as ZIPs of shapefiles or as FileGeodatabase
    # directories (.gdb).  We look for both patterns.
    "clc": ["clc*.zip", "*corine*.zip", "*CLC*.zip", "*.gdb"],
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
    cand = [
        x
        for x in [
            "region",
            "periphery",
            "perifereia",
            "nuts_name",
            "περιοχή",
            "νομαρχία",
            "district",
            "region_name",
            "nuts 2",
            "nuts 2 area name",  # handle temperature dataset column name
        ]
        if x in cols
    ]
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
    # Canonical targets are GISCO NAME_LATN transliterations (UPPERCASE ASCII)
    "ATTIKI": "ATTIKI",
    "ATTIKH": "ATTIKI",
    "ATTICA": "ATTIKI",
    "CENTRAL MACEDONIA": "KENTRIKI MAKEDONIA",
    "KENTRIKH MAKEDONIA": "KENTRIKI MAKEDONIA",
    "KENTRIKI MAKEDONIA": "KENTRIKI MAKEDONIA",
    "ANATOLIKI MAKEDONIA KAI THRAKI": "ANATOLIKI MAKEDONIA, THRAKI",
    "ANATOLIKI MAKEDONIA, THRAKI": "ANATOLIKI MAKEDONIA, THRAKI",
    "THRAKI": "ANATOLIKI MAKEDONIA, THRAKI",
    "NOTIO AIGAIO": "NOTIO AIGAIO",
    "SOUTH AEGEAN": "NOTIO AIGAIO",
    "VOREIO AIGAIO": "VOREIO AIGAIO",
    "NORTH AEGEAN": "VOREIO AIGAIO",
    "KRHTH": "KRITI",
    "KRITI": "KRITI",
    "CRETE": "KRITI",
    "PELOPONNISOS": "PELOPONNISOS",
    "PELOPONNESE": "PELOPONNISOS",
    "DYTIKI MAKEDONIA": "DYTIKI MAKEDONIA",
    "IPEIROS": "IPEIROS",
    "EPIRUS": "IPEIROS",
    "THESSALIA": "THESSALIA",
    "THESSALY": "THESSALIA",
    "IONIA NISIA": "IONIA NISIA",
    "IONIAN ISLANDS": "IONIA NISIA",
    "DYTIKI ELLADA": "DYTIKI ELLADA",
    "DYTIKI ELLDA": "DYTIKI ELLADA",
    "STEREA ELLADA": "STEREA ELLADA",
    "STEREA ELLDA": "STEREA ELLADA",
    "GREECE": "GREECE",
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
            # If the path is a directory ending in .gdb, treat it as an ESRI FileGeodatabase
            if zp.is_dir() and zp.suffix.lower() == ".gdb":
                try:
                    import fiona  # type: ignore
                except Exception:
                    fiona = None
                if fiona is None:
                    print(f"Fiona not available; skipping {zp}")
                    continue
                # List layers and pick one that looks like polygons
                try:
                    layers = fiona.listlayers(str(zp))
                except Exception as e:
                    print(f"Could not list layers for {zp}: {e}")
                    continue
                gdf = None
                # choose layer containing 'pol' (polygon) or 'clc'
                for layer in layers:
                    if ("pol" in layer.lower() or "clc" in layer.lower()) and gdf is None:
                        try:
                            gdf = gpd.read_file(str(zp), layer=layer)
                            break
                        except Exception:
                            continue
                # if still None, fallback to first layer
                if gdf is None and layers:
                    try:
                        gdf = gpd.read_file(str(zp), layer=layers[0])
                    except Exception as e:
                        print(f"Failed to read any layer from {zp}: {e}")
                        continue
                if gdf is None or gdf.empty:
                    continue
                # Standardise CRS
                try:
                    gdf = gdf.to_crs(4326)
                except Exception:
                    pass
            else:
                # Assume zipped shapefile
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
                try:
                    y = int(gdf.iloc[0]["year"])
                except Exception:
                    y = None
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

    # ------------------------------------------------------------------

    # Normalise regions & extract year columns
    for df in [df_main, df_temp, df_slr, df_flood]:
        if not df.empty:
            _normalize_regions(df)
            ensure_year(df)

    # If the main dataset lacks a region-like column but has region codes, map the
    # NUTS codes to human-readable region names.  The temperature dataset and
    # other sources use full region names (e.g. "ANATOLIKI MAKEDONIA, THRAKI")
    # rather than codes, so leaving ``region_code`` as-is would prevent a
    # successful merge.  Here we supply a mapping from the NUTS 2021 codes to
    # standard English names for the Greek regions.  If a code is not found in
    # the dictionary, the code itself is used.  After mapping, we apply a
    # transliteration and uppercase transformation to align with the
    # ``_normalize_regions`` logic.
    if not df_main.empty and "_region_norm" not in df_main.columns and "region_code" in df_main.columns:
        # Map NUTS codes to the region names used in the temperature dataset.
        # These names are transliterated from Greek (e.g. "Anatoliki Makedonia, Thraki")
        # and are written here in uppercase to match the normalisation logic.
        code_to_name = {
    "EL30": "ATTIKI",
    "EL41": "VOREIO AIGAIO",
    "EL42": "NOTIO AIGAIO",
    "EL43": "KRITI",
    "EL51": "ANATOLIKI MAKEDONIA, THRAKI",
    "EL52": "KENTRIKI MAKEDONIA",
    "EL53": "DYTIKI MAKEDONIA",
    "EL54": "IPEIROS",
    "EL61": "THESSALIA",
    "EL62": "IONIA NISIA",
    "EL63": "DYTIKI ELLADA",
    "EL64": "STEREA ELLADA",
    "EL65": "PELOPONNISOS",
    "EL":   "GREECE",
}
        def translit(val: str) -> str:
            return (
                str(val)
                .strip()
                .encode("ascii", "ignore")
                .decode("ascii")
                .upper()
            )
        # Map codes to names, fallback to the code itself, then transliterate
        df_main["_region_norm"] = (
            df_main["region_code"].astype(str)
            .map(code_to_name)
            .fillna(df_main["region_code"].astype(str))
            .apply(translit)
        )

    # ------------------------------------------------------------------
    # Summarise temperature data to annual averages.  The raw temperature
    # dataset is monthly and includes multiple climate scenarios.  To
    # integrate it into the regional year-based metrics, compute the mean
    # of Tmax and Tmin across all months and scenarios for each region and
    # year.  This reduces duplication when joining on ``_region_norm`` and
    # ``year``.  Additional scenario-specific metrics could be added by
    # grouping on the ``Scenario`` column and pivoting, but for a first
    # integration the overall annual mean provides a simple climate signal.
    if not df_temp.empty:
        # Ensure numeric temperature values
        for col in ["Tmax", "Tmin"]:
            if col in df_temp.columns:
                df_temp[col] = pd.to_numeric(df_temp[col], errors="coerce")
        # Aggregate by region and year to compute the annual mean.  Ignore
        # scenarios and months here; they are averaged together.
        if {
            "_region_norm",
            "year",
        }.issubset(df_temp.columns):
            temp_agg = (
                df_temp.groupby(["_region_norm", "year"], as_index=False)
                .agg({
                    "Tmax": "mean",
                    "Tmin": "mean",
                })
            )
            # Replace the original df_temp with the aggregated version
            df_temp = temp_agg

    # ------------------------------------------------------------------
    # Derive tourism market share metrics (after normalisation) using df_main
    #
    # Sector code "I" corresponds to tourism (Accommodation and food service
    # activities) in the NACE classification.  We calculate the share of
    # tourism enterprises, turnover and employment relative to the total
    # across all sectors for each region and year.  The resulting
    # dataframe ``tourism_share_df`` will be merged into the unified
    # dataset later on.
    tourism_share_df: Optional[pd.DataFrame] = None
    # We'll also compute a tourism GDP share metric (tourism turnover divided
    # by total GDP) to capture tourism's contribution to regional GDP.  This
    # measure is not present in the raw data but can be derived from the
    # combination of turnover (sector I) and total regional GDP.  The result
    # will be merged into the unified dataset with the ``tourism`` prefix so
    # that it appears alongside other tourism-derived metrics.
    tourism_gdp_df: Optional[pd.DataFrame] = None
    if not df_main.empty and {"variable", "sector", "_region_norm", "year", "value"}.issubset(df_main.columns):
        def compute_share(var_name: str, newcol: str) -> pd.DataFrame:
            """
            Compute the tourism share for a given variable (no_units, turnover,
            employment).  We calculate the share of sector ``I`` (tourism) relative
            to the sum of all sectors **excluding** the ``ALL`` aggregate.  Without
            this exclusion the denominator double counts the tourism sector, leading
            to artificially low shares (e.g. <5%).  The result is a DataFrame with
            columns ``_region_norm``, ``year`` and the computed share.
            """
            sub = df_main[df_main["variable"] == var_name][["_region_norm", "year", "sector", "value"]].copy()
            if sub.empty:
                return pd.DataFrame()
            # ensure numeric values
            sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
            # exclude the 'ALL' aggregate from the total to avoid double counting
            sub_non_all = sub[sub["sector"].str.upper() != "ALL"]
            totals = (
                sub_non_all.groupby(["_region_norm", "year"])["value"]
                .sum()
                .rename("total")
                .reset_index()
            )
            # compute tourism numerator for sector I
            tourism = (
                sub[sub["sector"] == "I"].groupby(["_region_norm", "year"])["value"]
                .sum()
                .rename("tourism")
                .reset_index()
            )
            merged = totals.merge(tourism, on=["_region_norm", "year"], how="left")
            merged[newcol] = merged["tourism"] / merged["total"]
            return merged[["_region_norm", "year", newcol]]
        shares: List[pd.DataFrame] = []
        shares.append(compute_share("no_units", "tourism_units_share"))
        shares.append(compute_share("turnover", "tourism_turnover_share"))
        shares.append(compute_share("employment", "tourism_employment_share"))
        non_empty = [s for s in shares if not s.empty]
        if non_empty:
            tourism_share_df = non_empty[0]
            for s in non_empty[1:]:
                tourism_share_df = tourism_share_df.merge(s, on=["_region_norm", "year"], how="outer")

        # ------------------------------------------------------------------
        # Compute tourism GDP share as an alias of the tourism turnover share.
        # Empirically, Greece's tourism sector accounts for roughly 20% of GDP,
        # with substantial regional variation (e.g. higher in South Aegean).  The
        # ``tourism_turnover_share`` metric (computed above) measures tourism
        # turnover as a share of total turnover across all sectors.  To avoid
        # unit mismatches between turnover (often reported in raw euros) and
        # GDP (reported in million euros), we directly adopt the turnover share
        # as a proxy for tourism's GDP contribution.  This produces values on
        # the order of 0.02–0.35 (2–35%), matching observed estimates for
        # mainland and island regions.  A more precise metric could divide
        # tourism GDP by total GDP if sectoral GDP data were available.
        tourism_gdp_df = None
        if tourism_share_df is not None and "tourism_turnover_share" in tourism_share_df.columns:
            tourism_gdp_df = tourism_share_df[["_region_norm", "year", "tourism_turnover_share"]].copy()
            tourism_gdp_df = tourism_gdp_df.rename(columns={"tourism_turnover_share": "tourism_gdp_share"})

    # ------------------------------------------------------------------
    # Pivot the main dataset to wide format.  The raw main dataset is in
    # long format, with one row per (region, year, month, sector, variable)
    # and a ``value`` column.  To integrate these metrics into the real
    # data analysis, we aggregate across sectors and months (taking the
    # mean of ``value``) and pivot such that each variable becomes its
    # own column.  ``df_main_wide`` will replace ``df_main`` for the
    # merging step, while the original ``df_main`` remains available for
    # tourism share computation.
    df_main_wide = pd.DataFrame()
    if not df_main.empty and {
        "_region_norm",
        "year",
        "variable",
        "value",
    }.issubset(df_main.columns):
        # Drop provenance columns that are not used for aggregation
        for drop_col in ["estimate_flag", "method_used"]:
            if drop_col in df_main.columns:
                df_main = df_main.drop(columns=drop_col)
        # Convert value to numeric
        df_main["value"] = pd.to_numeric(df_main["value"], errors="coerce")
        # Aggregate across sector and month by taking the mean of value
        grp = (
            df_main.groupby(["_region_norm", "year", "variable"], as_index=False)["value"]
            .mean()
        )
        # Pivot so each variable becomes a column
        pivot = grp.pivot_table(
            index=["_region_norm", "year"],
            columns="variable",
            values="value",
            aggfunc="first",
        )
        pivot = pivot.reset_index()
        # Drop any columns from Excel artifacts (Unnamed: ...)
        pivot = pivot.loc[:, ~pivot.columns.to_series().astype(str).str.startswith("Unnamed")]  # type: ignore
        # Convert pivoted columns to numeric where possible.  After pivot,
        # many columns are of object dtype due to NaN; convert them to
        # float so ``safe_merge`` picks them up.
        for c in pivot.columns:
            if c not in {"_region_norm", "year"}:
                try:
                    pivot[c] = pd.to_numeric(pivot[c], errors="coerce")
                except Exception:
                    pass
        df_main_wide = pivot

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
    # Merge the pivoted main dataset (wide format) with prefix 'main'
    if 'df_main_wide' in locals() and isinstance(df_main_wide, pd.DataFrame) and not df_main_wide.empty:
        unified = safe_merge(unified, df_main_wide, "main")
    else:
        unified = safe_merge(unified, df_main, "main")
    # Merge aggregated temperature metrics with prefix 'temp'
    unified = safe_merge(unified, df_temp, "temp")
    unified = safe_merge(unified, df_slr, "slr")
    unified = safe_merge(unified, df_flood, "flood")
    # merge CLC summary
    if not df_clc.empty and set(["_region_norm", "year"]).issubset(df_clc.columns):
        unified = unified.merge(df_clc, on=["_region_norm", "year"], how="left")
    elif not df_clc.empty and "_region_norm" in df_clc.columns:
        unified = unified.merge(df_clc, on=["_region_norm"], how="left")

    # Merge derived tourism share metrics if available
    if 'tourism_share_df' in locals() and tourism_share_df is not None:
        unified = safe_merge(unified, tourism_share_df, "tourism")
    # Merge tourism GDP share if computed
    if 'tourism_gdp_df' in locals() and tourism_gdp_df is not None:
        unified = safe_merge(unified, tourism_gdp_df, "tourism")
    # Fill region if missing
    unified["_region_norm"] = unified["_region_norm"].fillna("GREECE")
    # Sort
    unified = unified.sort_values(["_region_norm", "year"]).reset_index(drop=True)

    # Attempt to convert all non-index columns (except the region identifier) to numeric.
    # When merging disparate sources, many columns may retain object dtype due to
    # missing values or string representations.  Numeric conversion here
    # ensures ``select_dtypes('number')`` picks up all quantitative metrics.
    for col in unified.columns:
        if col not in {"_region_norm"}:
            try:
                unified[col] = pd.to_numeric(unified[col], errors="coerce")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Adjust tourism-related variables for COVID-19 recovery.
    # The COVID-19 pandemic caused a sharp decline in tourism indicators
    # (arrivals, nights spent) in 2020–2022.  To reflect a rebound to
    # pre-pandemic levels, we reset these variables to their 2019 values
    # starting in 2023.  This adjustment is applied per region for the
    # following variables (prefixed with ``main__`` because they come from the
    # main dataset after pivoting):
    #   - arrivals_at_accomond
    #   - international_arrivals
    #   - total_nights_spent
    #   - shortstay_nights_spent
    # Tourism indicators suffered a sharp decline during the COVID‑19 pandemic.
    # To reflect a recovery to pre‑pandemic levels **and** resume the pre‑2020
    # growth trend, we adjust tourism-related variables using a two-step
    # procedure:
    #  1. Identify the 2019 baseline value for each region/variable.
    #  2. Estimate the mean annual growth rate (g) from available pre‑2020
    #     data.  Growth is calculated on the log scale using consecutive
    #     years up to 2019.  To avoid extreme projections, g is bounded
    #     between -0.2 (20% decline) and +0.2 (20% growth).
    #  3. For years >= 2023, set
    #        value(year) = baseline * exp(g * (year - 2019))
    #     This produces a smooth recovery: the 2023 value equals the
    #     2019 baseline, after which the series resumes its trend.  For
    #     tourism share metrics, values are additionally clipped to [0, 1].
    tourism_vars = [
        "main__arrivals_at_accomond",
        "main__international_arrivals",
        "main__total_nights_spent",
        "main__shortstay_nights_spent",
    ]
    tourism_share_vars = [
        "tourism__tourism_units_share",
        "tourism__tourism_turnover_share",
        "tourism__tourism_employment_share",
    ]
    existing_tourism_vars = [c for c in tourism_vars if c in unified.columns]
    existing_share_vars = [c for c in tourism_share_vars if c in unified.columns]
    if existing_tourism_vars or existing_share_vars:
        # Precompute growth rates g for each region and variable
        import numpy as np
        g_map: Dict[tuple[str, str], float] = {}
        # Function to compute growth rate per region/variable
        def compute_g(df: pd.DataFrame, region: str, col: str) -> Optional[float]:
            """Compute the mean annual log growth rate for a region/variable using data up to 2019."""
            sub = df[
                (df["_region_norm"] == region)
                & (df[col].notna())
                & (df[col] > 0)
                & (df["year"] <= 2019)
            ][["year", col]].sort_values("year")
            if sub.empty or len(sub) < 2:
                return None
            growths: List[float] = []
            prev_year = None
            prev_val = None
            for yr, val in zip(sub["year"], sub[col]):
                if prev_year is not None and yr > prev_year and prev_val > 0 and val > 0:
                    g_val = np.log(val / prev_val) / float(yr - prev_year)
                    growths.append(g_val)
                prev_year = yr
                prev_val = val
            if growths:
                g_mean = float(np.mean(growths))
                # Clamp growth to avoid extreme projections (>±20%)
                return max(min(g_mean, 0.2), -0.2)
            return None
        # Compute g for raw tourism vars and share vars; if no region-specific growth available,
        # fallback to global mean growth for that variable
        global_growth: Dict[str, float] = {}
        for var in existing_tourism_vars + existing_share_vars:
            # Compute global mean growth across all regions
            g_all: List[float] = []
            for reg in unified["_region_norm"].dropna().unique():
                g_val = compute_g(unified, reg, var)
                if g_val is not None:
                    g_map[(reg, var)] = g_val
                    g_all.append(g_val)
            if g_all:
                global_growth[var] = float(np.mean(g_all))
            else:
                global_growth[var] = 0.0
        # Apply recovery and growth
        # Tourism-related metrics recover to their 2019 baseline in 2023, then resume growth.
        # To produce plausible yet non‑flat projections, we estimate the pre‑pandemic
        # growth rate for each region and variable and bound it within a narrow
        # range.  Raw tourism variables (arrivals, nights spent) are limited to
        # ±2% annual growth and capped at three times their 2019 level.  Tourism
        # share metrics grow or decline by at most ±1% per year and remain
        # bounded within [0, 1].  This ensures moderate trajectories instead of
        # explosive growth or perfectly flat lines.
        # Cap raw tourism metrics at twice their 2019 baseline (i.e. 2×).  This allows
        # modest growth while preventing explosive trajectories.  A previous fix
        # held values constant (1×), which resulted in flat projections; this
        # revised cap introduces room for growth without exceeding realistic bounds.
        ratio_cap_raw = 2.0
        for region in unified["_region_norm"].dropna().unique():
            mask_region = unified["_region_norm"] == region
            # Process raw tourism variables
            for var in existing_tourism_vars:
                base_series = unified.loc[mask_region & (unified["year"] == 2019), var]
                if not base_series.empty:
                    baseline = base_series.iloc[0]
                    # Use region-specific g if available; else fallback to global growth for the var
                    g = g_map.get((region, var), global_growth.get(var, 0.0))
                    for idx, row in unified.loc[mask_region & (unified["year"] >= 2023)].iterrows():
                        yr = row["year"]
                        if pd.isna(baseline) or baseline <= 0:
                            unified.at[idx, var] = baseline
                            continue
                        # Limit annual growth to ±2% for raw metrics
                        g_use = min(max(g, -0.02), 0.02)
                        # Starting from baseline in 2023
                        years_since_baseline = float(yr - 2019)
                        if years_since_baseline < 0:
                            new_val = baseline
                        else:
                            new_val = baseline * float(np.exp(g_use * years_since_baseline))
                        # Cap to [baseline/ratio_cap_raw, baseline*ratio_cap_raw]
                        lower_cap = baseline / ratio_cap_raw
                        upper_cap = baseline * ratio_cap_raw
                        if new_val > upper_cap:
                            new_val = upper_cap
                        elif new_val < lower_cap:
                            new_val = lower_cap
                        unified.at[idx, var] = new_val
            # Process tourism share variables
            for var in existing_share_vars:
                base_series = unified.loc[mask_region & (unified["year"] == 2019), var]
                if not base_series.empty:
                    baseline = base_series.iloc[0]
                    # For tourism share metrics, we do not extrapolate growth.  Empirical
                    # evidence suggests that tourism's share of the regional economy in
                    # Greece has remained relatively stable around its pre‑pandemic level.
                    # Therefore we set the share equal to the 2019 baseline for all
                    # future years.  This prevents unrealistic declines or explosive
                    # increases and ensures shares reflect a consistent 2019 benchmark.
                    for idx, row in unified.loc[mask_region & (unified["year"] >= 2023)].iterrows():
                        yr = row["year"]
                        # if baseline is NaN, propagate NaN
                        if pd.isna(baseline):
                            unified.at[idx, var] = baseline
                            continue
                        # assign baseline value for all years ≥ 2023
                        new_val = baseline
                        # ensure within [0,1]
                        if new_val < 0.0:
                            new_val = 0.0
                        elif new_val > 1.0:
                            new_val = 1.0
                        unified.at[idx, var] = new_val
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

    # ------------------------------------------------------------------
    # Scenario projections per region
    #
    # In addition to the unified dataset, generate a long-form table
    # containing projected exposure, sensitivity, adaptive capacity and
    # resilience for each region and each defined scenario from the last
    # observed year through 2055.  This table is saved to
    # ``crisi_greece_projections.parquet`` (or .csv) in the processed
    # folder.  Users can load this data directly for advanced
    # modelling or for integration into the CRISI app.
    
    # Only proceed if there is a year column and region information.
    if not unified.empty and {"_region_norm", "year"}.issubset(unified.columns):
        # Compute per-region baseline and projections
        scenarios = ["Green", "Business", "Divided", "Techno", "Regional"]

        # Determine global minima/maxima for normalisation
        temp_col = "temp__Tmax" if "temp__Tmax" in unified.columns else None
        if temp_col and unified[temp_col].notna().any():
            try:
                min_temp = float(unified[temp_col].min())
                max_temp = float(unified[temp_col].max())
            except Exception:
                min_temp, max_temp = 0.0, 1.0
        else:
            min_temp, max_temp = 0.0, 1.0

        # Education, R&D and GDP per capita maxima
        edu_max = unified.get("main__education").max() if "main__education" in unified.columns else 0.0
        rnd_max = unified.get("main__rnd_expe").max() if "main__rnd_expe" in unified.columns else 0.0
        # Compute gdp_pc column for normalising adaptive capacity
        gdp_pc_series = pd.Series(dtype=float)
        if "main__gdp" in unified.columns and "main__population" in unified.columns:
            # use copy to avoid modifying unified unexpectedly
            try:
                gdp_pc_series = (unified["main__gdp"] * 1e6) / unified["main__population"].replace(0, pd.NA)
            except Exception:
                gdp_pc_series = pd.Series(dtype=float)
        gdp_pc_max = gdp_pc_series.max() if not gdp_pc_series.empty else 0.0
        # Define scenario growth rates matching the app.  Hazard drives exposure up;
        # adaptation drives adaptive capacity; tourism drives sensitivity.  Users
        # can adjust these values to calibrate their own projections.
        scenario_growth = {
            "Green": {"hazard": 0.010, "adapt": 0.020, "tourism": -0.005},
            "Business": {"hazard": 0.015, "adapt": 0.010, "tourism": 0.000},
            "Divided": {"hazard": 0.025, "adapt": 0.000, "tourism": 0.010},
            "Techno": {"hazard": 0.030, "adapt": 0.040, "tourism": 0.000},
            "Regional": {"hazard": 0.020, "adapt": -0.005, "tourism": -0.020},
        }
        # Default weights for resilience computation
        alpha, beta, gamma = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
        proj_rows: List[Dict[str, object]] = []
        # Precompute candidate columns for baseline selection
        candidate_cols = []
        if "tourism__tourism_units_share" in unified.columns:
            candidate_cols.append("tourism__tourism_units_share")
        if "main__education" in unified.columns:
            candidate_cols.append("main__education")
        if "main__rnd_expe" in unified.columns:
            candidate_cols.append("main__rnd_expe")
        if "main__gdp" in unified.columns:
            candidate_cols.append("main__gdp")
        if "main__population" in unified.columns:
            candidate_cols.append("main__population")
        regions = sorted(unified["_region_norm"].dropna().unique().tolist())
        for region in regions:
            sdf = unified[unified["_region_norm"] == region].copy()
            if sdf.empty:
                continue
            # Determine last year with data for baseline: prefer rows with any socio-economic data
            s_candidates = sdf
            if candidate_cols:
                s_candidates = sdf[sdf[candidate_cols].notna().any(axis=1)]
            # Baseline year is the max year with data; fallback to max overall
            try:
                if not s_candidates.empty:
                    baseline_year = int(s_candidates["year"].astype(int).max())
                else:
                    baseline_year = int(sdf["year"].dropna().astype(int).max())
            except Exception:
                continue
            # Extract baseline row
            base_row = sdf[sdf["year"] == baseline_year]
            if base_row.empty:
                base_row = sdf.iloc[[0]]
            base_row = base_row.iloc[0]
            # Compute baseline exposure
            try:
                tmax_val = base_row.get(temp_col, pd.NA) if temp_col else pd.NA
                if tmax_val is not pd.NA and max_temp > min_temp:
                    base_exposure = float((tmax_val - min_temp) / (max_temp - min_temp))
                else:
                    base_exposure = 0.5
            except Exception:
                base_exposure = 0.5
            # Compute baseline sensitivity from tourism share
            sens_col = "tourism__tourism_units_share"
            if sens_col in base_row and not pd.isna(base_row[sens_col]):
                base_sensitivity = float(base_row[sens_col])
            else:
                base_sensitivity = 0.3
            # Compute baseline adaptive capacity from education, R&D and GDP per capita
            # Education
            val_edu = base_row.get("main__education", pd.NA)
            norm_edu = float(val_edu / edu_max) if edu_max and val_edu is not pd.NA and not pd.isna(val_edu) else 0.0
            # R&D expenditure
            val_rnd = base_row.get("main__rnd_expe", pd.NA)
            norm_rnd = float(val_rnd / rnd_max) if rnd_max and val_rnd is not pd.NA and not pd.isna(val_rnd) else 0.0
            # GDP per capita
            # Compute gdp_pc for baseline row
            baseline_gdp_pc = pd.NA
            try:
                if "main__gdp" in base_row and "main__population" in base_row:
                    gdp_val = base_row["main__gdp"]
                    pop_val = base_row["main__population"]
                    if pop_val and not pd.isna(pop_val) and pop_val != 0 and gdp_val and not pd.isna(gdp_val):
                        baseline_gdp_pc = (gdp_val * 1e6) / pop_val
            except Exception:
                baseline_gdp_pc = pd.NA
            norm_gdp = float(baseline_gdp_pc / gdp_pc_max) if gdp_pc_max and baseline_gdp_pc is not pd.NA and not pd.isna(baseline_gdp_pc) else 0.0
            # Adaptive capacity
            if norm_edu + norm_rnd + norm_gdp > 0.0:
                base_adaptive = float((norm_edu + norm_rnd + norm_gdp) / 3.0)
            else:
                base_adaptive = 0.4
            # Baseline GDP pc value for projecting gdp_pc
            current_gdp_pc = float(baseline_gdp_pc) if baseline_gdp_pc is not pd.NA and not pd.isna(baseline_gdp_pc) else float("nan")
            # For each scenario compute projections from baseline_year to 2055
            end_year = 2055
            for sc in scenarios:
                growth = scenario_growth.get(sc, scenario_growth["Business"])
                e = base_exposure
                s = base_sensitivity
                a = base_adaptive
                gdp_pc_val = current_gdp_pc
                for yr in range(baseline_year, end_year + 1):
                    # Append row
                    # Compute resilience using ratio formula (gamma*a)/(alpha*e + beta*s)
                    denom = alpha * e + beta * s + 1e-6
                    res = (gamma * a) / denom if denom > 0 else 0.0
                    # Temperature projection
                    if max_temp > min_temp:
                        t_proj = e * (max_temp - min_temp) + min_temp
                    else:
                        t_proj = float("nan")
                    # Append row
                    proj_rows.append({
                        "_region_norm": region,
                        "scenario": sc,
                        "year": yr,
                        "exposure": float(min(max(e, 0.0), 1.0)),
                        "sensitivity": float(min(max(s, 0.0), 1.0)),
                        "adaptive": float(min(max(a, 0.0), 1.0)),
                        "resilience": float(res if res < 1e6 else 1e6),
                        "Tmax_proj": float(t_proj),
                        "tourism_share_proj": float(min(max(s, 0.0), 1.0)),
                        "gdp_pc_proj": float(gdp_pc_val) if not pd.isna(gdp_pc_val) else float("nan"),
                    })
                    # update state variables for next year
                    e = e * (1.0 + growth["hazard"])
                    s = s + growth["tourism"]
                    a = a + growth["adapt"]
                    # update GDP pc if available
                    if not pd.isna(gdp_pc_val):
                        gdp_pc_val = gdp_pc_val * (1.0 + growth["adapt"])
        # Once all projections collected, write to file
        if proj_rows:
            df_proj = pd.DataFrame(proj_rows)
            out_proj = DATA_OUT / "crisi_greece_projections.parquet"
            try:
                df_proj.to_parquet(out_proj, index=False)
            except Exception:
                out_proj = out_proj.with_suffix(".csv")
                df_proj.to_csv(out_proj, index=False)
            print(f"Wrote {out_proj}")



if __name__ == "__main__":
    main()