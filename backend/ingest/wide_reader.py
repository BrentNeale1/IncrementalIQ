"""Auto-detect wide-format CSVs and convert to the long format expected by the pipeline.

Also handles multi-sheet Excel workbooks where each sheet is a data source
(Google Ads, Meta, Shopify, GA4) — sheets are merged on date before conversion.
"""

from io import BytesIO
import pandas as pd
from backend.ingest.csv_reader import ValidationError


# Column prefix → canonical channel name
PREFIX_TO_CHANNEL = {
    "ga_search": "google_search",
    "ga_shopping": "google_shopping",
    "ga_pmax": "google_pmax",
    "ga_youtube": "google_youtube",
    "ga_dgen": "google_demand_gen",
    "meta_fb": "meta_feed",
    "meta_insta": "meta_instagram",
    "meta_stories": "meta_stories",
}

# Metric suffix → long-format column name
SUFFIX_TO_METRIC = {
    "_impr": "impressions",
    "_clicks": "clicks",
    "_cost": "spend",
    "_conv": "in_platform_conversions",
}

# Site-wide columns mapped directly (present once per date row, repeated across channel rows)
SITE_WIDE_COLUMNS = {
    "shopify_revenue": "revenue",
    "shopify_orders": "orders",
    "ga4_sessions_organic": "sessions_organic",
    "ga4_sessions_direct": "sessions_direct",
    "ga4_sessions_email": "sessions_email",
    "ga4_sessions_referral": "sessions_referral",
}


def detect_format(df: pd.DataFrame) -> str:
    """Detect whether a DataFrame is long or wide format.

    Returns "long" if channel and campaign columns exist.
    Returns "wide" if no channel column but pattern-matched metric columns found.
    Raises ValidationError if neither pattern matches.
    """
    cols = set(df.columns.str.strip().str.lower())
    if "channel" in cols and "campaign" in cols:
        return "long"

    # Check for wide-format metric columns: prefix_suffix pattern
    suffixes = tuple(SUFFIX_TO_METRIC.keys())
    has_metric_cols = any(c.endswith(suffixes) for c in cols)
    if has_metric_cols:
        return "wide"

    raise ValidationError(
        "Unrecognised CSV format",
        details=[
            "Expected either long format (with 'channel' and 'campaign' columns) "
            "or wide format (with columns like ga_search_cost, meta_fb_impr, etc.)"
        ],
    )


def wide_to_long(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convert a wide-format DataFrame to long format.

    Wide format has one row per date with columns like ga_search_cost, meta_fb_impr, etc.
    Long format has one row per date per channel with standard column names.

    Returns (long_df, warnings).
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    warnings: list[str] = []

    if "date" not in df.columns:
        raise ValidationError("Missing 'date' column in wide-format CSV")

    # Identify which prefixes and site-wide columns are present
    suffixes = tuple(SUFFIX_TO_METRIC.keys())
    found_prefixes: dict[str, dict[str, str]] = {}  # prefix → {suffix → column_name}
    site_wide_found: dict[str, str] = {}  # long_name → wide_column_name
    classified_cols = {"date"}

    for col in df.columns:
        if col == "date":
            continue

        # Check site-wide columns
        if col in SITE_WIDE_COLUMNS:
            site_wide_found[SITE_WIDE_COLUMNS[col]] = col
            classified_cols.add(col)
            continue

        # Check metric columns (prefix_suffix)
        matched = False
        for suffix in SUFFIX_TO_METRIC:
            if col.endswith(suffix):
                prefix = col[: -len(suffix)]
                if prefix in PREFIX_TO_CHANNEL:
                    found_prefixes.setdefault(prefix, {})[suffix] = col
                    classified_cols.add(col)
                    matched = True
                break  # only one suffix can match

        if not matched and col not in classified_cols:
            # Check if it looks like a metric column with unknown prefix
            for suffix in SUFFIX_TO_METRIC:
                if col.endswith(suffix):
                    prefix = col[: -len(suffix)]
                    warnings.append(
                        f"Unknown column prefix '{prefix}' in column '{col}' — skipped"
                    )
                    classified_cols.add(col)
                    matched = True
                    break

            if not matched:
                # Completely unrecognised column
                warnings.append(f"Unrecognised column '{col}' — skipped")

    if not found_prefixes:
        raise ValidationError(
            "No recognised channel columns found in wide-format CSV",
            details=[
                "Expected columns like ga_search_cost, meta_fb_impr, etc. "
                f"Known prefixes: {', '.join(sorted(PREFIX_TO_CHANNEL.keys()))}"
            ],
        )

    # Build long-format rows
    long_rows = []
    for _, row in df.iterrows():
        date_val = row["date"]

        # Collect site-wide values for this date
        site_wide_vals = {}
        for long_name, wide_col in site_wide_found.items():
            site_wide_vals[long_name] = row[wide_col]

        for prefix, suffix_map in found_prefixes.items():
            channel = PREFIX_TO_CHANNEL[prefix]
            long_row = {
                "date": date_val,
                "channel": channel,
                "campaign": channel,  # no campaign breakdown in wide format
            }

            # Fill metrics from columns, default to 0 if suffix not present
            for suffix, metric_name in SUFFIX_TO_METRIC.items():
                if suffix in suffix_map:
                    long_row[metric_name] = row[suffix_map[suffix]]
                else:
                    long_row[metric_name] = 0

            # Fill site-wide values, default to 0
            for long_name in SITE_WIDE_COLUMNS.values():
                long_row[long_name] = site_wide_vals.get(long_name, 0)

            long_rows.append(long_row)

    long_df = pd.DataFrame(long_rows)
    return long_df, warnings


# GA4 source label → canonical column name
GA4_SOURCE_MAP = {
    "organic search": "sessions_organic",
    "direct": "sessions_direct",
    "email": "sessions_email",
    "referral": "sessions_referral",
}


def _pivot_ga4_sheet(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pivot a GA4 long-format sheet (date, source, sessions) into wide columns.

    Input:  date | source         | sessions
            ...  | Organic Search | 320
            ...  | Direct         | 180

    Output: date | sessions_organic | sessions_direct | sessions_email | sessions_referral
            ...  | 320              | 180              | 0              | 0

    Missing source/day combinations default to 0.
    """
    warnings: list[str] = []
    df = df.copy()
    df["source"] = df["source"].str.strip().str.lower()

    # Map source labels to canonical names
    df["col_name"] = df["source"].map(GA4_SOURCE_MAP)

    unknown = df[df["col_name"].isna()]["source"].unique()
    for src in sorted(unknown):
        warnings.append(f"Unknown GA4 source '{src}' — skipped")
    df = df[df["col_name"].notna()]

    # Pivot: one row per date, one column per session type
    pivoted = df.pivot_table(
        index="date", columns="col_name", values="sessions",
        aggfunc="sum", fill_value=0,
    ).reset_index()

    # Ensure all 4 session columns exist (fill missing with 0)
    for col in GA4_SOURCE_MAP.values():
        if col not in pivoted.columns:
            pivoted[col] = 0

    return pivoted, warnings


def read_excel_sheets(file_bytes: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Read a multi-sheet Excel workbook and combine into a single long-format DataFrame.

    Handles three kinds of sheets:
    - Channel sheets (have 'channel' column): long-format ad data (Google Ads, Meta).
      These are stacked vertically (concatenated).
    - GA4 sheets (have 'source' and 'sessions' columns): pivoted from long to wide,
      then merged by date.
    - Other site-wide sheets (no 'channel' column): metrics like revenue, orders.
      These are merged onto the channel data by date.

    Returns (combined_long_df, warnings).
    """
    warnings: list[str] = []
    sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None, engine="openpyxl")

    if not sheets:
        raise ValidationError("Excel file contains no sheets")

    channel_dfs: list[pd.DataFrame] = []
    site_wide_dfs: list[pd.DataFrame] = []

    for sheet_name, sheet_df in sheets.items():
        sheet_df.columns = sheet_df.columns.str.strip().str.lower().str.replace(" ", "_")

        if "date" not in sheet_df.columns:
            warnings.append(f"Sheet '{sheet_name}' has no 'date' column — skipped")
            continue

        if sheet_df.empty:
            warnings.append(f"Sheet '{sheet_name}' is empty — skipped")
            continue

        # Drop any fully-empty columns
        sheet_df = sheet_df.dropna(axis=1, how="all")

        # Normalise date column to YYYY-MM-DD strings
        sheet_df["date"] = pd.to_datetime(
            sheet_df["date"].astype(str).str.strip(), format="mixed", dayfirst=False
        ).dt.strftime("%Y-%m-%d")

        if "channel" in sheet_df.columns:
            channel_dfs.append(sheet_df)
        elif "source" in sheet_df.columns and "sessions" in sheet_df.columns:
            # GA4 long format — pivot to wide
            pivoted, ga4_warnings = _pivot_ga4_sheet(sheet_df)
            warnings.extend(ga4_warnings)
            site_wide_dfs.append(pivoted)
        else:
            site_wide_dfs.append(sheet_df)

    if not channel_dfs and not site_wide_dfs:
        raise ValidationError("No sheets with a 'date' column found in Excel file")

    # Stack channel sheets vertically
    if channel_dfs:
        combined = pd.concat(channel_dfs, ignore_index=True)
    else:
        raise ValidationError(
            "No channel sheets found in Excel file",
            details=["At least one sheet must have a 'channel' column (e.g. Google Ads, Meta)"],
        )

    # Merge site-wide sheets by date
    for site_df in site_wide_dfs:
        overlapping = set(combined.columns) & set(site_df.columns) - {"date"}
        if overlapping:
            warnings.append(
                f"Site-wide sheet has overlapping columns: "
                f"{', '.join(sorted(overlapping))} — later values will overwrite"
            )
        # Left-merge broadcasts site-wide values (revenue, orders, sessions) onto
        # every channel row for that date.  This is intentional — data_prep.py uses
        # "max" aggregation when grouping by date to de-duplicate back to one value.
        combined = combined.merge(site_df, on="date", how="left", suffixes=("", "_dup"))
        dup_cols = [c for c in combined.columns if c.endswith("_dup")]
        if dup_cols:
            combined = combined.drop(columns=dup_cols)

    # Fill missing numeric values with 0 (blank cells in source data)
    numeric_cols = combined.select_dtypes(include="number").columns
    combined[numeric_cols] = combined[numeric_cols].fillna(0)

    return combined, warnings
