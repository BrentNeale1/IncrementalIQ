"""Data preparation: transform ingested records into model-ready DataFrames.

Takes raw daily_records (date × channel × campaign) and produces:
- X: DataFrame with date column, one spend column per channel, control columns
- y: Series of the target variable (revenue or orders) aggregated per day
- metadata: channel names, control column names for model configuration
"""
from dataclasses import dataclass, field
import pandas as pd
from sqlalchemy.orm import Session
from backend.db.models import DailyRecord


SESSION_CONTROLS = [
    "sessions_organic",
    "sessions_direct",
    "sessions_email",
    "sessions_referral",
]


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    date_column: str = "date"
    channel_columns: list[str] = field(default_factory=list)
    control_columns: list[str] = field(default_factory=list)
    target_variable: str = "revenue"
    daily_rows: int = 0


def query_records(db: Session, upload_id: int) -> pd.DataFrame:
    """Load all DailyRecords for an upload into a DataFrame."""
    records = db.query(DailyRecord).filter_by(upload_id=upload_id).all()
    if not records:
        raise ValueError(f"No records found for upload_id={upload_id}")

    rows = []
    for r in records:
        rows.append({
            "date": r.date,
            "channel": r.channel,
            "spend": r.spend,
            "in_platform_conversions": r.in_platform_conversions,
            "revenue": r.revenue,
            "orders": r.orders,
            "sessions_organic": r.sessions_organic,
            "sessions_direct": r.sessions_direct,
            "sessions_email": r.sessions_email,
            "sessions_referral": r.sessions_referral,
        })
    return pd.DataFrame(rows)


def prepare_model_data(
    df: pd.DataFrame,
    target: str = "revenue",
    channel_config: dict | None = None,
) -> PreparedData:
    """Transform raw records into model-ready X and y.

    Parameters
    ----------
    df : DataFrame with columns: date, channel, spend, in_platform_conversions,
         revenue, orders, sessions_organic, sessions_direct, sessions_email,
         sessions_referral
    target : "revenue" or "orders" — the outcome variable
    channel_config : Optional dict with keys:
        - "merge": dict mapping source channel names to merged names
        - "channels": list of channels to include (after merging)
        - "min_spend_pct": float, auto-drop channels below this % of total spend

    Returns
    -------
    PreparedData with X (features), y (target), and metadata
    """
    if target not in ("revenue", "orders"):
        raise ValueError(f"target must be 'revenue' or 'orders', got '{target}'")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # --- Apply channel_config: merge → filter ---
    if channel_config:
        merge_map = channel_config.get("merge")
        if merge_map:
            df["channel"] = df["channel"].replace(merge_map)

        channel_list = channel_config.get("channels")
        min_spend_pct = channel_config.get("min_spend_pct")

        if channel_list:
            df = df[df["channel"].isin(channel_list)]
        elif min_spend_pct is not None:
            total_spend = df["spend"].sum()
            if total_spend > 0:
                spend_by_ch = df.groupby("channel")["spend"].sum()
                spend_pct = spend_by_ch / total_spend * 100
                keep = spend_pct[spend_pct >= min_spend_pct].index.tolist()
                df = df[df["channel"].isin(keep)]

    if df.empty:
        raise ValueError("No data remaining after channel filtering")

    channels = sorted(df["channel"].unique())

    # --- Pivot spend per channel (sum across campaigns) ---
    spend_pivot = df.pivot_table(
        index="date", columns="channel", values="spend",
        aggfunc="sum", fill_value=0,
    )
    spend_columns = [f"spend_{ch}" for ch in spend_pivot.columns]
    spend_pivot.columns = spend_columns

    # --- Pivot in-platform conversions per channel (covariate, never outcome) ---
    ipc_pivot = df.pivot_table(
        index="date", columns="channel", values="in_platform_conversions",
        aggfunc="sum", fill_value=0,
    )
    ipc_columns = [f"ipc_{ch}" for ch in ipc_pivot.columns]
    ipc_pivot.columns = ipc_columns

    # --- Aggregate target and session controls by date ---
    # Session columns are site-level metrics; take max per date to avoid
    # double-counting if they appear on multiple channel rows.
    daily = df.groupby("date").agg(
        target_sum=(target, "max"),
        sessions_organic=("sessions_organic", "max"),
        sessions_direct=("sessions_direct", "max"),
        sessions_email=("sessions_email", "max"),
        sessions_referral=("sessions_referral", "max"),
    )

    # --- Assemble X ---
    X = spend_pivot.join(ipc_pivot).join(
        daily[SESSION_CONTROLS]
    ).reset_index()

    # Sort by date
    X = X.sort_values("date").reset_index(drop=True)

    # y aligned with X
    y_series = daily["target_sum"].reindex(X["date"]).reset_index(drop=True)
    y_series.name = target

    # Identify control columns (session controls + in-platform conversion columns)
    control_columns = SESSION_CONTROLS + ipc_columns

    return PreparedData(
        X=X,
        y=y_series,
        date_column="date",
        channel_columns=spend_columns,
        control_columns=control_columns,
        target_variable=target,
        daily_rows=len(X),
    )


META_PLACEMENTS = ["meta_feed", "meta_instagram", "meta_stories",
                    "audience_network", "messenger", "threads"]

DEFAULT_MIN_SPEND_PCT = 0.5


def recommend_channel_config(df: pd.DataFrame) -> dict:
    """Analyze spend distribution and return a suggested channel_config.

    Returns dict with keys: channels, merge, dropped, reasons.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    total_spend = df["spend"].sum()
    total_days = df["date"].nunique()

    # Build merge map for Meta placements
    present_meta = [ch for ch in META_PLACEMENTS if ch in df["channel"].values]
    merge_map = {ch: "meta" for ch in present_meta} if len(present_meta) > 1 else {}

    # Apply merges to compute post-merge spend
    merged_df = df.copy()
    if merge_map:
        merged_df["channel"] = merged_df["channel"].replace(merge_map)

    # Spend stats per channel
    spend_by_ch = merged_df.groupby("channel")["spend"].sum()
    days_by_ch = merged_df.groupby("channel")["date"].nunique()

    if total_spend > 0:
        spend_pct = spend_by_ch / total_spend * 100
    else:
        spend_pct = spend_by_ch * 0

    # Drop channels below threshold
    keep = []
    dropped = []
    reasons = {}
    for ch in sorted(spend_pct.index):
        pct = spend_pct[ch]
        if pct < DEFAULT_MIN_SPEND_PCT:
            dropped.append(ch)
            reasons[ch] = f"{pct:.1f}% of total spend"
        else:
            keep.append(ch)

    # Build channel detail list (pre-filter, all channels)
    channels_detail = []
    for ch in sorted(spend_pct.index):
        channels_detail.append({
            "name": ch,
            "total_spend": round(float(spend_by_ch[ch]), 2),
            "spend_pct": round(float(spend_pct[ch]), 1),
            "active_days": int(days_by_ch[ch]),
            "total_days": total_days,
        })

    return {
        "channels": keep,
        "merge": merge_map,
        "dropped": dropped,
        "reasons": reasons,
        "channels_detail": channels_detail,
    }


def prepare_from_db(
    db: Session,
    upload_id: int,
    target: str = "revenue",
    channel_config: dict | None = None,
) -> PreparedData:
    """End-to-end: query DB records and prepare model data."""
    df = query_records(db, upload_id)
    return prepare_model_data(df, target=target, channel_config=channel_config)
