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
) -> PreparedData:
    """Transform raw records into model-ready X and y.

    Parameters
    ----------
    df : DataFrame with columns: date, channel, spend, in_platform_conversions,
         revenue, orders, sessions_organic, sessions_direct, sessions_email,
         sessions_referral
    target : "revenue" or "orders" — the outcome variable

    Returns
    -------
    PreparedData with X (features), y (target), and metadata
    """
    if target not in ("revenue", "orders"):
        raise ValueError(f"target must be 'revenue' or 'orders', got '{target}'")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

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


def prepare_from_db(
    db: Session,
    upload_id: int,
    target: str = "revenue",
) -> PreparedData:
    """End-to-end: query DB records and prepare model data."""
    df = query_records(db, upload_id)
    return prepare_model_data(df, target=target)
