"""Integration orchestration: sync connections and merge multiple data sources."""

from __future__ import annotations

import datetime
import json

import pandas as pd
from sqlalchemy.orm import Session

from backend.db.models import (
    ApiConnection,
    ApiSync,
    DailyRecord,
    QualityReport,
    Upload,
)
from backend.ingest.quality import generate_quality_report
from backend.ingest.schema import REQUIRED_COLUMNS, ingestion_schema
from backend.integrations.base import ConnectorConfig, FetchResult
from backend.integrations.registry import get_connector


def sync_connection(
    db: Session,
    connection_id: int,
    start_date: datetime.date,
    end_date: datetime.date,
) -> ApiSync:
    """Fetch data from an API connection and store as an Upload.

    1. Load connection from DB
    2. Instantiate connector and fetch data
    3. Create Upload + DailyRecords
    4. Record ApiSync entry

    Returns the ApiSync record.
    """
    conn = db.query(ApiConnection).filter_by(id=connection_id).first()
    if not conn or not conn.is_active:
        raise ValueError(f"Connection {connection_id} not found or inactive")

    config = ConnectorConfig(
        platform=conn.platform,
        credentials=json.loads(conn.credentials_json),
        config=json.loads(conn.config_json),
    )

    # Create sync record
    sync = ApiSync(
        connection_id=connection_id,
        upload_id=0,  # placeholder, updated after upload creation
        date_range_start=start_date,
        date_range_end=end_date,
        status="running",
    )
    db.add(sync)
    db.flush()

    try:
        connector = get_connector(config)
        result: FetchResult = connector.fetch(start_date, end_date)

        # Create Upload
        upload = Upload(
            filename=f"api_sync_{conn.platform}_{start_date}_{end_date}",
            row_count=result.rows_fetched,
            status="success",
        )
        db.add(upload)
        db.flush()

        # Store DailyRecords
        _store_records(db, upload.id, result.data)

        # Update sync record
        sync.upload_id = upload.id
        sync.rows_fetched = result.rows_fetched
        sync.status = "completed"
        sync.completed_at = datetime.datetime.now(datetime.UTC)

        # Update connection status
        conn.last_sync_status = "completed"

        db.commit()
        return sync

    except Exception as exc:
        sync.status = "failed"
        sync.error_message = str(exc)
        sync.completed_at = datetime.datetime.now(datetime.UTC)
        conn.last_sync_status = "failed"
        db.commit()
        raise


def merge_sources(
    db: Session,
    upload_ids: list[int],
) -> Upload:
    """Merge multiple source uploads into a single unified dataset.

    Merge logic:
    1. Ad data rows (google_*, meta_*) provide the skeleton with spend/clicks/impressions
    2. GA4 session data joined by date (site-level) — takes max per date
    3. Revenue/orders from ecommerce sources joined by date — prefers ecommerce source over zeros

    Returns the new merged Upload.
    """
    if len(upload_ids) < 2:
        raise ValueError("At least 2 uploads required for merge")

    # Load all DailyRecords grouped by source type
    ad_rows = []
    session_rows = []
    ecommerce_rows = []

    for uid in upload_ids:
        records = db.query(DailyRecord).filter_by(upload_id=uid).all()
        if not records:
            raise ValueError(f"Upload {uid} has no records")

        for r in records:
            row = {
                "date": r.date,
                "channel": r.channel,
                "campaign": r.campaign,
                "spend": r.spend,
                "impressions": r.impressions,
                "clicks": r.clicks,
                "in_platform_conversions": r.in_platform_conversions,
                "revenue": r.revenue,
                "orders": r.orders,
                "sessions_organic": r.sessions_organic,
                "sessions_direct": r.sessions_direct,
                "sessions_email": r.sessions_email,
                "sessions_referral": r.sessions_referral,
            }
            if r.channel == "organic_traffic":
                session_rows.append(row)
            elif r.channel == "ecommerce":
                ecommerce_rows.append(row)
            else:
                ad_rows.append(row)

    # Start with ad data as the skeleton
    if ad_rows:
        merged_df = pd.DataFrame(ad_rows)
    else:
        merged_df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Join GA4 sessions by date (site-level, not channel-level)
    if session_rows:
        sessions_df = pd.DataFrame(session_rows)
        session_cols = ["sessions_organic", "sessions_direct", "sessions_email", "sessions_referral"]
        # Take max sessions per date across all GA4 rows
        sessions_agg = sessions_df.groupby("date")[session_cols].max().reset_index()

        if not merged_df.empty:
            # Drop existing zero session columns and merge GA4 data
            merged_df = merged_df.drop(columns=session_cols, errors="ignore")
            merged_df = merged_df.merge(sessions_agg, on="date", how="left")
            for col in session_cols:
                merged_df[col] = merged_df[col].fillna(0).astype(int)
        else:
            # Only session data — use it as base
            merged_df = sessions_df

    # Join ecommerce revenue/orders by date
    if ecommerce_rows:
        ecom_df = pd.DataFrame(ecommerce_rows)
        ecom_agg = ecom_df.groupby("date")[["revenue", "orders"]].sum().reset_index()

        if not merged_df.empty:
            # Replace zero revenue/orders with ecommerce data
            merged_df = merged_df.drop(columns=["revenue", "orders"], errors="ignore")
            merged_df = merged_df.merge(ecom_agg, on="date", how="left")
            merged_df["revenue"] = merged_df["revenue"].fillna(0.0)
            merged_df["orders"] = merged_df["orders"].fillna(0).astype(int)
        else:
            merged_df = ecom_df

    if merged_df.empty:
        raise ValueError("No data to merge — all uploads empty")

    # Ensure full schema
    for col in REQUIRED_COLUMNS:
        if col not in merged_df.columns:
            merged_df[col] = 0
    merged_df = merged_df[REQUIRED_COLUMNS]

    # Ensure correct types before validation
    merged_df["date"] = pd.to_datetime(merged_df["date"])
    for col in ["impressions", "clicks", "orders", "sessions_organic",
                 "sessions_direct", "sessions_email", "sessions_referral"]:
        merged_df[col] = merged_df[col].astype(int)
    for col in ["spend", "in_platform_conversions", "revenue"]:
        merged_df[col] = merged_df[col].astype(float)

    # Create merged Upload
    upload = Upload(
        filename=f"merged_{'_'.join(str(uid) for uid in upload_ids)}",
        row_count=len(merged_df),
        status="success",
    )
    db.add(upload)
    db.flush()

    # Store merged records
    _store_records(db, upload.id, merged_df)

    # Generate and store quality report
    quality = generate_quality_report(merged_df)
    from dataclasses import asdict
    qr = QualityReport(
        upload_id=upload.id,
        history_days=quality.history_days,
        history_status=quality.history_status,
        gap_count=quality.gap_count,
        spike_count=quality.spike_count,
        channels_found=str([cq.channel for cq in quality.channels]),
        low_variance_channels=str(quality.low_variance_channels),
        report_json=quality.to_json(),
    )
    db.add(qr)
    db.commit()

    return upload


def _store_records(db: Session, upload_id: int, df: pd.DataFrame) -> None:
    """Store DataFrame rows as DailyRecord entries."""
    records = []
    for _, row in df.iterrows():
        records.append(DailyRecord(
            upload_id=upload_id,
            date=pd.Timestamp(row["date"]).date() if not isinstance(row["date"], datetime.date) else row["date"],
            channel=str(row["channel"]),
            campaign=str(row["campaign"]),
            spend=float(row["spend"]),
            impressions=int(row["impressions"]),
            clicks=int(row["clicks"]),
            in_platform_conversions=float(row["in_platform_conversions"]),
            revenue=float(row["revenue"]),
            orders=int(row["orders"]),
            sessions_organic=int(row["sessions_organic"]),
            sessions_direct=int(row["sessions_direct"]),
            sessions_email=int(row["sessions_email"]),
            sessions_referral=int(row["sessions_referral"]),
        ))
    db.bulk_save_objects(records)
