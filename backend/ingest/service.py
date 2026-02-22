from dataclasses import asdict
from sqlalchemy.orm import Session
from backend.db.models import Upload, DailyRecord, QualityReport
from backend.ingest.csv_reader import read_csv, ValidationError
from backend.ingest.quality import generate_quality_report, QualityResult


class IngestResult:
    def __init__(
        self,
        upload_id: int,
        status: str,
        rows_stored: int,
        quality: QualityResult,
        warnings: list[str],
    ):
        self.upload_id = upload_id
        self.status = status
        self.rows_stored = rows_stored
        self.quality = quality
        self.warnings = warnings


def ingest_csv(db: Session, filename: str, file_bytes: bytes) -> IngestResult:
    """Full ingestion pipeline: parse CSV, validate, assess quality, store.

    Returns IngestResult with upload ID, quality report, and any warnings.
    Raises ValidationError if the CSV is malformed or fails schema checks.
    """
    df, channel_warnings = read_csv(file_bytes)

    quality = generate_quality_report(df)
    all_warnings = channel_warnings + quality.warnings

    if quality.history_status == "rejected":
        status = "rejected"
    elif quality.history_status == "caution" or channel_warnings:
        status = "partial"
    else:
        status = "success"

    upload = Upload(
        filename=filename,
        row_count=len(df),
        status=status,
    )
    db.add(upload)
    db.flush()  # get upload.id

    records = []
    for _, row in df.iterrows():
        records.append(DailyRecord(
            upload_id=upload.id,
            date=row["date"],
            channel=row["channel"],
            campaign=row["campaign"],
            spend=row["spend"],
            impressions=int(row["impressions"]),
            clicks=int(row["clicks"]),
            in_platform_conversions=row["in_platform_conversions"],
            revenue=row["revenue"],
            orders=int(row["orders"]),
            sessions_organic=int(row["sessions_organic"]),
            sessions_direct=int(row["sessions_direct"]),
            sessions_email=int(row["sessions_email"]),
            sessions_referral=int(row["sessions_referral"]),
        ))
    db.bulk_save_objects(records)

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

    return IngestResult(
        upload_id=upload.id,
        status=status,
        rows_stored=len(df),
        quality=quality,
        warnings=all_warnings,
    )
