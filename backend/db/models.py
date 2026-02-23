import datetime
from sqlalchemy import (
    String, Float, Integer, Date, DateTime, Text, ForeignKey, UniqueConstraint,
    Boolean,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.db.config import Base


class Upload(Base):
    """Tracks each CSV file upload."""
    __tablename__ = "uploads"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    uploaded_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC)
    )
    row_count: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(50))  # "success" | "partial" | "rejected"

    rows: Mapped[list["DailyRecord"]] = relationship(back_populates="upload")
    quality_report: Mapped["QualityReport | None"] = relationship(back_populates="upload")


class DailyRecord(Base):
    """One row of marketing data: a single date × channel × campaign."""
    __tablename__ = "daily_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploads.id"))
    date: Mapped[datetime.date] = mapped_column(Date)
    channel: Mapped[str] = mapped_column(String(100))
    campaign: Mapped[str] = mapped_column(String(255))
    spend: Mapped[float] = mapped_column(Float)
    clicks: Mapped[int] = mapped_column(Integer)
    in_platform_conversions: Mapped[float] = mapped_column(Float)
    revenue: Mapped[float] = mapped_column(Float)
    orders: Mapped[int] = mapped_column(Integer)
    sessions_organic: Mapped[int] = mapped_column(Integer)
    sessions_direct: Mapped[int] = mapped_column(Integer)
    sessions_email: Mapped[int] = mapped_column(Integer)
    sessions_referral: Mapped[int] = mapped_column(Integer)

    upload: Mapped["Upload"] = relationship(back_populates="rows")

    __table_args__ = (
        UniqueConstraint("upload_id", "date", "channel", "campaign", name="uq_upload_date_channel_campaign"),
    )


class QualityReport(Base):
    """Stores the data quality assessment for an upload."""
    __tablename__ = "quality_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploads.id"), unique=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC)
    )
    history_days: Mapped[int] = mapped_column(Integer)
    history_status: Mapped[str] = mapped_column(String(50))  # "sufficient" | "caution" | "rejected"
    gap_count: Mapped[int] = mapped_column(Integer)
    spike_count: Mapped[int] = mapped_column(Integer)
    channels_found: Mapped[str] = mapped_column(Text)  # JSON list
    low_variance_channels: Mapped[str] = mapped_column(Text)  # JSON list
    report_json: Mapped[str] = mapped_column(Text)  # full report as JSON

    upload: Mapped["Upload"] = relationship(back_populates="quality_report")


class ModelRun(Base):
    """Tracks a Bayesian MMM model fitting run."""
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploads.id"))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC)
    )
    status: Mapped[str] = mapped_column(String(50))  # "running" | "completed" | "failed"
    target_variable: Mapped[str] = mapped_column(String(50))  # "revenue" | "orders"
    channels_used: Mapped[str] = mapped_column(Text)  # JSON list of channel names
    config_json: Mapped[str] = mapped_column(Text)  # model config (priors, adstock, saturation)
    results_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # posteriors, contributions
    diagnostics_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # MAPE, R², divergences
    idata_path: Mapped[str | None] = mapped_column(String(500), nullable=True)  # path to saved InferenceData
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    validation_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # ValidationReport JSON


class ExperimentResult(Base):
    """Stores a spend-scaling or product experiment result."""
    __tablename__ = "experiment_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploads.id"))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC)
    )
    experiment_type: Mapped[str] = mapped_column(String(50))  # "spend_scaling" | "product"
    channel: Mapped[str | None] = mapped_column(String(100), nullable=True)
    result_json: Mapped[str] = mapped_column(Text)  # full result as JSON


class ApiConnection(Base):
    """Stores a registered API connection (credentials + config)."""
    __tablename__ = "api_connections"

    id: Mapped[int] = mapped_column(primary_key=True)
    platform: Mapped[str] = mapped_column(String(50))  # google_ads | meta | ga4 | shopify | woocommerce
    display_name: Mapped[str] = mapped_column(String(255))
    credentials_json: Mapped[str] = mapped_column(Text)  # encrypted/stored credentials
    config_json: Mapped[str] = mapped_column(Text)  # account IDs, property IDs, etc.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC)
    )
    last_sync_status: Mapped[str | None] = mapped_column(String(50), nullable=True)

    syncs: Mapped[list["ApiSync"]] = relationship(back_populates="connection")


class ApiSync(Base):
    """Tracks each data sync run for an API connection."""
    __tablename__ = "api_syncs"

    id: Mapped[int] = mapped_column(primary_key=True)
    connection_id: Mapped[int] = mapped_column(ForeignKey("api_connections.id"))
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploads.id"))
    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC)
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(DateTime, nullable=True)
    date_range_start: Mapped[datetime.date] = mapped_column(Date)
    date_range_end: Mapped[datetime.date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(50))  # "running" | "completed" | "failed"
    rows_fetched: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    connection: Mapped["ApiConnection"] = relationship(back_populates="syncs")
    upload: Mapped["Upload"] = relationship()
