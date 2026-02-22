import io
import datetime
import pytest
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.db.config import Base
from backend.db.models import Upload, DailyRecord, QualityReport
from backend.ingest.csv_reader import read_csv, standardise_channel, ValidationError
from backend.ingest.quality import generate_quality_report
from backend.ingest.service import ingest_csv


# ---- fixtures ----

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _make_csv(rows: list[dict]) -> bytes:
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode()


def _make_row(date_str: str, channel: str = "google_search", **overrides) -> dict:
    base = {
        "date": date_str,
        "channel": channel,
        "campaign": "test_campaign",
        "spend": 100.0,
        "impressions": 1000,
        "clicks": 50,
        "in_platform_conversions": 5.0,
        "revenue": 500.0,
        "orders": 10,
        "sessions_organic": 200,
        "sessions_direct": 100,
        "sessions_email": 50,
        "sessions_referral": 30,
    }
    base.update(overrides)
    return base


def _date_range_rows(start: str, days: int, channel: str = "google_search") -> list[dict]:
    """Generate one row per day for a channel."""
    import random
    base_date = datetime.date.fromisoformat(start)
    rows = []
    for i in range(days):
        d = base_date + datetime.timedelta(days=i)
        rows.append(_make_row(
            d.isoformat(),
            channel=channel,
            spend=round(random.uniform(50, 200), 2),
            revenue=round(random.uniform(300, 700), 2),
        ))
    return rows


# ---- channel standardisation tests ----

class TestChannelStandardisation:
    def test_canonical_names_unchanged(self):
        assert standardise_channel("google_search") == "google_search"
        assert standardise_channel("meta_feed") == "meta_feed"

    def test_aliases_mapped(self):
        assert standardise_channel("facebook") == "meta_feed"
        assert standardise_channel("instagram") == "meta_instagram"
        assert standardise_channel("youtube") == "google_youtube"
        assert standardise_channel("performance_max") == "google_pmax"

    def test_whitespace_and_case(self):
        assert standardise_channel("  Google Search  ") == "google_search"
        assert standardise_channel("META_FEED") == "meta_feed"

    def test_unknown_channel_passed_through(self):
        assert standardise_channel("tiktok_ads") == "tiktok_ads"


# ---- CSV reader tests ----

class TestCsvReader:
    def test_valid_csv(self):
        rows = [_make_row("2024-01-01"), _make_row("2024-01-02")]
        df, warnings = read_csv(_make_csv(rows))
        assert len(df) == 2
        cols = set(df.columns)
        assert {"date", "channel", "spend", "revenue"}.issubset(cols)

    def test_missing_columns_raises(self):
        csv_bytes = b"date,channel,campaign\n2024-01-01,google_search,test\n"
        with pytest.raises(ValidationError, match="Missing required columns"):
            read_csv(csv_bytes)

    def test_negative_spend_raises(self):
        rows = [_make_row("2024-01-01", spend=-100)]
        with pytest.raises(ValidationError, match="Schema validation failed"):
            read_csv(_make_csv(rows))

    def test_channel_standardisation_during_read(self):
        rows = [_make_row("2024-01-01", channel="facebook")]
        df, _ = read_csv(_make_csv(rows))
        assert df["channel"].iloc[0] == "meta_feed"

    def test_unknown_channel_warning(self):
        rows = [_make_row("2024-01-01", channel="tiktok_ads")]
        df, warnings = read_csv(_make_csv(rows))
        assert any("tiktok_ads" in w for w in warnings)


# ---- quality report tests ----

class TestQualityReport:
    def test_sufficient_history(self):
        rows = _date_range_rows("2023-01-01", 600)
        df, _ = read_csv(_make_csv(rows))
        report = generate_quality_report(df)
        assert report.history_status == "sufficient"
        assert report.history_months >= 18

    def test_caution_history(self):
        rows = _date_range_rows("2024-01-01", 300)
        df, _ = read_csv(_make_csv(rows))
        report = generate_quality_report(df)
        assert report.history_status == "caution"

    def test_rejected_history(self):
        rows = _date_range_rows("2024-06-01", 100)
        df, _ = read_csv(_make_csv(rows))
        report = generate_quality_report(df)
        assert report.history_status == "rejected"
        assert any("9 months" in w for w in report.warnings)

    def test_gap_detection(self):
        # Create rows with a 5-day gap in the middle
        rows = _date_range_rows("2023-01-01", 30)
        # Remove days 10-14
        rows = [r for i, r in enumerate(rows) if not (10 <= i <= 14)]
        df, _ = read_csv(_make_csv(rows))
        report = generate_quality_report(df)
        assert report.gap_count >= 5

    def test_low_variance_detection(self):
        # Constant spend = zero variance
        rows = [
            _make_row(
                (datetime.date(2023, 1, 1) + datetime.timedelta(days=i)).isoformat(),
                spend=100.0,
                revenue=500.0,
            )
            for i in range(600)
        ]
        df, _ = read_csv(_make_csv(rows))
        report = generate_quality_report(df)
        assert "google_search" in report.low_variance_channels


# ---- full pipeline (service) tests ----

class TestIngestService:
    def test_successful_ingest(self, db_session):
        rows = _date_range_rows("2023-01-01", 600)
        result = ingest_csv(db_session, "test.csv", _make_csv(rows))
        assert result.status == "success"
        assert result.rows_stored == 600
        assert result.upload_id is not None

        # Verify DB records
        assert db_session.query(Upload).count() == 1
        assert db_session.query(DailyRecord).count() == 600
        assert db_session.query(QualityReport).count() == 1

    def test_rejected_ingest(self, db_session):
        rows = _date_range_rows("2024-09-01", 60)
        result = ingest_csv(db_session, "short.csv", _make_csv(rows))
        assert result.status == "rejected"
        assert any("9 months" in w for w in result.warnings)

    def test_caution_ingest(self, db_session):
        rows = _date_range_rows("2024-01-01", 300)
        result = ingest_csv(db_session, "medium.csv", _make_csv(rows))
        assert result.status == "partial"

    def test_validation_error_prevents_storage(self, db_session):
        csv_bytes = b"date,channel,campaign\n2024-01-01,google_search,test\n"
        with pytest.raises(ValidationError):
            ingest_csv(db_session, "bad.csv", csv_bytes)
        assert db_session.query(Upload).count() == 0
