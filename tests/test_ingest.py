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
from backend.ingest.wide_reader import detect_format, wide_to_long, read_excel_sheets


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


# ---- wide-format helpers ----

def _make_wide_row(date_str: str, **overrides) -> dict:
    """Generate one wide-format row (one row per date, all channels as columns)."""
    base = {
        "date": date_str,
        "ga_search_cost": 100.0,
        "ga_search_clicks": 200,
        "ga_search_conv": 10.0,
        "ga_shopping_cost": 80.0,
        "ga_shopping_clicks": 150,
        "ga_shopping_conv": 6.0,
        "ga_pmax_cost": 110.0,
        "ga_pmax_clicks": 190,
        "ga_pmax_conv": 14.0,
        "ga_youtube_cost": 50.0,
        "ga_youtube_clicks": 80,
        "ga_youtube_conv": 2.0,
        "ga_dgen_cost": 40.0,
        "ga_dgen_clicks": 60,
        "ga_dgen_conv": 3.0,
        "meta_fb_cost": 120.0,
        "meta_fb_clicks": 220,
        "meta_fb_conv": 9.0,
        "meta_insta_cost": 75.0,
        "meta_insta_clicks": 160,
        "meta_insta_conv": 7.0,
        "meta_stories_cost": 45.0,
        "meta_stories_clicks": 95,
        "meta_stories_conv": 4.0,
        "shopify_revenue": 2500.0,
        "shopify_orders": 30,
        "ga4_sessions_organic": 300,
        "ga4_sessions_direct": 180,
        "ga4_sessions_email": 40,
        "ga4_sessions_referral": 25,
    }
    base.update(overrides)
    return base


def _wide_date_range(start: str, days: int) -> list[dict]:
    """Generate wide-format rows for a date range."""
    import random
    base_date = datetime.date.fromisoformat(start)
    rows = []
    for i in range(days):
        d = base_date + datetime.timedelta(days=i)
        rows.append(_make_wide_row(
            d.isoformat(),
            ga_search_cost=round(random.uniform(50, 200), 2),
            shopify_revenue=round(random.uniform(1500, 3500), 2),
        ))
    return rows


# ---- wide-format tests ----

class TestWideFormat:
    def test_detect_long_format(self):
        rows = [_make_row("2024-01-01")]
        df = pd.DataFrame(rows)
        assert detect_format(df) == "long"

    def test_detect_wide_format(self):
        rows = [_make_wide_row("2024-01-01")]
        df = pd.DataFrame(rows)
        assert detect_format(df) == "wide"

    def test_detect_unknown_format_raises(self):
        df = pd.DataFrame({"date": ["2024-01-01"], "foo": [1], "bar": [2]})
        with pytest.raises(ValidationError, match="Unrecognised CSV format"):
            detect_format(df)

    def test_wide_to_long_shape(self):
        rows = [_make_wide_row("2024-01-01"), _make_wide_row("2024-01-02")]
        df = pd.DataFrame(rows)
        long_df, _ = wide_to_long(df)
        # 2 dates × 8 channels = 16 rows
        assert len(long_df) == 16

    def test_wide_to_long_columns(self):
        rows = [_make_wide_row("2024-01-01")]
        df = pd.DataFrame(rows)
        long_df, _ = wide_to_long(df)
        from backend.ingest.schema import REQUIRED_COLUMNS
        for col in REQUIRED_COLUMNS:
            assert col in long_df.columns, f"Missing column: {col}"

    def test_wide_channel_mapping(self):
        rows = [_make_wide_row("2024-01-01")]
        df = pd.DataFrame(rows)
        long_df, _ = wide_to_long(df)
        channels = set(long_df["channel"].unique())
        expected = {
            "google_search", "google_shopping", "google_pmax", "google_youtube",
            "google_demand_gen", "meta_feed", "meta_instagram", "meta_stories",
        }
        assert channels == expected

    def test_wide_metric_mapping(self):
        rows = [_make_wide_row("2024-01-01", ga_search_cost=123.45,
                               ga_search_clicks=456, ga_search_conv=7.8)]
        df = pd.DataFrame(rows)
        long_df, _ = wide_to_long(df)
        search_row = long_df[long_df["channel"] == "google_search"].iloc[0]
        assert search_row["spend"] == 123.45
        assert search_row["clicks"] == 456
        assert search_row["in_platform_conversions"] == 7.8

    def test_wide_site_wide_columns(self):
        rows = [_make_wide_row("2024-01-01", shopify_revenue=2500.0, shopify_orders=30,
                               ga4_sessions_organic=300, ga4_sessions_direct=180)]
        df = pd.DataFrame(rows)
        long_df, _ = wide_to_long(df)
        # Every channel row should have the same site-wide values
        for _, row in long_df.iterrows():
            assert row["revenue"] == 2500.0
            assert row["orders"] == 30
            assert row["sessions_organic"] == 300
            assert row["sessions_direct"] == 180

    def test_wide_unknown_prefix_warns(self):
        rows = [_make_wide_row("2024-01-01")]
        df = pd.DataFrame(rows)
        df["tiktok_cost"] = 50.0
        long_df, warnings = wide_to_long(df)
        assert any("tiktok" in w for w in warnings)
        # Should still produce valid output
        assert len(long_df) == 8  # 8 channels

    def test_wide_full_pipeline(self, db_session):
        """Wide CSV bytes → ingest_csv() → DB records stored correctly."""
        rows = _wide_date_range("2023-01-01", 600)
        csv_bytes = _make_csv(rows)
        result = ingest_csv(db_session, "wide_test.csv", csv_bytes)
        # 600 dates × 8 channels = 4800 rows
        assert result.rows_stored == 4800
        assert result.upload_id is not None
        assert db_session.query(DailyRecord).count() == 4800
        # Verify all 8 channels present
        channels = {r.channel for r in db_session.query(DailyRecord.channel).distinct()}
        assert len(channels) == 8


# ---- Excel multi-sheet helpers ----

def _make_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Create an in-memory .xlsx file from a dict of sheet_name → DataFrame."""
    from io import BytesIO
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


def _make_channel_row(date_str: str, channel: str, **overrides) -> dict:
    """One long-format row for a channel sheet (Google Ads or Meta)."""
    base = {
        "date": date_str,
        "channel": channel,
        "campaign": f"{channel}_campaign",
        "spend": 100.0,
        "clicks": 200,
        "in_platform_conversions": 10.0,
    }
    base.update(overrides)
    return base


def _excel_date_range_sheets(start: str, days: int) -> dict[str, pd.DataFrame]:
    """Build 4 long-format sheets (Google Ads, Meta, Shopify, GA4) for a date range."""
    import random
    base_date = datetime.date.fromisoformat(start)
    dates = [(base_date + datetime.timedelta(days=i)).isoformat() for i in range(days)]

    ga_channels = ["google_search", "google_shopping", "google_pmax",
                    "google_youtube", "google_demand_gen"]
    meta_channels = ["meta_feed", "meta_instagram", "meta_stories"]

    ga_rows = []
    meta_rows = []
    shop_rows = []
    ga4_rows = []
    for d in dates:
        for ch in ga_channels:
            ga_rows.append(_make_channel_row(
                d, ch, spend=round(random.uniform(50, 200), 2),
            ))
        for ch in meta_channels:
            meta_rows.append(_make_channel_row(
                d, ch, spend=round(random.uniform(30, 150), 2),
            ))
        shop_rows.append({
            "date": d,
            "revenue": round(random.uniform(1500, 3500), 2),
            "orders": 30,
        })
        for source, count in [("Organic Search", 300), ("Direct", 180),
                                ("Email", 40), ("Referral", 25)]:
            ga4_rows.append({"date": d, "source": source, "sessions": count})

    return {
        "Google Ads": pd.DataFrame(ga_rows),
        "Meta": pd.DataFrame(meta_rows),
        "Shopify": pd.DataFrame(shop_rows),
        "GA4": pd.DataFrame(ga4_rows),
    }


# ---- Excel multi-sheet tests ----

class TestExcelMultiSheet:
    def test_read_excel_stacks_channel_sheets(self):
        sheets = _excel_date_range_sheets("2024-01-01", 2)
        xlsx_bytes = _make_excel_bytes(sheets)
        combined, warnings = read_excel_sheets(xlsx_bytes)
        # 2 dates × (5 GA + 3 Meta) channels = 16 rows
        assert len(combined) == 16
        assert "channel" in combined.columns
        assert "spend" in combined.columns

    def test_read_excel_merges_site_wide(self):
        sheets = _excel_date_range_sheets("2024-01-01", 2)
        xlsx_bytes = _make_excel_bytes(sheets)
        combined, _ = read_excel_sheets(xlsx_bytes)
        # Site-wide columns from Shopify and GA4 merged by date
        assert "revenue" in combined.columns
        assert "orders" in combined.columns
        assert "sessions_organic" in combined.columns
        assert "sessions_referral" in combined.columns

    def test_read_excel_skips_sheet_without_date(self):
        sheets = {
            "Google Ads": pd.DataFrame([
                _make_channel_row("2024-01-01", "google_search"),
            ]),
            "Notes": pd.DataFrame({"info": ["some notes"]}),
        }
        xlsx_bytes = _make_excel_bytes(sheets)
        combined, warnings = read_excel_sheets(xlsx_bytes)
        assert any("Notes" in w and "no 'date'" in w for w in warnings)
        assert "channel" in combined.columns

    def test_read_excel_detected_as_long(self):
        sheets = _excel_date_range_sheets("2024-01-01", 2)
        xlsx_bytes = _make_excel_bytes(sheets)
        combined, _ = read_excel_sheets(xlsx_bytes)
        fmt = detect_format(combined)
        assert fmt == "long"

    def test_ga4_pivot_from_long_format(self):
        """GA4 sheet with date/source/sessions gets pivoted into session columns."""
        ga4_df = pd.DataFrame([
            {"date": "2024-01-01", "source": "Organic Search", "sessions": 300},
            {"date": "2024-01-01", "source": "Direct", "sessions": 180},
            {"date": "2024-01-01", "source": "Email", "sessions": 40},
            {"date": "2024-01-02", "source": "Organic Search", "sessions": 345},
            # Jan 2 has no Direct/Email/Referral — should default to 0
        ])
        sheets = {
            "Google Ads": pd.DataFrame([
                _make_channel_row("2024-01-01", "google_search"),
                _make_channel_row("2024-01-02", "google_search"),
            ]),
            "GA4": ga4_df,
        }
        xlsx_bytes = _make_excel_bytes(sheets)
        combined, _ = read_excel_sheets(xlsx_bytes)
        assert "sessions_organic" in combined.columns
        assert "sessions_direct" in combined.columns
        assert "sessions_email" in combined.columns
        assert "sessions_referral" in combined.columns
        # Jan 1 row should have 300 organic
        jan1 = combined[combined["date"] == "2024-01-01"].iloc[0]
        assert jan1["sessions_organic"] == 300
        assert jan1["sessions_direct"] == 180
        # Jan 2 missing sources should be 0
        jan2 = combined[combined["date"] == "2024-01-02"].iloc[0]
        assert jan2["sessions_direct"] == 0
        assert jan2["sessions_referral"] == 0

    def test_ga4_unknown_source_warns(self):
        """Unknown GA4 source values produce a warning."""
        ga4_df = pd.DataFrame([
            {"date": "2024-01-01", "source": "Organic Search", "sessions": 300},
            {"date": "2024-01-01", "source": "Social", "sessions": 50},
        ])
        sheets = {
            "Google Ads": pd.DataFrame([
                _make_channel_row("2024-01-01", "google_search"),
            ]),
            "GA4": ga4_df,
        }
        xlsx_bytes = _make_excel_bytes(sheets)
        combined, warnings = read_excel_sheets(xlsx_bytes)
        assert any("social" in w.lower() for w in warnings)

    def test_ga4_compact_date_format(self):
        """GA4 dates like 20231215 are normalised to YYYY-MM-DD."""
        ga4_df = pd.DataFrame([
            {"date": "20240101", "source": "Organic Search", "sessions": 300},
            {"date": "20240102", "source": "Direct", "sessions": 180},
        ])
        sheets = {
            "Google Ads": pd.DataFrame([
                _make_channel_row("2024-01-01", "google_search"),
                _make_channel_row("2024-01-02", "google_search"),
            ]),
            "GA4": ga4_df,
        }
        xlsx_bytes = _make_excel_bytes(sheets)
        combined, _ = read_excel_sheets(xlsx_bytes)
        # GA4 dates should merge correctly with channel dates
        jan1 = combined[combined["date"] == "2024-01-01"].iloc[0]
        assert jan1["sessions_organic"] == 300

    def test_excel_full_pipeline(self, db_session):
        """Multi-sheet .xlsx → ingest_csv() → DB records stored correctly."""
        sheets = _excel_date_range_sheets("2023-01-01", 600)
        xlsx_bytes = _make_excel_bytes(sheets)
        result = ingest_csv(db_session, "multi_sheet.xlsx", xlsx_bytes)
        # 600 dates × 8 channels = 4800 rows
        assert result.rows_stored == 4800
        assert result.upload_id is not None
        assert db_session.query(DailyRecord).count() == 4800
        channels = {r.channel for r in db_session.query(DailyRecord.channel).distinct()}
        assert len(channels) == 8
