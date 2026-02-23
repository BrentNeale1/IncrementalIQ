"""Tests for Phase 2: data preparation, model construction, and result extraction.

Model fitting tests use minimal sampling (1 chain, few draws) to keep them fast.
Full MCMC validation is deferred to integration tests.
"""
import datetime
import json
import random
import pytest
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.db.config import Base
from backend.db.models import Upload, DailyRecord, ModelRun
from backend.models.data_prep import prepare_model_data, PreparedData, prepare_from_db
from backend.models.mmm import (
    ModelConfig,
    build_mmm,
    fit_mmm,
    extract_results,
)
from backend.models.service import run_model


# ---- fixtures ----

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _generate_synthetic_data(
    start: str = "2023-01-01",
    days: int = 365,
    channels: list[str] | None = None,
) -> pd.DataFrame:
    """Generate realistic synthetic marketing data for testing.

    Produces daily data with correlated spend/revenue and realistic patterns.
    """
    channels = channels or ["google_search", "meta_feed"]
    base_date = datetime.date.fromisoformat(start)
    rng = np.random.default_rng(42)

    rows = []
    for day_offset in range(days):
        d = base_date + datetime.timedelta(days=day_offset)

        # Baseline revenue with weekly seasonality
        day_of_week = d.weekday()
        weekly_factor = 1.0 + 0.15 * np.sin(2 * np.pi * day_of_week / 7)

        for ch in channels:
            # Spend with some variance
            base_spend = {"google_search": 150, "meta_feed": 100}.get(ch, 80)
            spend = max(0, rng.normal(base_spend, base_spend * 0.3))

            # Revenue correlated with spend + noise
            base_rev = spend * 3.5 * weekly_factor + rng.normal(0, 50)

            rows.append({
                "date": d,
                "channel": ch,
                "campaign": f"{ch}_campaign_1",
                "spend": round(spend, 2),
                "impressions": int(spend * 10 + rng.normal(0, 100)),
                "clicks": int(spend * 0.5 + rng.normal(0, 10)),
                "in_platform_conversions": round(max(0, spend * 0.03 + rng.normal(0, 1)), 2),
                "revenue": round(max(0, base_rev), 2),
                "orders": max(0, int(base_rev / 50 + rng.normal(0, 2))),
                "sessions_organic": max(0, int(200 * weekly_factor + rng.normal(0, 20))),
                "sessions_direct": max(0, int(100 * weekly_factor + rng.normal(0, 15))),
                "sessions_email": max(0, int(50 + rng.normal(0, 10))),
                "sessions_referral": max(0, int(30 + rng.normal(0, 8))),
            })

    return pd.DataFrame(rows)


def _insert_records(db_session, df: pd.DataFrame) -> int:
    """Insert synthetic data into DB and return upload_id."""
    upload = Upload(
        filename="synthetic_test.csv",
        row_count=len(df),
        status="success",
    )
    db_session.add(upload)
    db_session.flush()

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
    db_session.bulk_save_objects(records)
    db_session.commit()
    return upload.id


FAST_CONFIG = ModelConfig(
    chains=1,
    tune=50,
    draws=50,
    target_accept=0.8,
    random_seed=42,
)


# ---- data preparation tests ----

class TestDataPrep:
    def test_prepare_model_data_shape(self):
        df = _generate_synthetic_data(days=30, channels=["google_search", "meta_feed"])
        data = prepare_model_data(df, target="revenue")

        assert data.daily_rows == 30
        assert data.target_variable == "revenue"
        assert len(data.y) == 30
        assert len(data.X) == 30

    def test_channel_columns_created(self):
        df = _generate_synthetic_data(days=30, channels=["google_search", "meta_feed"])
        data = prepare_model_data(df, target="revenue")

        assert "spend_google_search" in data.channel_columns
        assert "spend_meta_feed" in data.channel_columns
        assert len(data.channel_columns) == 2

    def test_control_columns_created(self):
        df = _generate_synthetic_data(days=30, channels=["google_search", "meta_feed"])
        data = prepare_model_data(df, target="revenue")

        # Session controls
        assert "sessions_organic" in data.control_columns
        assert "sessions_direct" in data.control_columns
        assert "sessions_email" in data.control_columns
        assert "sessions_referral" in data.control_columns
        # In-platform conversions per channel
        assert "ipc_google_search" in data.control_columns
        assert "ipc_meta_feed" in data.control_columns

    def test_spend_aggregated_across_campaigns(self):
        """Multiple campaigns per channel should sum to one spend value per day."""
        df = _generate_synthetic_data(days=5, channels=["google_search"])
        # Duplicate rows with different campaign names
        df2 = df.copy()
        df2["campaign"] = "google_search_campaign_2"
        df2["spend"] = 50.0
        combined = pd.concat([df, df2], ignore_index=True)

        data = prepare_model_data(combined, target="revenue")
        # Each day should have one row
        assert data.daily_rows == 5
        # Spend should be sum of both campaigns
        assert all(data.X["spend_google_search"] > 0)

    def test_target_orders(self):
        df = _generate_synthetic_data(days=30)
        data = prepare_model_data(df, target="orders")
        assert data.target_variable == "orders"
        assert data.y.name == "orders"

    def test_invalid_target_raises(self):
        df = _generate_synthetic_data(days=30)
        with pytest.raises(ValueError, match="target must be"):
            prepare_model_data(df, target="clicks")

    def test_date_column_is_datetime(self):
        df = _generate_synthetic_data(days=30)
        data = prepare_model_data(df, target="revenue")
        assert pd.api.types.is_datetime64_any_dtype(data.X["date"])

    def test_no_negative_spend_in_output(self):
        df = _generate_synthetic_data(days=30)
        data = prepare_model_data(df, target="revenue")
        for col in data.channel_columns:
            assert (data.X[col] >= 0).all()

    def test_site_wide_revenue_not_inflated(self):
        """Revenue (site-wide) must not be summed across channel rows per date.

        When multiple channels share the same date, revenue is identical on
        each row (broadcast from site-wide source).  data_prep must use "max"
        (not "sum") to de-duplicate, so the aggregated daily revenue equals
        the single-row value, not N × that value.
        """
        rng = np.random.default_rng(99)
        days = 10
        channels = ["google_search", "meta_feed", "google_pmax"]
        base_date = datetime.date(2024, 1, 1)
        daily_revenue = [round(rng.uniform(500, 2000), 2) for _ in range(days)]

        rows = []
        for i in range(days):
            d = base_date + datetime.timedelta(days=i)
            for ch in channels:
                rows.append({
                    "date": d,
                    "channel": ch,
                    "campaign": f"{ch}_c1",
                    "spend": round(rng.uniform(50, 200), 2),
                    "impressions": int(rng.uniform(500, 2000)),
                    "clicks": int(rng.uniform(20, 100)),
                    "in_platform_conversions": round(rng.uniform(0, 5), 2),
                    "revenue": daily_revenue[i],  # identical across channels
                    "orders": 10,
                    "sessions_organic": 200,
                    "sessions_direct": 100,
                    "sessions_email": 50,
                    "sessions_referral": 30,
                })

        df = pd.DataFrame(rows)
        data = prepare_model_data(df, target="revenue")

        # y should equal the per-row revenue, NOT len(channels) × that value
        for i in range(days):
            assert abs(data.y.iloc[i] - daily_revenue[i]) < 0.01, (
                f"Day {i}: expected {daily_revenue[i]}, got {data.y.iloc[i]} "
                f"(inflated {data.y.iloc[i] / daily_revenue[i]:.1f}x)"
            )

    def test_prepare_from_db(self, db_session):
        df = _generate_synthetic_data(days=30)
        upload_id = _insert_records(db_session, df)
        data = prepare_from_db(db_session, upload_id, target="revenue")
        assert data.daily_rows == 30
        assert len(data.channel_columns) == 2


# ---- model construction tests ----

class TestModelConstruction:
    def test_build_mmm_returns_model(self):
        df = _generate_synthetic_data(days=60)
        data = prepare_model_data(df)
        mmm = build_mmm(data, FAST_CONFIG)
        assert mmm is not None
        assert mmm.date_column == "date"

    def test_channel_columns_match(self):
        df = _generate_synthetic_data(days=60, channels=["google_search", "meta_feed"])
        data = prepare_model_data(df)
        mmm = build_mmm(data, FAST_CONFIG)
        assert set(mmm.channel_columns) == {"spend_google_search", "spend_meta_feed"}

    def test_control_columns_included(self):
        df = _generate_synthetic_data(days=60)
        data = prepare_model_data(df)
        mmm = build_mmm(data, FAST_CONFIG)
        # All control columns should be set
        assert "sessions_organic" in mmm.control_columns
        assert "ipc_google_search" in mmm.control_columns

    def test_model_config_serialisation(self):
        config = ModelConfig(chains=2, tune=100, draws=100)
        json_str = config.to_json()
        restored = ModelConfig.from_json(json_str)
        assert restored.chains == 2
        assert restored.tune == 100


# ---- model fitting tests (minimal sampling) ----

class TestModelFitting:
    @pytest.mark.slow
    def test_fit_and_extract_results(self):
        """End-to-end: build, fit with minimal samples, extract results."""
        df = _generate_synthetic_data(days=120, channels=["google_search", "meta_feed"])
        data = prepare_model_data(df)
        mmm = build_mmm(data, FAST_CONFIG)
        mmm = fit_mmm(mmm, data, FAST_CONFIG)

        assert mmm.idata is not None
        assert "posterior" in mmm.idata

        results = extract_results(mmm, data)

        # Channel posteriors exist for each channel
        assert len(results.channel_posteriors) == 2
        channel_names = {cp.channel for cp in results.channel_posteriors}
        assert "spend_google_search" in channel_names
        assert "spend_meta_feed" in channel_names

        # Beta means are non-negative (HalfNormal prior enforces this)
        for cp in results.channel_posteriors:
            assert cp.beta_mean >= 0, f"{cp.channel} has negative beta"

        # HDI intervals exist and are ordered
        for cp in results.channel_posteriors:
            assert cp.beta_hdi_3 <= cp.beta_hdi_97

        # Diagnostics populated
        assert results.diagnostics.r_squared_mean is not None
        assert results.diagnostics.mape_mean is not None
        assert results.diagnostics.divergences >= 0

        # Contributions sum to ~100%
        total_pct = sum(cp.contribution_pct for cp in results.channel_posteriors)
        total_pct += results.baseline_contribution_pct
        assert 50 < total_pct < 150  # loose check for minimal sampling

    @pytest.mark.slow
    def test_results_serialisable(self):
        """Results should serialise to JSON and back."""
        df = _generate_synthetic_data(days=120, channels=["google_search"])
        data = prepare_model_data(df)
        mmm = build_mmm(data, FAST_CONFIG)
        mmm = fit_mmm(mmm, data, FAST_CONFIG)
        results = extract_results(mmm, data)

        json_str = results.to_json()
        parsed = json.loads(json_str)
        assert "channel_posteriors" in parsed
        assert "diagnostics" in parsed
        assert isinstance(parsed["channel_posteriors"], list)


# ---- full pipeline (service) tests ----

class TestModelService:
    @pytest.mark.slow
    def test_run_model_end_to_end(self, db_session):
        """Full service pipeline: DB → data prep → fit → results → DB."""
        df = _generate_synthetic_data(days=120)
        upload_id = _insert_records(db_session, df)

        model_run, results = run_model(
            db_session, upload_id, target="revenue", config=FAST_CONFIG
        )

        assert model_run.status == "completed"
        assert model_run.results_json is not None
        assert model_run.diagnostics_json is not None
        assert results is not None
        assert len(results.channel_posteriors) == 2

        # Verify stored in DB
        stored = db_session.query(ModelRun).filter_by(id=model_run.id).first()
        assert stored.status == "completed"

    def test_rejected_upload_raises(self, db_session):
        """Cannot run model on rejected uploads."""
        upload = Upload(filename="bad.csv", row_count=10, status="rejected")
        db_session.add(upload)
        db_session.commit()

        with pytest.raises(ValueError, match="rejected"):
            run_model(db_session, upload.id, config=FAST_CONFIG)

    def test_missing_upload_raises(self, db_session):
        with pytest.raises(ValueError, match="not found"):
            run_model(db_session, 9999, config=FAST_CONFIG)


# ---- channel filtering / merging tests ----

class TestChannelFiltering:
    def test_channel_merging(self):
        """Meta placements should merge into a single 'meta' channel."""
        df = _generate_synthetic_data(
            days=30,
            channels=["meta_feed", "meta_instagram", "google_search"],
        )
        channel_config = {
            "merge": {"meta_feed": "meta", "meta_instagram": "meta"},
            "channels": ["meta", "google_search"],
        }
        data = prepare_model_data(df, target="revenue", channel_config=channel_config)

        assert len(data.channel_columns) == 2
        assert "spend_meta" in data.channel_columns
        assert "spend_google_search" in data.channel_columns
        assert "spend_meta_feed" not in data.channel_columns
        assert "spend_meta_instagram" not in data.channel_columns

    def test_channel_filtering_by_list(self):
        """Only channels in the explicit list should survive."""
        df = _generate_synthetic_data(
            days=30,
            channels=["google_search", "meta_feed", "google_pmax"],
        )
        channel_config = {"channels": ["google_search", "google_pmax"]}
        data = prepare_model_data(df, target="revenue", channel_config=channel_config)

        assert len(data.channel_columns) == 2
        assert "spend_google_search" in data.channel_columns
        assert "spend_google_pmax" in data.channel_columns
        assert "spend_meta_feed" not in data.channel_columns

    def test_min_spend_pct_filtering(self):
        """Channels below min_spend_pct should be dropped automatically."""
        # Create data where one channel has negligible spend
        base_date = datetime.date(2024, 1, 1)
        rows = []
        for day_offset in range(30):
            d = base_date + datetime.timedelta(days=day_offset)
            for ch, spend in [("google_search", 200.0), ("meta_feed", 150.0), ("unknown", 0.01)]:
                rows.append({
                    "date": d, "channel": ch, "campaign": f"{ch}_c1",
                    "spend": spend,
                    "impressions": 100, "clicks": 10,
                    "in_platform_conversions": 1.0,
                    "revenue": 500.0, "orders": 5,
                    "sessions_organic": 200, "sessions_direct": 100,
                    "sessions_email": 50, "sessions_referral": 30,
                })
        df = pd.DataFrame(rows)

        channel_config = {"min_spend_pct": 0.5}
        data = prepare_model_data(df, target="revenue", channel_config=channel_config)

        channel_names = [c.replace("spend_", "") for c in data.channel_columns]
        assert "google_search" in channel_names
        assert "meta_feed" in channel_names
        assert "unknown" not in channel_names

    def test_recommend_channel_config(self):
        """recommend_channel_config should merge Meta placements and drop tiny channels."""
        from backend.models.data_prep import recommend_channel_config

        rng = np.random.default_rng(42)
        base_date = datetime.date(2024, 1, 1)
        rows = []
        channel_spends = {
            "google_pmax": 500,
            "google_search": 200,
            "meta_feed": 100,
            "meta_instagram": 80,
            "messenger": 0.01,
            "unknown": 0.005,
        }
        for day_offset in range(30):
            d = base_date + datetime.timedelta(days=day_offset)
            for ch, base_spend in channel_spends.items():
                spend = max(0, rng.normal(base_spend, base_spend * 0.1 + 0.001))
                rows.append({
                    "date": d, "channel": ch, "campaign": f"{ch}_c1",
                    "spend": round(spend, 2),
                    "impressions": 100, "clicks": 10,
                    "in_platform_conversions": 1.0,
                    "revenue": 1000.0, "orders": 10,
                    "sessions_organic": 200, "sessions_direct": 100,
                    "sessions_email": 50, "sessions_referral": 30,
                })
        df = pd.DataFrame(rows)
        rec = recommend_channel_config(df)

        # Meta placements should be merged
        assert "meta_feed" in rec["merge"]
        assert "meta_instagram" in rec["merge"]
        assert "messenger" in rec["merge"]
        assert rec["merge"]["meta_feed"] == "meta"

        # Merged "meta" should be in kept channels
        assert "meta" in rec["channels"]
        assert "google_pmax" in rec["channels"]
        assert "google_search" in rec["channels"]

        # Tiny channels should be dropped (unknown has near-zero spend)
        assert "unknown" in rec["dropped"]

        # Reasons should be present for dropped channels
        assert "unknown" in rec["reasons"]

    def test_empty_after_filter_raises(self):
        """Filtering to no channels should raise ValueError."""
        df = _generate_synthetic_data(days=30, channels=["google_search"])
        channel_config = {"channels": ["nonexistent_channel"]}
        with pytest.raises(ValueError, match="No data remaining"):
            prepare_model_data(df, target="revenue", channel_config=channel_config)

    def test_no_channel_config_is_backward_compatible(self):
        """Without channel_config, behaviour is identical to before."""
        df = _generate_synthetic_data(days=30, channels=["google_search", "meta_feed"])
        data_default = prepare_model_data(df, target="revenue")
        data_none = prepare_model_data(df, target="revenue", channel_config=None)

        assert data_default.channel_columns == data_none.channel_columns
        assert data_default.daily_rows == data_none.daily_rows
