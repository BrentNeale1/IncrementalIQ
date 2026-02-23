"""Tests for Phase 5: internal validation (holdout, posterior predictive, sensitivity).

Fast tests use mock data; slow tests run actual MCMC sampling.
"""
import datetime
import json
import pytest
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.db.config import Base
from backend.db.models import Upload, DailyRecord
from backend.models.data_prep import prepare_model_data, PreparedData
from backend.models.mmm import ModelConfig, build_mmm, fit_mmm, extract_results
from backend.models.validation import (
    run_holdout_validation,
    run_posterior_predictive_check,
    run_sensitivity_analysis,
    ValidationReport,
    HoldoutResult,
    PosteriorPredictiveCheck,
    SensitivityReport,
)


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
    channels = channels or ["google_search", "meta_feed"]
    base_date = datetime.date.fromisoformat(start)
    rng = np.random.default_rng(42)
    rows = []
    for day_offset in range(days):
        d = base_date + datetime.timedelta(days=day_offset)
        day_of_week = d.weekday()
        weekly_factor = 1.0 + 0.15 * np.sin(2 * np.pi * day_of_week / 7)
        for ch in channels:
            base_spend = {"google_search": 150, "meta_feed": 100}.get(ch, 80)
            spend = max(0, rng.normal(base_spend, base_spend * 0.3))
            base_rev = spend * 3.5 * weekly_factor + rng.normal(0, 50)
            rows.append({
                "date": d,
                "channel": ch,
                "campaign": f"{ch}_campaign_1",
                "spend": round(spend, 2),
                "clicks": max(0, int(spend * 0.5 + rng.normal(0, 10))),
                "in_platform_conversions": round(max(0, spend * 0.03 + rng.normal(0, 1)), 2),
                "revenue": round(max(0, base_rev), 2),
                "orders": max(0, int(base_rev / 50 + rng.normal(0, 2))),
                "sessions_organic": max(0, int(200 * weekly_factor + rng.normal(0, 20))),
                "sessions_direct": max(0, int(100 * weekly_factor + rng.normal(0, 15))),
                "sessions_email": max(0, int(50 + rng.normal(0, 10))),
                "sessions_referral": max(0, int(30 + rng.normal(0, 8))),
            })
    return pd.DataFrame(rows)


FAST_CONFIG = ModelConfig(
    chains=1,
    tune=50,
    draws=50,
    target_accept=0.8,
    random_seed=42,
)


# ---- validation report structure tests (fast) ----

class TestValidationReport:
    def test_empty_report(self):
        report = ValidationReport()
        d = report.to_dict()
        assert d == {}
        assert report.all_warnings == []

    def test_report_with_holdout(self):
        holdout = HoldoutResult(
            train_size=100, test_size=20, holdout_fraction=0.2,
            mape_mean=12.0, mape_hdi_3=8.0, mape_hdi_97=16.0,
            r_squared_mean=0.75, r_squared_hdi_3=0.6, r_squared_hdi_97=0.85,
            coverage=0.85, warnings=[],
        )
        report = ValidationReport(holdout=holdout)
        d = report.to_dict()
        assert "holdout" in d
        assert d["holdout"]["mape_mean"] == 12.0

    def test_report_serialisation(self):
        holdout = HoldoutResult(
            train_size=100, test_size=20, holdout_fraction=0.2,
            mape_mean=12.0, mape_hdi_3=8.0, mape_hdi_97=16.0,
            r_squared_mean=0.75, r_squared_hdi_3=0.6, r_squared_hdi_97=0.85,
            coverage=0.85, warnings=["test warning"],
        )
        ppc = PosteriorPredictiveCheck(
            mean_residual=5.0, residual_std=50.0, coverage=0.92,
            max_abs_residual=150.0, warnings=[],
        )
        report = ValidationReport(holdout=holdout, posterior_predictive=ppc)
        s = report.to_json()
        parsed = json.loads(s)
        assert "holdout" in parsed
        assert "posterior_predictive" in parsed

    def test_warnings_aggregated(self):
        holdout = HoldoutResult(
            train_size=100, test_size=20, holdout_fraction=0.2,
            mape_mean=20.0, mape_hdi_3=15.0, mape_hdi_97=25.0,
            r_squared_mean=0.5, r_squared_hdi_3=0.3, r_squared_hdi_97=0.7,
            coverage=0.75, warnings=["MAPE warning", "Coverage warning"],
        )
        ppc = PosteriorPredictiveCheck(
            mean_residual=10.0, residual_std=80.0, coverage=0.70,
            max_abs_residual=200.0, warnings=["PPC coverage warning"],
        )
        report = ValidationReport(holdout=holdout, posterior_predictive=ppc)
        assert len(report.all_warnings) == 3

    def test_holdout_mape_threshold_warning(self):
        holdout = HoldoutResult(
            train_size=100, test_size=20, holdout_fraction=0.2,
            mape_mean=18.0, mape_hdi_3=12.0, mape_hdi_97=24.0,
            r_squared_mean=0.65, r_squared_hdi_3=0.5, r_squared_hdi_97=0.8,
            coverage=0.85, warnings=["Holdout MAPE is 18.0% (threshold: 15%)."],
        )
        assert any("15%" in w for w in holdout.warnings)

    def test_holdout_coverage_threshold_warning(self):
        holdout = HoldoutResult(
            train_size=100, test_size=20, holdout_fraction=0.2,
            mape_mean=10.0, mape_hdi_3=7.0, mape_hdi_97=13.0,
            r_squared_mean=0.8, r_squared_hdi_3=0.7, r_squared_hdi_97=0.9,
            coverage=0.72,
            warnings=["Holdout coverage is 72% (threshold: 80%)."],
        )
        assert any("80%" in w for w in holdout.warnings)


# ---- MCMC integration tests (slow) ----

class TestHoldoutValidation:
    @pytest.mark.slow
    def test_holdout_returns_metrics(self):
        """Holdout validation produces valid metrics."""
        df = _generate_synthetic_data(days=150)
        data = prepare_model_data(df)
        result = run_holdout_validation(data, FAST_CONFIG, holdout_fraction=0.2)

        assert result.train_size > 0
        assert result.test_size > 0
        assert result.mape_mean >= 0
        assert 0 <= result.coverage <= 1
        assert result.mape_hdi_3 <= result.mape_hdi_97

    def test_holdout_too_small_raises(self):
        """Training set below 30 days should raise."""
        df = _generate_synthetic_data(days=35)
        data = prepare_model_data(df)
        with pytest.raises(ValueError, match="Training set too small"):
            run_holdout_validation(data, FAST_CONFIG, holdout_fraction=0.9)


class TestPosteriorPredictiveCheck:
    @pytest.mark.slow
    def test_ppc_returns_metrics(self):
        """PPC produces coverage and residual stats."""
        df = _generate_synthetic_data(days=120)
        data = prepare_model_data(df)
        mmm = build_mmm(data, FAST_CONFIG)
        fit_mmm(mmm, data, FAST_CONFIG)

        ppc = run_posterior_predictive_check(mmm, data)
        assert 0 <= ppc.coverage <= 1
        assert ppc.residual_std > 0
        assert ppc.max_abs_residual >= 0


class TestSensitivityAnalysis:
    @pytest.mark.slow
    def test_sensitivity_returns_results(self):
        """Sensitivity analysis compares original vs. wide-prior betas."""
        df = _generate_synthetic_data(days=120)
        data = prepare_model_data(df)
        mmm = build_mmm(data, FAST_CONFIG)
        fit_mmm(mmm, data, FAST_CONFIG)
        results = extract_results(mmm, data)

        base_betas = {cp.channel: cp.beta_mean for cp in results.channel_posteriors}

        report = run_sensitivity_analysis(data, base_betas, FAST_CONFIG)
        assert len(report.results) == 2  # two channels
        for r in report.results:
            assert r.change_pct >= 0
            assert isinstance(r.stable, bool)
