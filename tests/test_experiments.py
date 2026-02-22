"""Tests for Phase 3: lag detection, spend-scaling, and product experiments."""
import json
import numpy as np
import pandas as pd
import pytest

from backend.experiments.lag_detection import (
    detect_lag,
    detect_channel_lags,
    detect_lags_from_prepared_data,
    ChannelLag,
    LagReport,
    DEFAULT_MAX_LAG,
    SIGNIFICANCE_THRESHOLD,
)
from backend.experiments.product_experiment import (
    run_product_experiment,
    validate_campaign_eligibility,
    ProductExperimentError,
    ProductExperimentResult,
    INVALID_CAMPAIGN_TYPES,
    INVALID_BIDDING_STRATEGIES,
    SELECTION_BIAS_EXPLANATION,
)
from backend.experiments.spend_scaling import (
    SpendScalingResult,
    _build_causalimpact_data,
)


# ============================================================
# Helpers
# ============================================================

def _make_daily_data(n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Create synthetic daily data with 2 channels and outcome."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    spend_a = rng.uniform(100, 500, n_days)
    spend_b = rng.uniform(50, 300, n_days)
    # Revenue correlated with spend_a (lagged by 3 days) + noise
    revenue = np.zeros(n_days)
    for i in range(n_days):
        lag_idx = max(0, i - 3)
        revenue[i] = 1000 + 2.0 * spend_a[lag_idx] + 0.5 * spend_b[i] + rng.normal(0, 100)

    return pd.DataFrame({
        "date": dates,
        "spend_channel_a": spend_a,
        "spend_channel_b": spend_b,
        "revenue": revenue,
    })


def _make_product_data(n_products: int = 100, n_advertised: int = 30, seed: int = 42) -> pd.DataFrame:
    """Create synthetic product performance data."""
    rng = np.random.default_rng(seed)
    n_adv = min(n_advertised, n_products)
    n_non_adv = n_products - n_adv
    product_ids = [f"PROD_{i:04d}" for i in range(n_products)]
    # Advertised products have higher revenue on average
    revenue = np.concatenate([
        rng.normal(500, 100, n_adv),
        rng.normal(300, 100, n_non_adv) if n_non_adv > 0 else np.array([]),
    ])
    orders = np.concatenate([
        rng.poisson(10, n_adv),
        rng.poisson(6, n_non_adv) if n_non_adv > 0 else np.array([], dtype=int),
    ])
    return pd.DataFrame({
        "product_id": product_ids,
        "revenue": np.maximum(revenue, 0),
        "orders": orders,
    })


# ============================================================
# Lag Detection Tests
# ============================================================

class TestDetectLag:
    def test_basic_detection(self):
        """Detect lag returns valid structure and non-trivial correlation."""
        rng = np.random.default_rng(42)
        n = 200
        spend = rng.uniform(100, 500, n)
        outcome = spend * 2 + rng.normal(0, 50, n)  # correlated, lag 0

        lag, peak_corr, corr_list = detect_lag(spend, outcome, max_lag=10)

        assert 0 <= lag <= 10
        assert abs(peak_corr) > 0.1
        assert len(corr_list) >= 10  # at least max_lag values
        # Each correlation in valid range
        for c in corr_list:
            assert -1.0 <= c <= 1.0

    def test_zero_lag(self):
        """When outcome is directly correlated with spend, lag should be 0."""
        rng = np.random.default_rng(99)
        n = 200
        spend = rng.uniform(100, 500, n)
        outcome = spend * 2 + rng.normal(0, 20, n)

        lag, peak_corr, corr_list = detect_lag(spend, outcome, max_lag=10)

        assert lag == 0
        assert peak_corr > 0.5

    def test_short_series_reduces_max_lag(self):
        """Short data should automatically reduce max_lag."""
        rng = np.random.default_rng(42)
        n = 20  # much less than DEFAULT_MAX_LAG + 10
        spend = rng.uniform(100, 500, n)
        outcome = rng.uniform(500, 1500, n)

        lag, peak_corr, corr_list = detect_lag(spend, outcome)
        # max_lag should be reduced to n//3 = 6
        assert len(corr_list) <= n // 3 + 1

    def test_correlation_list_structure(self):
        rng = np.random.default_rng(42)
        n = 100
        spend = rng.uniform(100, 500, n)
        outcome = rng.uniform(500, 1500, n)

        lag, peak_corr, corr_list = detect_lag(spend, outcome, max_lag=10)

        # All values should be valid floats between -1 and 1
        for c in corr_list:
            assert isinstance(c, float)
            assert -1.0 <= c <= 1.0


class TestDetectChannelLags:
    def test_multi_channel(self):
        df = _make_daily_data(120)
        report = detect_channel_lags(
            df,
            channel_columns=["spend_channel_a", "spend_channel_b"],
            outcome_column="revenue",
            max_lag=10,
        )

        assert isinstance(report, LagReport)
        assert len(report.channel_lags) == 2
        assert report.max_lag_searched == 10

        # Channel A has a lag built into the synthetic data
        ch_a = next(cl for cl in report.channel_lags if cl.channel == "spend_channel_a")
        assert ch_a.optimal_lag_days >= 0
        assert isinstance(ch_a.significant, bool)

    def test_missing_channel_skipped(self):
        df = _make_daily_data(120)
        report = detect_channel_lags(
            df,
            channel_columns=["spend_channel_a", "nonexistent_channel"],
            outcome_column="revenue",
        )
        # Only the existing channel should be in results
        assert len(report.channel_lags) == 1
        assert report.channel_lags[0].channel == "spend_channel_a"

    def test_zero_variance_channel(self):
        df = _make_daily_data(120)
        df["spend_constant"] = 100.0  # constant spend — no variance

        report = detect_channel_lags(
            df,
            channel_columns=["spend_channel_a", "spend_constant"],
            outcome_column="revenue",
            max_lag=10,
        )

        constant_lag = next(cl for cl in report.channel_lags if cl.channel == "spend_constant")
        assert constant_lag.optimal_lag_days == 0
        assert constant_lag.peak_correlation == 0.0
        assert constant_lag.significant is False

    def test_missing_outcome_raises(self):
        df = _make_daily_data(120)
        with pytest.raises(ValueError, match="Outcome column"):
            detect_channel_lags(df, ["spend_channel_a"], outcome_column="nonexistent")

    def test_serialisation(self):
        df = _make_daily_data(120)
        report = detect_channel_lags(
            df, ["spend_channel_a"], outcome_column="revenue", max_lag=5,
        )
        d = report.to_dict()
        assert "channel_lags" in d
        s = report.to_json()
        parsed = json.loads(s)
        assert len(parsed["channel_lags"]) == 1


class TestDetectLagsFromPreparedData:
    def test_wrapper(self):
        df = _make_daily_data(120)
        X = df[["spend_channel_a", "spend_channel_b"]].copy()
        y = df["revenue"].copy()
        y.name = "revenue"

        report = detect_lags_from_prepared_data(
            X, y, channel_columns=["spend_channel_a", "spend_channel_b"], max_lag=10,
        )
        assert len(report.channel_lags) == 2


# ============================================================
# Product Experiment Tests
# ============================================================

class TestCampaignEligibility:
    """HARD CONSTRAINT: PMax and auto-bidding must be refused."""

    def test_pmax_refused(self):
        with pytest.raises(ProductExperimentError, match="selection bias"):
            validate_campaign_eligibility("pmax", "manual_cpc")

    def test_performance_max_refused(self):
        with pytest.raises(ProductExperimentError, match="selection bias"):
            validate_campaign_eligibility("Performance Max", "manual_cpc")

    def test_smart_shopping_refused(self):
        with pytest.raises(ProductExperimentError, match="selection bias"):
            validate_campaign_eligibility("Smart Shopping", "manual_cpc")

    def test_target_roas_refused(self):
        with pytest.raises(ProductExperimentError, match="selection bias"):
            validate_campaign_eligibility("shopping", "target_roas")

    def test_target_cpa_refused(self):
        with pytest.raises(ProductExperimentError, match="selection bias"):
            validate_campaign_eligibility("shopping", "Target CPA")

    def test_maximize_conversions_refused(self):
        with pytest.raises(ProductExperimentError, match="selection bias"):
            validate_campaign_eligibility("search", "maximize_conversions")

    def test_maximize_conversion_value_refused(self):
        with pytest.raises(ProductExperimentError, match="selection bias"):
            validate_campaign_eligibility("search", "Maximize Conversion Value")

    def test_manual_cpc_allowed(self):
        # Should not raise
        validate_campaign_eligibility("shopping", "manual_cpc")

    def test_manual_cpm_allowed(self):
        validate_campaign_eligibility("search", "manual_cpm")

    def test_whitespace_handling(self):
        with pytest.raises(ProductExperimentError):
            validate_campaign_eligibility("  PMax  ", "manual_cpc")

    def test_case_insensitive(self):
        with pytest.raises(ProductExperimentError):
            validate_campaign_eligibility("PMAX", "manual_cpc")
        with pytest.raises(ProductExperimentError):
            validate_campaign_eligibility("shopping", "TARGET_ROAS")


class TestProductExperiment:
    def test_valid_experiment(self):
        df = _make_product_data(100)
        advertised = [f"PROD_{i:04d}" for i in range(30)]

        result = run_product_experiment(
            product_data=df,
            advertised_products=advertised,
            campaign_name="Test Shopping",
            campaign_type="shopping",
            bidding_strategy="manual_cpc",
        )

        assert isinstance(result, ProductExperimentResult)
        assert result.advertised_product_count == 30
        assert result.non_advertised_product_count == 70
        assert result.revenue_lift_pct > 0  # advertised products have higher revenue
        assert result.campaign_name == "Test Shopping"
        assert result.campaign_type == "shopping"
        assert result.bidding_strategy == "manual_cpc"

    def test_pmax_campaign_refused(self):
        df = _make_product_data(100)
        advertised = [f"PROD_{i:04d}" for i in range(30)]

        with pytest.raises(ProductExperimentError, match="selection bias"):
            run_product_experiment(
                product_data=df,
                advertised_products=advertised,
                campaign_name="PMax Campaign",
                campaign_type="pmax",
                bidding_strategy="manual_cpc",
            )

    def test_auto_bidding_refused(self):
        df = _make_product_data(100)
        advertised = [f"PROD_{i:04d}" for i in range(30)]

        with pytest.raises(ProductExperimentError, match="selection bias"):
            run_product_experiment(
                product_data=df,
                advertised_products=advertised,
                campaign_name="Shopping Campaign",
                campaign_type="shopping",
                bidding_strategy="target_roas",
            )

    def test_missing_columns(self):
        df = pd.DataFrame({"product_id": ["a"], "revenue": [100]})  # no orders
        with pytest.raises(ValueError, match="Missing required columns"):
            run_product_experiment(df, ["a"], "c", "shopping", "manual_cpc")

    def test_no_advertised_products_found(self):
        df = _make_product_data(100)
        with pytest.raises(ValueError, match="No advertised products"):
            run_product_experiment(df, ["NONEXISTENT"], "c", "shopping", "manual_cpc")

    def test_no_non_advertised_products(self):
        df = _make_product_data(5, n_advertised=5)
        all_products = df["product_id"].tolist()
        with pytest.raises(ValueError, match="No non-advertised"):
            run_product_experiment(df, all_products, "c", "shopping", "manual_cpc")

    def test_small_sample_warnings(self):
        df = _make_product_data(40)
        advertised = [f"PROD_{i:04d}" for i in range(10)]  # only 10 advertised

        result = run_product_experiment(df, advertised, "c", "shopping", "manual_cpc")

        assert any("small sample" in w.lower() for w in result.warnings)

    def test_significance_fields_present(self):
        df = _make_product_data(100)
        advertised = [f"PROD_{i:04d}" for i in range(30)]

        result = run_product_experiment(df, advertised, "c", "shopping", "manual_cpc")

        assert isinstance(result.revenue_significant, bool)
        assert isinstance(result.orders_significant, bool)
        assert 0.0 <= result.p_value_revenue <= 1.0
        assert 0.0 <= result.p_value_orders <= 1.0

    def test_serialisation(self):
        df = _make_product_data(100)
        advertised = [f"PROD_{i:04d}" for i in range(30)]

        result = run_product_experiment(df, advertised, "c", "shopping", "manual_cpc")

        d = result.to_dict()
        assert "advertised_product_count" in d
        assert "warnings" in d

        s = result.to_json()
        parsed = json.loads(s)
        assert parsed["advertised_product_count"] == 30


# ============================================================
# Spend-Scaling: CausalImpact data preparation tests
# ============================================================

class TestBuildCausalImpactData:
    """Test the data preparation step (CausalImpact itself is integration-level)."""

    def test_builds_correct_structure(self):
        df = _make_daily_data(60)
        ci_data = _build_causalimpact_data(
            df,
            target_channel="spend_channel_a",
            outcome_column="revenue",
            channel_columns=["spend_channel_a", "spend_channel_b"],
        )

        # Column 0 should be the outcome, columns 1+ should be controls (excluding target)
        assert ci_data.columns[0] == "revenue"
        assert "spend_channel_b" in ci_data.columns
        assert "spend_channel_a" not in ci_data.columns  # target excluded from controls
        assert pd.api.types.is_datetime64_any_dtype(ci_data.index)

    def test_no_controls_raises(self):
        df = _make_daily_data(60)
        with pytest.raises(ValueError, match="(?i)at least one other channel"):
            _build_causalimpact_data(
                df,
                target_channel="spend_channel_a",
                outcome_column="revenue",
                channel_columns=["spend_channel_a"],  # only one channel — no controls
            )

    def test_multiple_controls(self):
        df = _make_daily_data(60)
        df["spend_channel_c"] = np.random.default_rng(1).uniform(50, 200, 60)

        ci_data = _build_causalimpact_data(
            df,
            target_channel="spend_channel_a",
            outcome_column="revenue",
            channel_columns=["spend_channel_a", "spend_channel_b", "spend_channel_c"],
        )

        assert "spend_channel_b" in ci_data.columns
        assert "spend_channel_c" in ci_data.columns
        assert "spend_channel_a" not in ci_data.columns


class TestSpendScalingResult:
    def test_serialisation(self):
        result = SpendScalingResult(
            channel="spend_google_search",
            pre_period_start="2024-01-01",
            pre_period_end="2024-03-31",
            post_period_start="2024-04-01",
            post_period_end="2024-04-30",
            estimated_impact=5000.0,
            impact_ci_lower=2000.0,
            impact_ci_upper=8000.0,
            relative_effect=0.15,
            relative_effect_ci_lower=0.05,
            relative_effect_ci_upper=0.25,
            p_value=0.01,
            significant=True,
            summary="Test summary",
        )

        d = result.to_dict()
        assert d["channel"] == "spend_google_search"
        assert d["significant"] is True

        s = result.to_json()
        parsed = json.loads(s)
        assert parsed["estimated_impact"] == 5000.0

    def test_mmm_calibration_fields(self):
        result = SpendScalingResult(
            channel="ch",
            pre_period_start="2024-01-01",
            pre_period_end="2024-03-31",
            post_period_start="2024-04-01",
            post_period_end="2024-04-30",
            estimated_impact=5000.0,
            impact_ci_lower=2000.0,
            impact_ci_upper=8000.0,
            relative_effect=0.15,
            relative_effect_ci_lower=0.05,
            relative_effect_ci_upper=0.25,
            p_value=0.01,
            significant=True,
            mmm_prediction=3000.0,
            discrepancy_pct=40.0,
            recalibration_recommended=True,
            summary="Test",
        )
        d = result.to_dict()
        assert d["recalibration_recommended"] is True
        assert d["discrepancy_pct"] == 40.0
