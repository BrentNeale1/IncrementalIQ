"""Tests for Phase 4 output views (simple, intermediate, advanced) and trust score."""
import json
import pytest
from backend.models.mmm import ModelResults, ChannelPosterior, ModelDiagnostics
from backend.outputs.trust_score import (
    compute_trust_score,
    compute_data_quality_score,
    compute_model_fit_score,
    compute_calibration_score,
    TIER_RELIABLE,
    TIER_CAUTION,
    TIER_INSUFFICIENT,
)
from backend.outputs.views import (
    build_simple_view,
    build_intermediate_view,
    build_advanced_view,
    _confidence_label,
    _channel_display_name,
    _generate_recommendations,
    OBSERVATIONAL_CAVEAT,
)


# ---- helpers ----

def _make_channel_posterior(
    channel: str = "spend_google_search",
    beta_mean: float = 0.5,
    beta_sd: float = 0.1,
    contribution_pct: float = 30.0,
    contribution_mean: float = 5000.0,
    **kwargs,
) -> ChannelPosterior:
    defaults = dict(
        channel=channel,
        beta_mean=beta_mean,
        beta_sd=beta_sd,
        beta_hdi_3=beta_mean - 2 * beta_sd,
        beta_hdi_97=beta_mean + 2 * beta_sd,
        adstock_alpha_mean=0.3,
        adstock_alpha_sd=0.05,
        adstock_alpha_hdi_3=0.2,
        adstock_alpha_hdi_97=0.4,
        saturation_lam_mean=2.0,
        saturation_lam_sd=0.3,
        saturation_lam_hdi_3=1.4,
        saturation_lam_hdi_97=2.6,
        contribution_mean=contribution_mean,
        contribution_pct=contribution_pct,
        contribution_hdi_3=contribution_mean * 0.7,
        contribution_hdi_97=contribution_mean * 1.3,
    )
    defaults.update(kwargs)
    return ChannelPosterior(**defaults)


def _make_diagnostics(**kwargs) -> ModelDiagnostics:
    defaults = dict(
        r_squared_mean=0.85,
        r_squared_hdi_3=0.78,
        r_squared_hdi_97=0.92,
        mape_mean=10.5,
        mape_hdi_3=8.0,
        mape_hdi_97=13.0,
        divergences=0,
        warnings=[],
    )
    defaults.update(kwargs)
    return ModelDiagnostics(**defaults)


def _make_results(**kwargs) -> ModelResults:
    channels = kwargs.pop("channels", [
        _make_channel_posterior("spend_google_search", contribution_pct=40),
        _make_channel_posterior("spend_meta_feed", contribution_pct=25),
    ])
    return ModelResults(
        channel_posteriors=channels,
        baseline_contribution_pct=kwargs.pop("baseline_pct", 35.0),
        diagnostics=kwargs.pop("diagnostics", _make_diagnostics()),
    )


def _make_trust(**kwargs):
    defaults = dict(
        history_status="sufficient",
        gap_count=0,
        total_days=600,
        low_variance_channels=[],
        total_channels=2,
        mape_mean=10.5,
        r_squared_mean=0.85,
        divergences=0,
        total_draws=4000,
    )
    defaults.update(kwargs)
    return compute_trust_score(**defaults)


# ---- trust score tests ----

class TestTrustScore:
    def test_reliable_tier(self):
        trust = _make_trust()
        assert trust.overall_tier == TIER_RELIABLE
        assert trust.overall_score > 0.7

    def test_caution_tier_bad_mape(self):
        trust = _make_trust(
            history_status="caution",
            mape_mean=20.0, r_squared_mean=0.6,
            low_variance_channels=["ch1"], total_channels=2,
        )
        assert trust.overall_tier == TIER_CAUTION

    def test_insufficient_tier_rejected(self):
        trust = _make_trust(history_status="rejected", mape_mean=30.0, r_squared_mean=0.3)
        assert trust.overall_tier == TIER_INSUFFICIENT

    def test_data_quality_score_ranges(self):
        score, _ = compute_data_quality_score("sufficient", 0, 600, [], 3)
        assert score == 1.0

        score, flags = compute_data_quality_score("rejected", 100, 200, ["ch1", "ch2"], 3)
        assert score < 0.5
        assert any("9 months" in f for f in flags)

    def test_model_fit_score_ranges(self):
        score, _ = compute_model_fit_score(8.0, 0.9, 0, 4000)
        assert score == 1.0

        score, flags = compute_model_fit_score(30.0, 0.4, 500, 4000)
        assert score < 0.5
        assert len(flags) >= 2

    def test_calibration_neutral_default(self):
        score, flags = compute_calibration_score(None)
        assert score == 0.5
        assert any("No experiment" in f for f in flags)

    def test_calibration_with_experiment(self):
        score, _ = compute_calibration_score(0.05)
        assert score == 1.0

        score, flags = compute_calibration_score(0.35)
        assert score < 0.5
        assert any("Recalibration" in f for f in flags)

    def test_holdout_flags_in_model_fit(self):
        _, flags = compute_model_fit_score(10, 0.8, 0, 4000, holdout_mape=20.0)
        assert any("Holdout MAPE" in f for f in flags)

        _, flags = compute_model_fit_score(10, 0.8, 0, 4000, holdout_coverage=0.70)
        assert any("coverage" in f for f in flags)

    def test_trust_score_serialisable(self):
        trust = _make_trust()
        d = trust.to_dict()
        assert "overall_tier" in d
        s = trust.to_json()
        parsed = json.loads(s)
        assert parsed["overall_tier"] == TIER_RELIABLE


# ---- confidence label tests ----

class TestConfidenceLabels:
    def test_high_confidence(self):
        assert "High" in _confidence_label(20, 0.1, 0.5)

    def test_moderate_confidence(self):
        assert "Moderate" in _confidence_label(50, 0.25, 0.5)

    def test_low_confidence(self):
        assert "Low" in _confidence_label(100, 0.8, 0.5)

    def test_zero_beta(self):
        assert "Low" in _confidence_label(0, 0, 0)


# ---- channel display names ----

class TestChannelDisplayNames:
    def test_known_channels(self):
        assert _channel_display_name("spend_google_search") == "Google Search"
        assert _channel_display_name("spend_meta_feed") == "Meta (Facebook Feed)"
        assert _channel_display_name("spend_google_youtube") == "YouTube"

    def test_unknown_channel(self):
        assert _channel_display_name("spend_tiktok_ads") == "Tiktok Ads"


# ---- simple view tests ----

class TestSimpleView:
    def test_structure(self):
        results = _make_results()
        trust = _make_trust()
        view = build_simple_view(results, trust)

        assert len(view.channels) == 2
        assert view.trust_tier == TIER_RELIABLE
        assert view.caveat  # observational inference caveat present

    def test_channels_sorted_by_contribution(self):
        results = _make_results()
        trust = _make_trust()
        view = build_simple_view(results, trust)
        pcts = [c.contribution_pct for c in view.channels]
        assert pcts == sorted(pcts, reverse=True)

    def test_confidence_labels_present(self):
        results = _make_results()
        trust = _make_trust()
        view = build_simple_view(results, trust)
        for ch in view.channels:
            assert ch.confidence in [
                "High confidence",
                "Moderate confidence",
                "Low confidence — more data needed",
            ]

    def test_serialisable(self):
        results = _make_results()
        trust = _make_trust()
        view = build_simple_view(results, trust)
        d = view.to_dict()
        s = view.to_json()
        parsed = json.loads(s)
        assert "channels" in parsed
        assert "caveat" in parsed


# ---- intermediate view tests ----

class TestIntermediateView:
    def test_structure(self):
        results = _make_results()
        trust = _make_trust()
        view = build_intermediate_view(results, trust)

        assert len(view.channels) == 2
        assert len(view.adstock_curves) == 2
        assert len(view.saturation_curves) == 2
        assert "overall_tier" in view.trust

    def test_adstock_curves(self):
        results = _make_results()
        trust = _make_trust()
        view = build_intermediate_view(results, trust)

        for curve in view.adstock_curves:
            assert len(curve.decay_curve) == 9  # l_max=8 → 0..8
            assert abs(sum(curve.decay_curve) - 1.0) < 0.01  # normalised
            assert curve.half_life_days > 0

    def test_saturation_curves(self):
        results = _make_results()
        trust = _make_trust()
        view = build_intermediate_view(results, trust, channel_max_spends={"spend_google_search": 200})

        for curve in view.saturation_curves:
            assert len(curve.curve_x) == 50
            assert len(curve.curve_y) == 50
            assert curve.curve_y[0] == 0.0  # zero spend = zero response
            assert curve.curve_y[-1] > curve.curve_y[0]  # monotonic increase

    def test_in_platform_conversions_note(self):
        results = _make_results()
        trust = _make_trust()
        view = build_intermediate_view(results, trust)
        for ch in view.channels:
            assert "covariate" in ch.in_platform_conversions_note.lower()

    def test_serialisable(self):
        results = _make_results()
        trust = _make_trust()
        view = build_intermediate_view(results, trust)
        s = view.to_json()
        parsed = json.loads(s)
        assert "adstock_curves" in parsed
        assert "saturation_curves" in parsed


# ---- recommendation engine tests ----

class TestRecommendations:
    def test_wide_interval_triggers_experiment_rec(self):
        # Create a channel with very wide intervals
        wide_ch = _make_channel_posterior(
            "spend_google_search",
            beta_mean=0.5,
            beta_sd=0.5,
            beta_hdi_3=0.01,
            beta_hdi_97=1.5,
        )
        results = _make_results(channels=[wide_ch])
        trust = _make_trust()
        recs = _generate_recommendations(results, trust)
        assert any(r["type"] == "experiment_recommended" for r in recs)

    def test_low_contribution_flagged(self):
        low_ch = _make_channel_posterior(
            "spend_meta_feed",
            contribution_pct=2.0,
            beta_mean=0.1,
        )
        results = _make_results(channels=[low_ch])
        trust = _make_trust()
        recs = _generate_recommendations(results, trust)
        assert any(r["type"] == "underperforming" for r in recs)

    def test_insufficient_data_critical_rec(self):
        results = _make_results()
        trust = _make_trust(history_status="rejected", mape_mean=30, r_squared_mean=0.3)
        recs = _generate_recommendations(results, trust)
        assert recs[0]["priority"] == "critical"


# ---- advanced view tests ----

class TestAdvancedView:
    def test_structure(self):
        results = _make_results()
        trust = _make_trust()
        view = build_advanced_view(results, trust)

        assert len(view.channels) == 2
        assert view.baseline_contribution_pct == 35.0
        assert view.diagnostics is not None
        assert view.caveat == OBSERVATIONAL_CAVEAT
        assert isinstance(view.data_quality_flags, list)
        assert isinstance(view.experiments, list)
        assert view.validation is None  # no validation passed

    def test_channels_sorted_by_contribution(self):
        results = _make_results()
        trust = _make_trust()
        view = build_advanced_view(results, trust)
        pcts = [c.contribution_pct for c in view.channels]
        assert pcts == sorted(pcts, reverse=True)

    def test_full_posterior_fields(self):
        """Advanced view must expose mean, SD, and 94% HDI for all parameters."""
        results = _make_results()
        trust = _make_trust()
        view = build_advanced_view(results, trust)

        for ch in view.channels:
            # Beta
            assert ch.beta_mean > 0
            assert ch.beta_sd > 0
            assert ch.beta_hdi_3 < ch.beta_hdi_97
            # Adstock
            assert 0 <= ch.adstock_alpha_mean <= 1
            assert ch.adstock_alpha_hdi_3 < ch.adstock_alpha_hdi_97
            # Saturation
            assert ch.saturation_lam_mean > 0
            assert ch.saturation_lam_hdi_3 < ch.saturation_lam_hdi_97
            # Contribution
            assert ch.contribution_pct > 0
            assert ch.contribution_hdi_3 < ch.contribution_hdi_97
            # Display name
            assert ch.display_name != ch.channel

    def test_diagnostics_structure(self):
        results = _make_results()
        trust = _make_trust()
        view = build_advanced_view(results, trust, total_draws=4000)

        diag = view.diagnostics
        assert diag.r_squared_mean > 0
        assert diag.r_squared_hdi_3 < diag.r_squared_hdi_97
        assert diag.mape_mean > 0
        assert diag.mape_hdi_3 < diag.mape_hdi_97
        assert diag.divergences == 0
        assert diag.divergence_pct == 0.0

    def test_divergence_percentage(self):
        results = _make_results(diagnostics=_make_diagnostics(divergences=100))
        trust = _make_trust()
        view = build_advanced_view(results, trust, total_draws=4000)
        assert view.diagnostics.divergence_pct == 2.5  # 100/4000 * 100

    def test_data_quality_flags_from_trust(self):
        trust = _make_trust(
            history_status="caution",
            low_variance_channels=["ch1"],
            total_channels=2,
        )
        results = _make_results()
        view = build_advanced_view(results, trust)
        assert len(view.data_quality_flags) > 0

    def test_trust_dict_included(self):
        results = _make_results()
        trust = _make_trust()
        view = build_advanced_view(results, trust)
        assert "overall_tier" in view.trust
        assert "overall_score" in view.trust

    def test_with_validation(self):
        results = _make_results()
        trust = _make_trust()
        validation = {
            "holdout": {
                "train_size": 240,
                "test_size": 60,
                "holdout_fraction": 0.2,
                "mape_mean": 12.5,
                "mape_hdi_3": 10.0,
                "mape_hdi_97": 15.0,
                "r_squared_mean": 0.82,
                "r_squared_hdi_3": 0.75,
                "r_squared_hdi_97": 0.89,
                "coverage": 0.92,
                "warnings": [],
            },
        }
        view = build_advanced_view(results, trust, validation=validation)
        assert view.validation is not None
        assert view.validation["holdout"]["mape_mean"] == 12.5

    def test_with_experiments(self):
        results = _make_results()
        trust = _make_trust()
        experiments = [
            {
                "channel": "spend_google_search",
                "estimated_impact": 5000.0,
                "impact_ci_lower": 2000.0,
                "impact_ci_upper": 8000.0,
                "significant": True,
                "p_value": 0.01,
                "recalibration_recommended": False,
            },
        ]
        view = build_advanced_view(results, trust, experiments=experiments)
        assert len(view.experiments) == 1
        assert view.experiments[0]["channel"] == "spend_google_search"

    def test_no_experiments_is_empty_list(self):
        results = _make_results()
        trust = _make_trust()
        view = build_advanced_view(results, trust)
        assert view.experiments == []

    def test_recommendations_present(self):
        results = _make_results()
        trust = _make_trust()
        view = build_advanced_view(results, trust)
        assert isinstance(view.recommendations, list)

    def test_serialisable(self):
        results = _make_results()
        trust = _make_trust()
        validation = {"holdout": {"mape_mean": 12.0, "warnings": []}}
        experiments = [{"channel": "ch1", "impact": 500}]
        view = build_advanced_view(
            results, trust,
            validation=validation,
            experiments=experiments,
        )

        d = view.to_dict()
        assert "channels" in d
        assert "diagnostics" in d
        assert "data_quality_flags" in d
        assert "validation" in d
        assert "experiments" in d
        assert "caveat" in d

        s = view.to_json()
        parsed = json.loads(s)
        assert len(parsed["channels"]) == 2
        assert parsed["diagnostics"]["r_squared_mean"] > 0
        assert parsed["validation"]["holdout"]["mape_mean"] == 12.0
        assert len(parsed["experiments"]) == 1

    def test_no_p_values_in_channel_output(self):
        """CLAUDE.md: advanced view uses 94% HDI, not p-values."""
        results = _make_results()
        trust = _make_trust()
        view = build_advanced_view(results, trust)
        d = view.to_dict()
        # Channel results should have HDI fields, not p-value fields
        for ch in d["channels"]:
            assert "beta_hdi_3" in ch
            assert "beta_hdi_97" in ch
            assert "p_value" not in ch
