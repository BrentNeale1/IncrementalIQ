"""Output views: simple, intermediate, and advanced report generation.

Simple view:    plain-language, no statistical terminology, confidence labels.
Intermediate:   confidence intervals, adstock/saturation curves, lag summary.
Advanced view:  full posterior distributions, model diagnostics, validation, experiments.

Design rule (CLAUDE.md): uncertainty is represented at every output level.
Caveat (CLAUDE.md): MMM is observational causal inference, not RCT-level certainty.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
import numpy as np

from backend.models.mmm import ModelResults, ChannelPosterior
from backend.outputs.trust_score import TrustScore


OBSERVATIONAL_CAVEAT = (
    "These results are based on observational causal inference (Bayesian Media Mix Modelling), "
    "not a randomised controlled experiment. They represent the model's best estimate of "
    "incremental impact given the available data."
)


# --- Confidence labels for the simple view ---

def _confidence_label(hdi_width_pct: float, beta_sd: float, beta_mean: float) -> str:
    """Map posterior uncertainty to a plain-language confidence label.

    Uses the relative width of the HDI and the coefficient of variation.
    """
    if beta_mean == 0:
        return "Low confidence — more data needed"
    cv = beta_sd / beta_mean if beta_mean > 0 else float("inf")
    if cv < 0.3 and hdi_width_pct < 40:
        return "High confidence"
    if cv < 0.6 and hdi_width_pct < 70:
        return "Moderate confidence"
    return "Low confidence — more data needed"


def _channel_display_name(channel: str) -> str:
    """Convert internal channel names to readable labels."""
    name = channel.replace("spend_", "")
    labels = {
        "google_search": "Google Search",
        "google_shopping": "Google Shopping",
        "google_pmax": "Google Performance Max",
        "google_youtube": "YouTube",
        "meta_feed": "Meta (Facebook Feed)",
        "meta_instagram": "Meta (Instagram)",
        "meta_stories": "Meta (Stories)",
    }
    return labels.get(name, name.replace("_", " ").title())


# --- Recommendation engine ---

def _generate_recommendations(
    results: ModelResults,
    trust: TrustScore,
) -> list[dict]:
    """Generate plain-language recommendations based on model results.

    Flags underperforming channels, recommends spend-scaling tests for
    wide intervals (collinearity), and surfaces data quality issues.
    """
    recs = []

    for cp in results.channel_posteriors:
        display = _channel_display_name(cp.channel)
        hdi_width = cp.contribution_hdi_97 - cp.contribution_hdi_3

        # Wide interval → recommend experiment
        if cp.beta_mean > 0:
            relative_width = (cp.beta_hdi_97 - cp.beta_hdi_3) / cp.beta_mean
        else:
            relative_width = float("inf")

        if relative_width > 1.5:
            recs.append({
                "channel": display,
                "type": "experiment_recommended",
                "message": (
                    f"The model has wide uncertainty for {display}. "
                    "Consider running a spend-scaling experiment to narrow the estimate."
                ),
                "priority": "high",
            })

        # Low contribution with significant spend
        if cp.contribution_pct < 5 and cp.beta_mean > 0:
            recs.append({
                "channel": display,
                "type": "underperforming",
                "message": (
                    f"{display} contributes only {cp.contribution_pct:.1f}% of modelled revenue. "
                    "Consider whether current spend levels are justified."
                ),
                "priority": "medium",
            })

    # Trust score based recommendations
    if trust.overall_tier == "Insufficient data":
        recs.insert(0, {
            "channel": "All",
            "type": "data_quality",
            "message": (
                "Data quality or model fit is insufficient for reliable results. "
                "Collect more data or run experiments before making budget decisions."
            ),
            "priority": "critical",
        })

    return recs


# --- Simple view ---

@dataclass
class SimpleChannelResult:
    channel: str
    display_name: str
    contribution_pct: float
    confidence: str


@dataclass
class SimpleView:
    channels: list[SimpleChannelResult]
    baseline_pct: float
    trust_tier: str
    trust_score: float
    recommendations: list[dict]
    caveat: str = OBSERVATIONAL_CAVEAT

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def build_simple_view(results: ModelResults, trust: TrustScore) -> SimpleView:
    """Build the simple view — plain language, no statistics visible."""
    channels = []
    for cp in results.channel_posteriors:
        hdi_width_pct = (
            (cp.contribution_hdi_97 - cp.contribution_hdi_3)
            / max(abs(cp.contribution_mean), 1) * 100
        )
        channels.append(SimpleChannelResult(
            channel=cp.channel,
            display_name=_channel_display_name(cp.channel),
            contribution_pct=round(cp.contribution_pct, 1),
            confidence=_confidence_label(hdi_width_pct, cp.beta_sd, cp.beta_mean),
        ))

    # Sort by contribution descending
    channels.sort(key=lambda c: c.contribution_pct, reverse=True)

    return SimpleView(
        channels=channels,
        baseline_pct=round(results.baseline_contribution_pct, 1),
        trust_tier=trust.overall_tier,
        trust_score=trust.overall_score,
        recommendations=_generate_recommendations(results, trust),
    )


# --- Intermediate view ---

@dataclass
class AdstockCurve:
    """Geometric decay curve data for a channel."""
    channel: str
    display_name: str
    alpha_mean: float
    alpha_hdi_3: float
    alpha_hdi_97: float
    half_life_days: float  # days for effect to halve
    decay_curve: list[float]  # normalised weights for lags 0..l_max


@dataclass
class SaturationCurve:
    """Saturation curve data for a channel."""
    channel: str
    display_name: str
    lam_mean: float
    lam_hdi_3: float
    lam_hdi_97: float
    curve_x: list[float]  # spend levels (0 to 2× max observed)
    curve_y: list[float]  # saturated response


@dataclass
class IntermediateChannelResult:
    channel: str
    display_name: str
    contribution_pct: float
    contribution_mean: float
    contribution_hdi_3: float
    contribution_hdi_97: float
    confidence: str
    in_platform_conversions_note: str


@dataclass
class IntermediateView:
    channels: list[IntermediateChannelResult]
    baseline_pct: float
    adstock_curves: list[AdstockCurve]
    saturation_curves: list[SaturationCurve]
    trust: dict
    recommendations: list[dict]
    caveat: str = OBSERVATIONAL_CAVEAT

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _build_adstock_curve(cp: ChannelPosterior, l_max: int = 8) -> AdstockCurve:
    """Generate the geometric decay curve from the adstock alpha posterior."""
    alpha = cp.adstock_alpha_mean
    # Geometric decay weights: alpha^t for t = 0..l_max
    decay = [alpha ** t for t in range(l_max + 1)]
    # Normalise so weights sum to 1
    total = sum(decay)
    decay_norm = [round(d / total, 4) if total > 0 else 0 for d in decay]

    # Half-life: solve alpha^t = 0.5 → t = log(0.5)/log(alpha)
    if 0 < alpha < 1:
        half_life = np.log(0.5) / np.log(alpha)
    else:
        half_life = 0.0

    return AdstockCurve(
        channel=cp.channel,
        display_name=_channel_display_name(cp.channel),
        alpha_mean=round(cp.adstock_alpha_mean, 4),
        alpha_hdi_3=round(cp.adstock_alpha_hdi_3, 4),
        alpha_hdi_97=round(cp.adstock_alpha_hdi_97, 4),
        half_life_days=round(half_life, 1),
        decay_curve=decay_norm,
    )


def _build_saturation_curve(
    cp: ChannelPosterior,
    max_spend: float,
    n_points: int = 50,
) -> SaturationCurve:
    """Generate saturation curve from the saturation lambda posterior.

    Uses LogisticSaturation formula: (1 - exp(-lam * x)) / (1 + exp(-lam * x))
    """
    lam = cp.saturation_lam_mean
    x_range = np.linspace(0, max_spend * 2, n_points)
    y_values = (1 - np.exp(-lam * x_range)) / (1 + np.exp(-lam * x_range))

    return SaturationCurve(
        channel=cp.channel,
        display_name=_channel_display_name(cp.channel),
        lam_mean=round(cp.saturation_lam_mean, 4),
        lam_hdi_3=round(cp.saturation_lam_hdi_3, 4),
        lam_hdi_97=round(cp.saturation_lam_hdi_97, 4),
        curve_x=[round(float(x), 2) for x in x_range],
        curve_y=[round(float(y), 4) for y in y_values],
    )


def build_intermediate_view(
    results: ModelResults,
    trust: TrustScore,
    channel_max_spends: dict[str, float] | None = None,
    l_max: int = 8,
) -> IntermediateView:
    """Build the intermediate view with confidence intervals, adstock/saturation curves."""
    channel_max_spends = channel_max_spends or {}

    channels = []
    adstock_curves = []
    saturation_curves = []

    for cp in results.channel_posteriors:
        hdi_width_pct = (
            (cp.contribution_hdi_97 - cp.contribution_hdi_3)
            / max(abs(cp.contribution_mean), 1) * 100
        )
        channels.append(IntermediateChannelResult(
            channel=cp.channel,
            display_name=_channel_display_name(cp.channel),
            contribution_pct=round(cp.contribution_pct, 1),
            contribution_mean=round(cp.contribution_mean, 2),
            contribution_hdi_3=round(cp.contribution_hdi_3, 2),
            contribution_hdi_97=round(cp.contribution_hdi_97, 2),
            confidence=_confidence_label(hdi_width_pct, cp.beta_sd, cp.beta_mean),
            in_platform_conversions_note=(
                "In-platform conversions are included as a model covariate only — "
                "they are not used as an outcome variable."
            ),
        ))

        adstock_curves.append(_build_adstock_curve(cp, l_max))

        max_spend = channel_max_spends.get(cp.channel, 1.0)
        saturation_curves.append(_build_saturation_curve(cp, max_spend))

    channels.sort(key=lambda c: c.contribution_pct, reverse=True)

    return IntermediateView(
        channels=channels,
        baseline_pct=round(results.baseline_contribution_pct, 1),
        adstock_curves=adstock_curves,
        saturation_curves=saturation_curves,
        trust=trust.to_dict(),
        recommendations=_generate_recommendations(results, trust),
    )


# --- Advanced view ---

@dataclass
class AdvancedChannelResult:
    """Full posterior distribution summaries for a channel — 94% HDI, not p-values."""
    channel: str
    display_name: str
    # Channel coefficient (beta)
    beta_mean: float
    beta_sd: float
    beta_hdi_3: float
    beta_hdi_97: float
    # Adstock decay
    adstock_alpha_mean: float
    adstock_alpha_sd: float
    adstock_alpha_hdi_3: float
    adstock_alpha_hdi_97: float
    # Saturation
    saturation_lam_mean: float
    saturation_lam_sd: float
    saturation_lam_hdi_3: float
    saturation_lam_hdi_97: float
    # Contribution
    contribution_mean: float
    contribution_pct: float
    contribution_hdi_3: float
    contribution_hdi_97: float


@dataclass
class AdvancedDiagnostics:
    """Model fit diagnostics with full uncertainty."""
    r_squared_mean: float
    r_squared_hdi_3: float
    r_squared_hdi_97: float
    mape_mean: float
    mape_hdi_3: float
    mape_hdi_97: float
    divergences: int
    divergence_pct: float
    warnings: list[str]


@dataclass
class AdvancedView:
    """Advanced output view — full posteriors, diagnostics, validation, experiments.

    CLAUDE.md spec:
    - Full posterior distribution summaries (mean, SD, 94% HDI — not p-values)
    - Model fit diagnostics: R², MAPE, posterior predictive check plots
    - Adstock parameter estimates with credible intervals
    - Data quality flags
    - CausalImpact experiment results alongside MMM posterior updates
    """
    channels: list[AdvancedChannelResult]
    baseline_contribution_pct: float
    diagnostics: AdvancedDiagnostics
    data_quality_flags: list[str]
    trust: dict
    validation: dict | None
    experiments: list[dict]
    recommendations: list[dict]
    caveat: str = OBSERVATIONAL_CAVEAT

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def build_advanced_view(
    results: ModelResults,
    trust: TrustScore,
    total_draws: int = 4000,
    validation: dict | None = None,
    experiments: list[dict] | None = None,
) -> AdvancedView:
    """Build the advanced view — full posterior summaries, diagnostics, validation, experiments.

    Parameters
    ----------
    results : ModelResults from the fitted MMM
    trust : TrustScore computed from data quality + model fit
    total_draws : total posterior draws (chains × draws) for divergence %
    validation : ValidationReport.to_dict() if validation was run
    experiments : list of SpendScalingResult.to_dict() or ProductExperimentResult.to_dict()
    """
    channels = []
    for cp in results.channel_posteriors:
        channels.append(AdvancedChannelResult(
            channel=cp.channel,
            display_name=_channel_display_name(cp.channel),
            beta_mean=round(cp.beta_mean, 4),
            beta_sd=round(cp.beta_sd, 4),
            beta_hdi_3=round(cp.beta_hdi_3, 4),
            beta_hdi_97=round(cp.beta_hdi_97, 4),
            adstock_alpha_mean=round(cp.adstock_alpha_mean, 4),
            adstock_alpha_sd=round(cp.adstock_alpha_sd, 4),
            adstock_alpha_hdi_3=round(cp.adstock_alpha_hdi_3, 4),
            adstock_alpha_hdi_97=round(cp.adstock_alpha_hdi_97, 4),
            saturation_lam_mean=round(cp.saturation_lam_mean, 4),
            saturation_lam_sd=round(cp.saturation_lam_sd, 4),
            saturation_lam_hdi_3=round(cp.saturation_lam_hdi_3, 4),
            saturation_lam_hdi_97=round(cp.saturation_lam_hdi_97, 4),
            contribution_mean=round(cp.contribution_mean, 2),
            contribution_pct=round(cp.contribution_pct, 1),
            contribution_hdi_3=round(cp.contribution_hdi_3, 2),
            contribution_hdi_97=round(cp.contribution_hdi_97, 2),
        ))

    channels.sort(key=lambda c: c.contribution_pct, reverse=True)

    diag = results.diagnostics
    divergence_pct = (diag.divergences / total_draws * 100) if total_draws > 0 else 0.0

    diagnostics = AdvancedDiagnostics(
        r_squared_mean=round(diag.r_squared_mean, 4),
        r_squared_hdi_3=round(diag.r_squared_hdi_3, 4),
        r_squared_hdi_97=round(diag.r_squared_hdi_97, 4),
        mape_mean=round(diag.mape_mean, 2),
        mape_hdi_3=round(diag.mape_hdi_3, 2),
        mape_hdi_97=round(diag.mape_hdi_97, 2),
        divergences=diag.divergences,
        divergence_pct=round(divergence_pct, 2),
        warnings=diag.warnings,
    )

    return AdvancedView(
        channels=channels,
        baseline_contribution_pct=round(results.baseline_contribution_pct, 1),
        diagnostics=diagnostics,
        data_quality_flags=trust.flags,
        trust=trust.to_dict(),
        validation=validation,
        experiments=experiments or [],
        recommendations=_generate_recommendations(results, trust),
    )
