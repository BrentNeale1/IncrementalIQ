"""Trust Score — displayed on every report.

Composite of three sub-scores:
1. Data quality score (history length, gap frequency, spend variance)
2. Model fit score (holdout MAPE, posterior predictive check result)
3. Calibration score (experiment vs. MMM prediction agreement — Phase 3, defaults to neutral)

Three tiers:
- "Model results are reliable" — good data, strong fit, calibration within tolerance
- "Use with caution" — moderate data or fit issues, directionally useful
- "Insufficient data" — history too short, too gappy, or fit too poor
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json


TIER_RELIABLE = "Model results are reliable"
TIER_CAUTION = "Use with caution"
TIER_INSUFFICIENT = "Insufficient data"


@dataclass
class TrustScore:
    overall_tier: str
    overall_score: float  # 0.0–1.0
    data_quality_score: float
    model_fit_score: float
    calibration_score: float  # defaults to 0.5 (neutral) until Phase 3
    flags: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def compute_data_quality_score(
    history_status: str,
    gap_count: int,
    total_days: int,
    low_variance_channels: list[str],
    total_channels: int,
) -> tuple[float, list[str]]:
    """Score data quality from 0.0 (terrible) to 1.0 (excellent).

    Returns (score, list of flags).
    """
    score = 1.0
    flags = []

    # History length
    if history_status == "rejected":
        score -= 0.5
        flags.append("Data history below 9 months — model cannot be run reliably.")
    elif history_status == "caution":
        score -= 0.2
        flags.append("Data history below 18 months — results should be interpreted with caution.")

    # Gaps
    if total_days > 0:
        gap_ratio = gap_count / total_days
        if gap_ratio > 0.1:
            score -= 0.2
            flags.append(f"High gap frequency: {gap_count} days missing out of {total_days}.")
        elif gap_ratio > 0.03:
            score -= 0.1
            flags.append(f"Moderate gap frequency: {gap_count} days missing out of {total_days}.")

    # Low variance channels
    if total_channels > 0:
        low_var_ratio = len(low_variance_channels) / total_channels
        if low_var_ratio > 0.5:
            score -= 0.2
            flags.append(
                f"Most channels ({len(low_variance_channels)}/{total_channels}) have low spend variance. "
                "Adstock decay rates may not be reliably estimated."
            )
        elif low_variance_channels:
            score -= 0.1
            flags.append(
                f"{len(low_variance_channels)} channel(s) with low spend variance: "
                f"{', '.join(low_variance_channels)}."
            )

    return max(0.0, score), flags


def compute_model_fit_score(
    mape_mean: float,
    r_squared_mean: float,
    divergences: int,
    total_draws: int,
    holdout_mape: float | None = None,
    holdout_coverage: float | None = None,
) -> tuple[float, list[str]]:
    """Score model fit from 0.0 to 1.0.

    Returns (score, list of flags).
    """
    score = 1.0
    flags = []

    # In-sample MAPE
    if mape_mean > 25:
        score -= 0.3
        flags.append(f"In-sample MAPE is {mape_mean:.1f}% — poor model fit.")
    elif mape_mean > 15:
        score -= 0.15
        flags.append(f"In-sample MAPE is {mape_mean:.1f}% — moderate model fit (threshold: 15%).")

    # R²
    if r_squared_mean < 0.5:
        score -= 0.2
        flags.append(f"R² is {r_squared_mean:.2f} — the model explains less than half the variance.")
    elif r_squared_mean < 0.7:
        score -= 0.1
        flags.append(f"R² is {r_squared_mean:.2f} — moderate explanatory power.")

    # Divergences
    if total_draws > 0:
        div_rate = divergences / total_draws
        if div_rate > 0.1:
            score -= 0.2
            flags.append(
                f"{divergences} divergent transitions ({div_rate:.0%} of samples). "
                "Posterior estimates may be unreliable."
            )
        elif divergences > 0:
            score -= 0.05
            flags.append(f"{divergences} divergent transition(s) detected.")

    # Holdout MAPE (from Phase 5 validation)
    if holdout_mape is not None:
        if holdout_mape > 25:
            score -= 0.25
            flags.append(f"Holdout MAPE is {holdout_mape:.1f}% — poor out-of-sample fit.")
        elif holdout_mape > 15:
            score -= 0.1
            flags.append(f"Holdout MAPE is {holdout_mape:.1f}% (threshold: 15%).")

    # Holdout coverage
    if holdout_coverage is not None and holdout_coverage < 0.80:
        score -= 0.1
        flags.append(
            f"Holdout coverage is {holdout_coverage:.0%} (threshold: 80%). "
            "Posterior predictive intervals may be too narrow."
        )

    return max(0.0, score), flags


def compute_calibration_score(
    experiment_discrepancy: float | None = None,
) -> tuple[float, list[str]]:
    """Score calibration from experiment results. Defaults to neutral (0.5) until Phase 3.

    Returns (score, list of flags).
    """
    if experiment_discrepancy is None:
        return 0.5, ["No experiment calibration data available yet."]

    flags = []
    if experiment_discrepancy > 0.30:
        score = 0.2
        flags.append(
            f"Experiment vs. MMM discrepancy is {experiment_discrepancy:.0%} (threshold: 30%). "
            "Recalibration recommended."
        )
    elif experiment_discrepancy > 0.15:
        score = 0.6
        flags.append(
            f"Experiment vs. MMM discrepancy is {experiment_discrepancy:.0%} — moderate agreement."
        )
    else:
        score = 1.0
        flags.append(
            f"Experiment vs. MMM discrepancy is {experiment_discrepancy:.0%} — good agreement."
        )
    return score, flags


def compute_trust_score(
    # Data quality inputs
    history_status: str,
    gap_count: int,
    total_days: int,
    low_variance_channels: list[str],
    total_channels: int,
    # Model fit inputs
    mape_mean: float,
    r_squared_mean: float,
    divergences: int,
    total_draws: int,
    holdout_mape: float | None = None,
    holdout_coverage: float | None = None,
    # Calibration inputs (Phase 3)
    experiment_discrepancy: float | None = None,
) -> TrustScore:
    """Compute the composite trust score."""
    dq_score, dq_flags = compute_data_quality_score(
        history_status, gap_count, total_days, low_variance_channels, total_channels,
    )
    mf_score, mf_flags = compute_model_fit_score(
        mape_mean, r_squared_mean, divergences, total_draws,
        holdout_mape, holdout_coverage,
    )
    cal_score, cal_flags = compute_calibration_score(experiment_discrepancy)

    # Weighted composite: data quality 35%, model fit 45%, calibration 20%
    overall = 0.35 * dq_score + 0.45 * mf_score + 0.20 * cal_score

    # Determine tier
    # "rejected" history status always → insufficient, regardless of scores
    if history_status == "rejected":
        tier = TIER_INSUFFICIENT
    elif overall >= 0.7 and dq_score >= 0.6 and mf_score >= 0.5:
        tier = TIER_RELIABLE
    elif overall >= 0.35 and dq_score >= 0.3:
        tier = TIER_CAUTION
    else:
        tier = TIER_INSUFFICIENT

    return TrustScore(
        overall_tier=tier,
        overall_score=round(overall, 3),
        data_quality_score=round(dq_score, 3),
        model_fit_score=round(mf_score, 3),
        calibration_score=round(cal_score, 3),
        flags=dq_flags + mf_flags + cal_flags,
    )
