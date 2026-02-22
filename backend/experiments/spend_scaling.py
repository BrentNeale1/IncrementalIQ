"""Spend-scaling experiments: primary validation path for MMM.

CLAUDE.md spec:
- User tags a date range + campaign/channel as a deliberate spend change
- CausalImpact runs on that window using other channels as controls
- Output: incremental impact estimate with confidence interval
- Result is used to calibrate MMM posterior for that channel

The causalimpact library (Python port of Google's R package) fits a
Bayesian structural time series model to estimate the counterfactual
(what would have happened without the spend change).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date
import numpy as np
import pandas as pd
from causalimpact import CausalImpact


@dataclass
class SpendScalingResult:
    """Output from a spend-scaling experiment."""
    channel: str
    pre_period_start: str
    pre_period_end: str
    post_period_start: str
    post_period_end: str
    # Incremental impact
    estimated_impact: float          # cumulative causal effect
    impact_ci_lower: float           # lower bound of CI
    impact_ci_upper: float           # upper bound of CI
    relative_effect: float           # % change attributed to the intervention
    relative_effect_ci_lower: float
    relative_effect_ci_upper: float
    # Probability
    p_value: float
    significant: bool
    # MMM calibration
    mmm_prediction: float | None = None  # MMM's predicted impact (if available)
    discrepancy_pct: float | None = None  # |experiment - MMM| / experiment
    recalibration_recommended: bool = False
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _build_causalimpact_data(
    daily_data: pd.DataFrame,
    target_channel: str,
    outcome_column: str,
    channel_columns: list[str],
) -> pd.DataFrame:
    """Build the DataFrame for CausalImpact.

    Column 0 = outcome (response variable)
    Columns 1+ = control series (other channels' spend, used as predictors)
    """
    controls = [c for c in channel_columns if c != target_channel]
    if not controls:
        raise ValueError(
            "At least one other channel is required as a control. "
            "Cannot run spend-scaling experiment with only one channel."
        )

    cols = [outcome_column] + controls
    ci_df = daily_data[cols].copy()
    ci_df.index = pd.to_datetime(daily_data["date"])
    return ci_df


def run_spend_scaling_experiment(
    daily_data: pd.DataFrame,
    target_channel: str,
    pre_period: tuple[str, str],
    post_period: tuple[str, str],
    outcome_column: str = "revenue",
    channel_columns: list[str] | None = None,
    alpha: float = 0.05,
    mmm_predicted_impact: float | None = None,
) -> SpendScalingResult:
    """Run a spend-scaling experiment using CausalImpact.

    Parameters
    ----------
    daily_data : DataFrame with 'date', outcome, and channel spend columns.
                 Must be sorted by date, one row per day.
    target_channel : the channel that had its spend deliberately changed
    pre_period : (start_date, end_date) strings for the pre-intervention period
    post_period : (start_date, end_date) strings for the post-intervention period
    outcome_column : name of the outcome column (default "revenue")
    channel_columns : list of all channel spend columns (to use others as controls)
    alpha : significance level for confidence intervals (default 0.05)
    mmm_predicted_impact : optional MMM prediction for this channel's impact,
                          used to compute discrepancy

    Returns
    -------
    SpendScalingResult with impact estimate, confidence interval, and calibration data
    """
    if channel_columns is None:
        channel_columns = [c for c in daily_data.columns if c.startswith("spend_")]

    if target_channel not in channel_columns:
        raise ValueError(f"Target channel '{target_channel}' not found in channel_columns")

    # Build the CausalImpact input data
    ci_data = _build_causalimpact_data(
        daily_data, target_channel, outcome_column, channel_columns,
    )

    # Run CausalImpact
    ci = CausalImpact(
        data=ci_data,
        pre_period=[pre_period[0], pre_period[1]],
        post_period=[post_period[0], post_period[1]],
        alpha=alpha,
    )

    # Extract results from the summary
    summary_df = ci.summary_data
    # summary_data columns: average, cumulative
    # Rows: actual, predicted, predicted_lower, predicted_upper,
    #        abs_effect, abs_effect_lower, abs_effect_upper,
    #        rel_effect, rel_effect_lower, rel_effect_upper

    cumulative = summary_df["cumulative"]
    estimated_impact = float(cumulative.loc["abs_effect"])
    impact_ci_lower = float(cumulative.loc["abs_effect_lower"])
    impact_ci_upper = float(cumulative.loc["abs_effect_upper"])
    relative_effect = float(cumulative.loc["rel_effect"])
    rel_ci_lower = float(cumulative.loc["rel_effect_lower"])
    rel_ci_upper = float(cumulative.loc["rel_effect_upper"])

    # p-value from CausalImpact
    p_value = float(ci.p_value)
    significant = p_value < alpha

    # MMM calibration check (CLAUDE.md: discrepancy > 30% → recalibration)
    discrepancy_pct = None
    recalibration = False
    if mmm_predicted_impact is not None and abs(estimated_impact) > 0:
        discrepancy_pct = abs(mmm_predicted_impact - estimated_impact) / abs(estimated_impact) * 100
        recalibration = discrepancy_pct > 30

    # Generate summary text
    direction = "increase" if estimated_impact > 0 else "decrease"
    summary = (
        f"The spend change on {target_channel} caused an estimated "
        f"{direction} of {abs(estimated_impact):,.0f} in {outcome_column} "
        f"(CI: [{impact_ci_lower:,.0f}, {impact_ci_upper:,.0f}], "
        f"p={p_value:.3f})."
    )

    return SpendScalingResult(
        channel=target_channel,
        pre_period_start=pre_period[0],
        pre_period_end=pre_period[1],
        post_period_start=post_period[0],
        post_period_end=post_period[1],
        estimated_impact=round(estimated_impact, 2),
        impact_ci_lower=round(impact_ci_lower, 2),
        impact_ci_upper=round(impact_ci_upper, 2),
        relative_effect=round(relative_effect, 4),
        relative_effect_ci_lower=round(rel_ci_lower, 4),
        relative_effect_ci_upper=round(rel_ci_upper, 4),
        p_value=round(p_value, 4),
        significant=significant,
        mmm_prediction=round(mmm_predicted_impact, 2) if mmm_predicted_impact is not None else None,
        discrepancy_pct=round(discrepancy_pct, 1) if discrepancy_pct is not None else None,
        recalibration_recommended=recalibration,
        summary=summary,
    )
