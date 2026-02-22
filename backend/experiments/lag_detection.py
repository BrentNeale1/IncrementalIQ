"""Lag detection: cross-correlation between channel spend and outcome.

Uses statsmodels.tsa.stattools.ccf to detect the optimal lag (in days)
between each channel's spend and the revenue/orders outcome.

This is a preprocessing step — the detected lags inform adstock configuration
and help users understand the delay between spend and impact.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import ccf


DEFAULT_MAX_LAG = 21  # max days to search for lag
SIGNIFICANCE_THRESHOLD = 0.1  # min absolute correlation to consider meaningful


@dataclass
class ChannelLag:
    channel: str
    optimal_lag_days: int
    peak_correlation: float
    significant: bool
    correlations: list[float]  # ccf values from lag 0 to max_lag


@dataclass
class LagReport:
    channel_lags: list[ChannelLag]
    max_lag_searched: int

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def detect_lag(
    spend: np.ndarray,
    outcome: np.ndarray,
    max_lag: int = DEFAULT_MAX_LAG,
) -> tuple[int, float, list[float]]:
    """Detect optimal lag between a spend series and outcome series.

    Parameters
    ----------
    spend : 1D array of daily spend values
    outcome : 1D array of daily outcome values (revenue/orders)
    max_lag : maximum number of days to search

    Returns
    -------
    (optimal_lag, peak_correlation, correlations_list)
    """
    n = len(spend)
    if n < max_lag + 10:
        max_lag = max(1, n // 3)

    # Compute cross-correlation function
    # ccf(x, y) at lag k measures correlation between x[t] and y[t+k]
    # We want: how does spend at time t correlate with outcome at time t+k?
    correlations = ccf(spend, outcome, nlags=max_lag, adjusted=False, alpha=None)

    # correlations[0] is lag 0, correlations[k] is lag k
    corr_list = correlations[:max_lag + 1].tolist()

    # Find the lag with the highest absolute correlation
    abs_corr = np.abs(correlations[:max_lag + 1])
    optimal_lag = int(np.argmax(abs_corr))
    peak_corr = float(correlations[optimal_lag])

    return optimal_lag, peak_corr, corr_list


def detect_channel_lags(
    df: pd.DataFrame,
    channel_columns: list[str],
    outcome_column: str = "revenue",
    max_lag: int = DEFAULT_MAX_LAG,
) -> LagReport:
    """Detect lags for all channels in a prepared dataset.

    Parameters
    ----------
    df : DataFrame with date-indexed rows containing channel spend and outcome columns
    channel_columns : list of spend column names (e.g. ["spend_google_search", "spend_meta_feed"])
    outcome_column : name of the outcome column in the same DataFrame
    max_lag : maximum lag days to search

    Returns
    -------
    LagReport with per-channel lag estimates
    """
    if outcome_column not in df.columns:
        raise ValueError(f"Outcome column '{outcome_column}' not found in DataFrame")

    outcome = df[outcome_column].values

    channel_lags = []
    for ch_col in channel_columns:
        if ch_col not in df.columns:
            continue

        spend = df[ch_col].values

        # Skip channels with zero or near-zero variance
        if np.std(spend) < 1e-8:
            channel_lags.append(ChannelLag(
                channel=ch_col,
                optimal_lag_days=0,
                peak_correlation=0.0,
                significant=False,
                correlations=[0.0] * (max_lag + 1),
            ))
            continue

        optimal_lag, peak_corr, corr_list = detect_lag(spend, outcome, max_lag)

        channel_lags.append(ChannelLag(
            channel=ch_col,
            optimal_lag_days=optimal_lag,
            peak_correlation=round(peak_corr, 4),
            significant=abs(peak_corr) > SIGNIFICANCE_THRESHOLD,
            correlations=[round(c, 4) for c in corr_list],
        ))

    return LagReport(
        channel_lags=channel_lags,
        max_lag_searched=max_lag,
    )


def detect_lags_from_prepared_data(
    X: pd.DataFrame,
    y: pd.Series,
    channel_columns: list[str],
    max_lag: int = DEFAULT_MAX_LAG,
) -> LagReport:
    """Convenience wrapper: detect lags from PreparedData's X and y."""
    combined = X.copy()
    combined[y.name] = y.values
    return detect_channel_lags(combined, channel_columns, y.name, max_lag)
