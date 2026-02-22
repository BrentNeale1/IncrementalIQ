import json
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
import pandas as pd
import numpy as np


@dataclass
class ChannelQuality:
    channel: str
    row_count: int
    total_spend: float
    spend_variance_cv: float  # coefficient of variation
    low_variance: bool
    gap_days: int
    spike_count: int


@dataclass
class QualityResult:
    history_days: int
    history_months: float
    history_status: str  # "sufficient" | "caution" | "rejected"
    date_range_start: str
    date_range_end: str
    total_rows: int
    channels: list[ChannelQuality] = field(default_factory=list)
    gap_count: int = 0
    spike_count: int = 0
    low_variance_channels: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# --- thresholds from CLAUDE.md ---
REJECT_MONTHS = 9
CAUTION_MONTHS = 18
LOW_CV_THRESHOLD = 0.15  # spend CV below this = low variance
SPIKE_Z_THRESHOLD = 3.0  # z-score for spike detection


def assess_history_length(days: int) -> str:
    months = days / 30.44
    if months < REJECT_MONTHS:
        return "rejected"
    if months < CAUTION_MONTHS:
        return "caution"
    return "sufficient"


def detect_gaps(dates: pd.Series) -> list[dict]:
    """Find date gaps in the full dataset (missing days with no rows at all)."""
    all_dates = pd.date_range(dates.min(), dates.max(), freq="D").date
    present = set(dates.unique())
    missing = sorted(set(all_dates) - present)
    gaps = []
    if not missing:
        return gaps

    # Group consecutive missing days into ranges
    start = missing[0]
    prev = missing[0]
    for d in missing[1:]:
        if d - prev > timedelta(days=1):
            gaps.append({"start": str(start), "end": str(prev), "days": (prev - start).days + 1})
            start = d
        prev = d
    gaps.append({"start": str(start), "end": str(prev), "days": (prev - start).days + 1})
    return gaps


def detect_spikes(series: pd.Series, label: str) -> list[dict]:
    """Detect values more than SPIKE_Z_THRESHOLD standard deviations from mean."""
    if series.std() == 0:
        return []
    z = (series - series.mean()) / series.std()
    spike_mask = z.abs() > SPIKE_Z_THRESHOLD
    spikes = []
    for idx in series.index[spike_mask]:
        spikes.append({
            "metric": label,
            "index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            "value": float(series.loc[idx]),
            "z_score": float(z.loc[idx]),
        })
    return spikes


def generate_quality_report(df: pd.DataFrame) -> QualityResult:
    """Produce a full data quality assessment.

    Expects a DataFrame with 'date' column already as Python date objects
    and all schema columns present.
    """
    dates = pd.Series(df["date"])
    date_min = dates.min()
    date_max = dates.max()
    history_days = (date_max - date_min).days + 1
    history_months = history_days / 30.44
    history_status = assess_history_length(history_days)

    # Gap detection across the full dataset
    gaps = detect_gaps(dates)
    total_gap_days = sum(g["days"] for g in gaps)

    # Per-channel quality
    channels_quality: list[ChannelQuality] = []
    all_spikes: list[dict] = []
    low_variance_channels: list[str] = []

    for ch, group in df.groupby("channel"):
        spend = group["spend"]
        cv = float(spend.std() / spend.mean()) if spend.mean() > 0 else 0.0
        low_var = cv < LOW_CV_THRESHOLD

        ch_dates = pd.Series(group["date"])
        ch_gaps = detect_gaps(ch_dates)
        ch_gap_days = sum(g["days"] for g in ch_gaps)

        ch_spikes = detect_spikes(spend.reset_index(drop=True), f"{ch}_spend")
        rev_spikes = detect_spikes(
            group["revenue"].reset_index(drop=True), f"{ch}_revenue"
        )

        cq = ChannelQuality(
            channel=str(ch),
            row_count=len(group),
            total_spend=float(spend.sum()),
            spend_variance_cv=round(cv, 4),
            low_variance=low_var,
            gap_days=ch_gap_days,
            spike_count=len(ch_spikes) + len(rev_spikes),
        )
        channels_quality.append(cq)
        all_spikes.extend(ch_spikes)
        all_spikes.extend(rev_spikes)

        if low_var:
            low_variance_channels.append(str(ch))

    # Assemble warnings
    warnings: list[str] = []
    if history_status == "rejected":
        warnings.append(
            f"Data covers only {history_months:.1f} months ({history_days} days). "
            f"At least 9 months required to run the model."
        )
    elif history_status == "caution":
        warnings.append(
            f"Data covers {history_months:.1f} months ({history_days} days). "
            f"18+ months recommended for reliable results."
        )

    if total_gap_days > 0:
        warnings.append(
            f"{total_gap_days} total days with missing data across {len(gaps)} gap(s)."
        )

    for ch in low_variance_channels:
        warnings.append(
            f"Channel '{ch}' has low spend variance (CV < {LOW_CV_THRESHOLD}). "
            f"Adstock decay rates may not be reliably estimated."
        )

    if all_spikes:
        warnings.append(f"{len(all_spikes)} statistical spike(s) detected in spend/revenue.")

    return QualityResult(
        history_days=history_days,
        history_months=round(history_months, 1),
        history_status=history_status,
        date_range_start=str(date_min),
        date_range_end=str(date_max),
        total_rows=len(df),
        channels=channels_quality,
        gap_count=total_gap_days,
        spike_count=len(all_spikes),
        low_variance_channels=low_variance_channels,
        warnings=warnings,
    )
