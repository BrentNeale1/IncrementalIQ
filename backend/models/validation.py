"""Phase 5: Internal validation — holdout testing, posterior predictive checks, sensitivity analysis.

CLAUDE.md thresholds:
- MAPE > 15% on holdout → surface warning
- Holdout coverage < 80% → surface warning
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
import arviz as az
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM
from pymc_extras.prior import Prior

from backend.models.data_prep import PreparedData
from backend.models.mmm import ModelConfig, _build_model_config, _hdi_numpy

logger = logging.getLogger(__name__)


# --- Holdout validation ---

@dataclass
class HoldoutResult:
    """Results from time-based holdout validation."""
    train_size: int
    test_size: int
    holdout_fraction: float
    mape_mean: float
    mape_hdi_3: float
    mape_hdi_97: float
    r_squared_mean: float
    r_squared_hdi_3: float
    r_squared_hdi_97: float
    coverage: float  # fraction of test observations within 94% HDI
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def run_holdout_validation(
    data: PreparedData,
    config: ModelConfig | None = None,
    holdout_fraction: float = 0.2,
) -> HoldoutResult:
    """Train on first (1-holdout_fraction) of data, test on remainder.

    Returns holdout metrics (MAPE, R², coverage) with uncertainty.
    """
    config = config or ModelConfig()
    n = len(data.X)
    split_idx = int(n * (1 - holdout_fraction))

    if split_idx < 30:
        raise ValueError(
            f"Training set too small ({split_idx} days). "
            "Need at least 30 days for holdout validation."
        )

    X_train = data.X.iloc[:split_idx].reset_index(drop=True)
    y_train = data.y.iloc[:split_idx].reset_index(drop=True)
    X_test = data.X.iloc[split_idx:].reset_index(drop=True)
    y_test = data.y.iloc[split_idx:].values

    # Build and fit on training data
    train_data = PreparedData(
        X=X_train, y=y_train,
        date_column=data.date_column,
        channel_columns=data.channel_columns,
        control_columns=data.control_columns,
        target_variable=data.target_variable,
        daily_rows=len(X_train),
    )

    model_config = _build_model_config(train_data)
    mmm = MMM(
        date_column=data.date_column,
        channel_columns=data.channel_columns,
        target_column=data.target_variable,
        control_columns=data.control_columns,
        adstock=GeometricAdstock(l_max=config.adstock_l_max),
        saturation=LogisticSaturation(),
        model_config=model_config,
        sampler_config={"progressbar": True},
        yearly_seasonality=config.yearly_seasonality,
    )

    logger.info("Holdout validation: fitting on %d days, testing on %d days", split_idx, n - split_idx)
    mmm.fit(
        X=X_train, y=y_train,
        chains=config.chains, tune=config.tune, draws=config.draws,
        target_accept=config.target_accept, random_seed=config.random_seed,
    )

    # Predict on holdout
    pp = mmm.sample_posterior_predictive(X_test, extend_idata=False, combined=True)

    # Extract predictions
    import xarray as xr
    if isinstance(pp, (xr.DataArray, xr.Dataset)):
        if isinstance(pp, xr.Dataset):
            var_name = data.target_variable if data.target_variable in pp else list(pp.data_vars)[0]
            pred_samples = pp[var_name].values
        else:
            pred_samples = pp.values
    else:
        pred_samples = pp.posterior_predictive[data.target_variable].values

    # Ensure (samples, obs) orientation
    n_test = len(y_test)
    if pred_samples.ndim == 1:
        pred_samples = pred_samples.reshape(1, -1)
    elif pred_samples.ndim == 3:
        pred_samples = pred_samples.reshape(-1, pred_samples.shape[-1])
    if pred_samples.shape[0] == n_test and pred_samples.shape[1] != n_test:
        pred_samples = pred_samples.T

    # MAPE on holdout
    nonzero = y_test != 0
    if nonzero.any():
        ape = np.abs((y_test[nonzero] - pred_samples[:, nonzero]) / y_test[nonzero])
        mape_samples = ape.mean(axis=1) * 100
        mape_mean = float(np.mean(mape_samples))
        mape_hdi = _hdi_numpy(mape_samples, 0.94)
    else:
        mape_mean = 0.0
        mape_hdi = (0.0, 0.0)

    # R² on holdout
    ss_res = ((y_test - pred_samples) ** 2).sum(axis=1)
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    r2_samples = 1.0 - ss_res / ss_tot
    r2_mean = float(np.mean(r2_samples))
    r2_hdi = _hdi_numpy(r2_samples, 0.94)

    # Coverage: fraction of test obs within 94% posterior predictive interval
    pred_low = np.percentile(pred_samples, 3, axis=0)
    pred_high = np.percentile(pred_samples, 97, axis=0)
    within = ((y_test >= pred_low) & (y_test <= pred_high)).mean()
    coverage = float(within)

    # Warnings per CLAUDE.md thresholds
    warnings = []
    if mape_mean > 15:
        warnings.append(f"Holdout MAPE is {mape_mean:.1f}% (threshold: 15%).")
    if coverage < 0.80:
        warnings.append(
            f"Holdout coverage is {coverage:.0%} (threshold: 80%). "
            "Posterior predictive intervals may be miscalibrated."
        )

    return HoldoutResult(
        train_size=split_idx,
        test_size=n - split_idx,
        holdout_fraction=holdout_fraction,
        mape_mean=round(mape_mean, 2),
        mape_hdi_3=round(mape_hdi[0], 2),
        mape_hdi_97=round(mape_hdi[1], 2),
        r_squared_mean=round(r2_mean, 4),
        r_squared_hdi_3=round(r2_hdi[0], 4),
        r_squared_hdi_97=round(r2_hdi[1], 4),
        coverage=round(coverage, 4),
        warnings=warnings,
    )


# --- Posterior predictive check ---

@dataclass
class PosteriorPredictiveCheck:
    """Results from posterior predictive check on training data."""
    mean_residual: float
    residual_std: float
    coverage: float  # fraction of training obs within 94% posterior predictive
    max_abs_residual: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def run_posterior_predictive_check(
    mmm: MMM,
    data: PreparedData,
) -> PosteriorPredictiveCheck:
    """Assess how well the fitted model reproduces the training data.

    Checks calibration of uncertainty by computing coverage of posterior
    predictive intervals on the training set.
    """
    import xarray as xr
    pp = mmm.sample_posterior_predictive(data.X, extend_idata=False, combined=True)

    if isinstance(pp, (xr.DataArray, xr.Dataset)):
        if isinstance(pp, xr.Dataset):
            var_name = data.target_variable if data.target_variable in pp else list(pp.data_vars)[0]
            pred_samples = pp[var_name].values
        else:
            pred_samples = pp.values
    else:
        pred_samples = pp.posterior_predictive[data.target_variable].values

    n_obs = len(data.y)
    if pred_samples.ndim == 1:
        pred_samples = pred_samples.reshape(1, -1)
    elif pred_samples.ndim == 3:
        pred_samples = pred_samples.reshape(-1, pred_samples.shape[-1])
    if pred_samples.shape[0] == n_obs and pred_samples.shape[1] != n_obs:
        pred_samples = pred_samples.T

    y_true = data.y.values
    pred_mean = pred_samples.mean(axis=0)
    residuals = y_true - pred_mean

    # Coverage
    pred_low = np.percentile(pred_samples, 3, axis=0)
    pred_high = np.percentile(pred_samples, 97, axis=0)
    within = ((y_true >= pred_low) & (y_true <= pred_high)).mean()

    warnings = []
    if within < 0.80:
        warnings.append(
            f"Posterior predictive coverage is {within:.0%} (expected ~94%). "
            "Model may be underestimating uncertainty."
        )

    return PosteriorPredictiveCheck(
        mean_residual=round(float(residuals.mean()), 2),
        residual_std=round(float(residuals.std()), 2),
        coverage=round(float(within), 4),
        max_abs_residual=round(float(np.abs(residuals).max()), 2),
        warnings=warnings,
    )


# --- Sensitivity analysis ---

@dataclass
class SensitivityResult:
    """Results from prior sensitivity analysis."""
    parameter_varied: str
    original_value: str
    varied_value: str
    channel: str
    original_beta_mean: float
    varied_beta_mean: float
    change_pct: float
    stable: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SensitivityReport:
    results: list[SensitivityResult]
    overall_stable: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def run_sensitivity_analysis(
    data: PreparedData,
    base_results: dict[str, float],  # channel_name → beta_mean from original fit
    config: ModelConfig | None = None,
) -> SensitivityReport:
    """Test sensitivity of channel coefficients to prior specification.

    Refits the model with wider priors on saturation_beta and checks
    whether coefficients change substantially.
    """
    config = config or ModelConfig()
    model_config = _build_model_config(data)

    # Vary: double the sigma on channel coefficients
    original_sigma = model_config["saturation_beta"].parameters.get("sigma", 2)
    if isinstance(original_sigma, np.ndarray):
        wide_sigma = original_sigma * 2
        orig_str = f"HalfNormal(sigma={np.round(original_sigma, 2).tolist()})"
        wide_str = f"HalfNormal(sigma={np.round(wide_sigma, 2).tolist()})"
    else:
        wide_sigma = original_sigma * 2
        orig_str = f"HalfNormal(sigma={original_sigma})"
        wide_str = f"HalfNormal(sigma={wide_sigma})"

    model_config_wide = model_config.copy()
    model_config_wide["saturation_beta"] = Prior("HalfNormal", sigma=wide_sigma, dims="channel")

    mmm_wide = MMM(
        date_column=data.date_column,
        channel_columns=data.channel_columns,
        target_column=data.target_variable,
        control_columns=data.control_columns,
        adstock=GeometricAdstock(l_max=config.adstock_l_max),
        saturation=LogisticSaturation(),
        model_config=model_config_wide,
        sampler_config={"progressbar": True},
        yearly_seasonality=config.yearly_seasonality,
    )

    logger.info("Sensitivity analysis: refitting with wider priors")
    mmm_wide.fit(
        X=data.X, y=data.y,
        chains=config.chains, tune=config.tune, draws=config.draws,
        target_accept=config.target_accept, random_seed=config.random_seed + 1,
    )

    # Extract varied betas
    summary_wide = az.summary(
        mmm_wide.idata,
        var_names=["saturation_beta"],
        hdi_prob=0.94,
    )

    results = []
    warnings = []
    all_stable = True

    channel_names = [
        str(c.values) if hasattr(c, 'values') else str(c)
        for c in mmm_wide.idata.posterior.coords.get("channel", [])
    ]

    for ch_name in channel_names:
        key = f"saturation_beta[{ch_name}]"
        varied_mean = float(summary_wide.loc[key, "mean"]) if key in summary_wide.index else 0.0
        original_mean = base_results.get(ch_name, 0.0)

        if original_mean > 0:
            change_pct = abs(varied_mean - original_mean) / original_mean * 100
        else:
            change_pct = 0.0 if varied_mean == 0 else 100.0

        stable = change_pct < 30  # <30% change = stable
        if not stable:
            all_stable = False

        results.append(SensitivityResult(
            parameter_varied="saturation_beta",
            original_value=orig_str,
            varied_value=wide_str,
            channel=ch_name,
            original_beta_mean=round(original_mean, 4),
            varied_beta_mean=round(varied_mean, 4),
            change_pct=round(change_pct, 1),
            stable=stable,
        ))

    if not all_stable:
        unstable = [r.channel for r in results if not r.stable]
        warnings.append(
            f"Channel coefficients are sensitive to prior specification for: "
            f"{', '.join(unstable)}. Results for these channels should be interpreted cautiously."
        )

    return SensitivityReport(
        results=results,
        overall_stable=all_stable,
        warnings=warnings,
    )


# --- Combined validation report ---

@dataclass
class ValidationReport:
    holdout: HoldoutResult | None = None
    posterior_predictive: PosteriorPredictiveCheck | None = None
    sensitivity: SensitivityReport | None = None

    def to_dict(self) -> dict:
        d = {}
        if self.holdout:
            d["holdout"] = self.holdout.to_dict()
        if self.posterior_predictive:
            d["posterior_predictive"] = self.posterior_predictive.to_dict()
        if self.sensitivity:
            d["sensitivity"] = self.sensitivity.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @property
    def all_warnings(self) -> list[str]:
        warnings = []
        if self.holdout:
            warnings.extend(self.holdout.warnings)
        if self.posterior_predictive:
            warnings.extend(self.posterior_predictive.warnings)
        if self.sensitivity:
            warnings.extend(self.sensitivity.warnings)
        return warnings
