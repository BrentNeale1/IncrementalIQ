"""Core Bayesian Media Mix Model wrapper around pymc-marketing.

Model structure (from CLAUDE.md):
    Revenue(t) = baseline(t) + Σ [β_c × adstock_c(spend_c(t))] + Σ [γ_o × organic_o(t)] + ε(t)

Key constraints enforced here:
- GeometricAdstock (Koyck decay): adstock(t) = spend(t) + λ × adstock(t-1)
- LogisticSaturation: saturated(x) = 1 - exp(-α × x)  (closest match to spec)
- HalfNormal priors on channel coefficients (β_c ≥ 0 always)
- in_platform_conversions as covariate with weakly informative prior, never outcome
- Fourier seasonality
- All results include full posterior distributions, never just point estimates
"""
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
import arviz as az
import numpy as np
import pandas as pd
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM
from pymc_extras.prior import Prior

from backend.models.data_prep import PreparedData

logger = logging.getLogger(__name__)

MODEL_ARTIFACTS_DIR = os.environ.get("MODEL_ARTIFACTS_DIR", "model_artifacts")

# --- Default model configuration aligned with CLAUDE.md ---

DEFAULT_ADSTOCK_L_MAX = 8  # max lag in days for geometric decay

DEFAULT_YEARLY_SEASONALITY = 6  # Fourier modes for annual seasonality

DEFAULT_SAMPLER_CONFIG = {
    "chains": 4,
    "tune": 1500,
    "draws": 1000,
    "target_accept": 0.9,
    "random_seed": 42,
}


@dataclass
class ModelConfig:
    """Configuration for a model run. Serialisable to JSON for storage."""
    adstock_l_max: int = DEFAULT_ADSTOCK_L_MAX
    yearly_seasonality: int = DEFAULT_YEARLY_SEASONALITY
    chains: int = 4
    tune: int = 1500
    draws: int = 1000
    target_accept: float = 0.9
    random_seed: int = 42

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "ModelConfig":
        return cls(**json.loads(s))


@dataclass
class ChannelPosterior:
    """Summary of posterior distribution for a single channel."""
    channel: str
    beta_mean: float
    beta_sd: float
    beta_hdi_3: float
    beta_hdi_97: float
    adstock_alpha_mean: float
    adstock_alpha_sd: float
    adstock_alpha_hdi_3: float
    adstock_alpha_hdi_97: float
    saturation_lam_mean: float
    saturation_lam_sd: float
    saturation_lam_hdi_3: float
    saturation_lam_hdi_97: float
    contribution_mean: float
    contribution_pct: float
    contribution_hdi_3: float
    contribution_hdi_97: float


@dataclass
class ControlPosterior:
    """Summary of posterior distribution for a single control variable."""
    control: str           # e.g. "sessions_organic"
    display_name: str      # e.g. "Organic Sessions"
    gamma_mean: float      # coefficient posterior mean
    gamma_sd: float
    gamma_hdi_3: float
    gamma_hdi_97: float
    contribution_mean: float   # total contribution in dollars
    contribution_pct: float    # % of total revenue
    contribution_hdi_3: float
    contribution_hdi_97: float


@dataclass
class ModelDiagnostics:
    """Model fit diagnostics."""
    r_squared_mean: float
    r_squared_hdi_3: float
    r_squared_hdi_97: float
    mape_mean: float
    mape_hdi_3: float
    mape_hdi_97: float
    divergences: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class ModelResults:
    """Complete model results for storage and API responses."""
    channel_posteriors: list[ChannelPosterior]
    control_posteriors: list[ControlPosterior]
    baseline_contribution_pct: float  # now = 100 - channels - controls
    diagnostics: ModelDiagnostics

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _control_display_name(control: str) -> str:
    """Convert internal control names to readable labels."""
    labels = {
        "sessions_organic": "Organic Sessions",
        "sessions_direct": "Direct Sessions",
        "sessions_email": "Email Sessions",
        "sessions_referral": "Referral Sessions",
    }
    if control in labels:
        return labels[control]
    # ipc_* controls: "In-Platform Conv. (Channel Name)"
    if control.startswith("ipc_"):
        ch_name = control[4:].replace("_", " ").title()
        return f"In-Platform Conv. ({ch_name})"
    return control.replace("_", " ").title()


def _build_model_config(data: PreparedData) -> dict:
    """Build pymc-marketing model_config dict with CLAUDE.md-specified priors.

    - saturation_beta: HalfNormal — enforces β_c ≥ 0 (spend never decreases outcome)
    - gamma_control: Normal with small sigma — weakly informative for in_platform_conversions
    - likelihood: Normal with HalfNormal sigma
    """
    n_channels = len(data.channel_columns)

    # Spend-share-weighted sigma for channel priors (more informative for
    # high-spend channels, wider for low-spend).
    spend_totals = data.X[data.channel_columns].sum(axis=0).values
    total_spend = spend_totals.sum()
    if total_spend > 0:
        spend_share = spend_totals / total_spend
        prior_sigma = np.maximum(n_channels * spend_share, 0.5)
    else:
        prior_sigma = np.full(n_channels, 2.0)

    return {
        "intercept": Prior("Normal", mu=0, sigma=2),
        "saturation_beta": Prior("HalfNormal", sigma=prior_sigma, dims="channel"),
        "saturation_lam": Prior("Gamma", alpha=3, beta=1, dims="channel"),
        "adstock_alpha": Prior("Beta", alpha=1, beta=3, dims="channel"),
        "gamma_control": Prior("Normal", mu=0, sigma=2, dims="control"),
        "gamma_fourier": Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
    }


def build_mmm(data: PreparedData, config: ModelConfig | None = None) -> MMM:
    """Construct the pymc-marketing MMM instance.

    Returns an unfitted model ready for .fit().
    """
    config = config or ModelConfig()

    model_config = _build_model_config(data)

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

    return mmm


def fit_mmm(
    mmm: MMM,
    data: PreparedData,
    config: ModelConfig | None = None,
) -> MMM:
    """Fit the MMM using MCMC sampling.

    Returns the fitted model (mmm.idata will be populated).
    """
    config = config or ModelConfig()

    logger.info(
        "Starting MCMC sampling: %d chains, %d tune, %d draws",
        config.chains, config.tune, config.draws,
    )

    mmm.fit(
        X=data.X,
        y=data.y,
        chains=config.chains,
        tune=config.tune,
        draws=config.draws,
        target_accept=config.target_accept,
        random_seed=config.random_seed,
        nuts_sampler="nutpie",
    )

    logger.info("Sampling complete")
    return mmm


def _hdi_numpy(arr: np.ndarray, hdi_prob: float = 0.94) -> tuple[float, float]:
    """Compute HDI from a 1D numpy array, avoiding xarray issues with az.hdi."""
    arr = arr.flatten()
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    arr_sorted = np.sort(arr)
    interval_size = max(1, int(np.ceil(hdi_prob * n)))
    if interval_size >= n:
        return float(arr_sorted[0]), float(arr_sorted[-1])
    widths = arr_sorted[interval_size:] - arr_sorted[:n - interval_size]
    best_idx = int(np.argmin(widths))
    return float(arr_sorted[best_idx]), float(arr_sorted[best_idx + interval_size])


def extract_results(mmm: MMM, data: PreparedData) -> ModelResults:
    """Extract structured results from a fitted MMM.

    Returns channel posteriors with full uncertainty quantification,
    contribution decomposition, and diagnostics.
    """
    posterior = mmm.idata.posterior

    # --- Channel posteriors ---
    channel_posteriors = []

    # Channel names as they appear in the model coords (convert from xarray to str)
    channel_names = [str(c.values) if hasattr(c, 'values') else str(c)
                     for c in posterior.coords.get("channel", [])]

    # Extract posterior summaries for each channel
    summary = az.summary(
        mmm.idata,
        var_names=["saturation_beta", "saturation_lam", "adstock_alpha"],
        hdi_prob=0.94,
    )

    # Compute posterior predictive for diagnostics
    pp = mmm.sample_posterior_predictive(data.X, extend_idata=True, combined=True)

    # Get channel contributions
    # The posterior contains channel_contribution (adstocked + saturated spend × beta)
    if "channel_contribution" in posterior:
        channel_contrib = posterior["channel_contribution"]
        # Shape: (chain, draw, date, channel) — sum over dates for total contribution
        total_contrib_samples = channel_contrib.sum(dim="date")
        total_all_channels = total_contrib_samples.sum(dim="channel")
    else:
        total_contrib_samples = None
        total_all_channels = None

    # Determine target scaler — pymc-marketing internally divides y by max(y),
    # so channel_contribution values are in that normalized scale.  Rescale to
    # original dollars and compute percentages against actual observed revenue.
    target_scale = 1.0
    if hasattr(mmm, 'scalers') and hasattr(mmm.scalers, '_target'):
        target_scale = float(mmm.scalers._target.values)
    total_revenue = float(data.y.sum())

    def _safe(row, col, default=0.0):
        if row is not None and col in row.index:
            return float(row[col])
        return default

    # --- Collect raw contribution means (in dollars) for all components ---
    # We'll normalize percentages at the end so channels + controls + baseline = 100%
    channel_contrib_means = {}  # ch_name → mean dollars
    channel_contrib_hdis = {}   # ch_name → (low, high) dollars

    for i, ch_name in enumerate(channel_names):
        if total_contrib_samples is not None:
            ch_contrib_np = total_contrib_samples.sel(channel=ch_name).values.flatten()
            contrib_mean = float(np.mean(ch_contrib_np)) * target_scale
            low, high = _hdi_numpy(ch_contrib_np, 0.94)
            channel_contrib_means[ch_name] = contrib_mean
            channel_contrib_hdis[ch_name] = (low * target_scale, high * target_scale)
        else:
            channel_contrib_means[ch_name] = 0.0
            channel_contrib_hdis[ch_name] = (0.0, 0.0)

    control_contrib_means = {}  # ctrl_name → mean dollars
    control_contrib_hdis = {}   # ctrl_name → (low, high) dollars

    control_names = [str(c.values) if hasattr(c, 'values') else str(c)
                     for c in posterior.coords.get("control", [])]

    gamma_summary = None
    if control_names and "control_contribution" in posterior:
        ctrl_contrib = posterior["control_contribution"]
        total_ctrl_samples = ctrl_contrib.sum(dim="date")

        gamma_summary = az.summary(
            mmm.idata,
            var_names=["gamma_control"],
            hdi_prob=0.94,
        )

        for i, ctrl_name in enumerate(control_names):
            ctrl_np = total_ctrl_samples.sel(control=ctrl_name).values.flatten()
            ctrl_mean = float(np.mean(ctrl_np)) * target_scale
            ctrl_low, ctrl_high = _hdi_numpy(ctrl_np, 0.94)
            control_contrib_means[ctrl_name] = ctrl_mean
            control_contrib_hdis[ctrl_name] = (ctrl_low * target_scale, ctrl_high * target_scale)

    # --- Extract baseline (intercept + seasonality) from posterior ---
    baseline_mean = 0.0
    try:
        intercept = posterior["intercept_contribution"].values  # (chain, draw)
        n_obs = data.daily_rows
        intercept_total = intercept.flatten() * n_obs

        fourier = posterior["yearly_seasonality_contribution"].values  # (chain, draw, date)
        fourier_total = fourier.reshape(-1, fourier.shape[-1]).sum(axis=1)

        baseline_samples = (intercept_total + fourier_total) * target_scale
        baseline_mean = float(np.mean(baseline_samples))
    except (KeyError, Exception):
        pass  # baseline_mean stays 0

    # --- Normalize percentages so all components sum to 100% ---
    # Use the model's own predicted total (channels + controls + baseline) as denominator.
    # This avoids the issue where posterior means computed against observed revenue
    # can sum to more or less than 100%.
    model_total = (
        sum(channel_contrib_means.values())
        + sum(control_contrib_means.values())
        + max(0.0, baseline_mean)
    )

    def _pct(value):
        return round((value / model_total * 100) if model_total > 0 else 0, 2)

    baseline_pct = _pct(max(0.0, baseline_mean))

    # --- Build channel posteriors with normalized percentages ---
    for i, ch_name in enumerate(channel_names):
        beta_key = f"saturation_beta[{ch_name}]"
        if beta_key in summary.index:
            beta_row = summary.loc[beta_key]
        else:
            beta_row = summary.iloc[i] if i < len(summary) else None

        alpha_key = f"adstock_alpha[{ch_name}]"
        alpha_row = summary.loc[alpha_key] if alpha_key in summary.index else None

        lam_key = f"saturation_lam[{ch_name}]"
        lam_row = summary.loc[lam_key] if lam_key in summary.index else None

        contrib_mean = channel_contrib_means[ch_name]
        contrib_hdi_low, contrib_hdi_high = channel_contrib_hdis[ch_name]

        channel_posteriors.append(ChannelPosterior(
            channel=ch_name,
            beta_mean=_safe(beta_row, "mean"),
            beta_sd=_safe(beta_row, "sd"),
            beta_hdi_3=_safe(beta_row, "hdi_3%"),
            beta_hdi_97=_safe(beta_row, "hdi_97%"),
            adstock_alpha_mean=_safe(alpha_row, "mean"),
            adstock_alpha_sd=_safe(alpha_row, "sd"),
            adstock_alpha_hdi_3=_safe(alpha_row, "hdi_3%"),
            adstock_alpha_hdi_97=_safe(alpha_row, "hdi_97%"),
            saturation_lam_mean=_safe(lam_row, "mean"),
            saturation_lam_sd=_safe(lam_row, "sd"),
            saturation_lam_hdi_3=_safe(lam_row, "hdi_3%"),
            saturation_lam_hdi_97=_safe(lam_row, "hdi_97%"),
            contribution_mean=contrib_mean,
            contribution_pct=_pct(contrib_mean),
            contribution_hdi_3=contrib_hdi_low,
            contribution_hdi_97=contrib_hdi_high,
        ))

    # --- Build control posteriors with normalized percentages ---
    control_posteriors = []
    for i, ctrl_name in enumerate(control_names):
        gamma_key = f"gamma_control[{ctrl_name}]"
        if gamma_summary is not None and gamma_key in gamma_summary.index:
            gamma_row = gamma_summary.loc[gamma_key]
        elif gamma_summary is not None and i < len(gamma_summary):
            gamma_row = gamma_summary.iloc[i]
        else:
            gamma_row = None

        ctrl_mean = control_contrib_means.get(ctrl_name, 0.0)
        ctrl_hdi = control_contrib_hdis.get(ctrl_name, (0.0, 0.0))

        control_posteriors.append(ControlPosterior(
            control=ctrl_name,
            display_name=_control_display_name(ctrl_name),
            gamma_mean=_safe(gamma_row, "mean"),
            gamma_sd=_safe(gamma_row, "sd"),
            gamma_hdi_3=_safe(gamma_row, "hdi_3%"),
            gamma_hdi_97=_safe(gamma_row, "hdi_97%"),
            contribution_mean=ctrl_mean,
            contribution_pct=_pct(ctrl_mean),
            contribution_hdi_3=ctrl_hdi[0],
            contribution_hdi_97=ctrl_hdi[1],
        ))

    # --- Diagnostics ---
    diagnostics = _compute_diagnostics(mmm, data, pp)

    return ModelResults(
        channel_posteriors=channel_posteriors,
        control_posteriors=control_posteriors,
        baseline_contribution_pct=round(baseline_pct, 2),
        diagnostics=diagnostics,
    )


def _extract_pp_samples(mmm: MMM, pp, target_var: str, n_obs: int) -> np.ndarray | None:
    """Extract posterior predictive samples as a 2D numpy array (samples, obs).

    Handles the various return types from sample_posterior_predictive:
    - xarray DataArray (multidimensional MMM returns this directly)
    - InferenceData with posterior_predictive group

    Parameters
    ----------
    n_obs : expected number of observations, used to orient ambiguous arrays
    """
    import xarray as xr

    def _orient(arr: np.ndarray) -> np.ndarray:
        """Ensure array is (samples, obs). If shape is (obs, samples), transpose."""
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 3:
            # (chain, draw, obs) → (chain*draw, obs)
            return arr.reshape(-1, arr.shape[-1])
        # 2D: figure out which axis is obs
        if arr.shape[0] == n_obs and arr.shape[1] != n_obs:
            return arr.T  # was (obs, samples), transpose to (samples, obs)
        return arr

    # pp might be an xarray DataArray directly
    if isinstance(pp, (xr.DataArray, xr.Dataset)):
        if isinstance(pp, xr.Dataset):
            var_name = target_var if target_var in pp else list(pp.data_vars)[0]
            arr = pp[var_name].values
        else:
            arr = pp.values
        return _orient(arr)

    # pp might be InferenceData
    if hasattr(pp, "posterior_predictive"):
        pp_group = pp.posterior_predictive
        var_name = target_var if target_var in pp_group else "y"
        if var_name in pp_group:
            return _orient(pp_group[var_name].values)

    return None


def _reconstruct_mu_draws(
    posterior, target_scale: float, n_obs: int,
) -> np.ndarray | None:
    """Reconstruct noiseless predicted values (mu) per draw from contributions.

    Returns array of shape (n_samples, n_obs) in original scale, or None on failure.
    The model is: y[t] = intercept + channels[t] + controls[t] + fourier[t] + noise[t]
    We sum contributions excluding noise to get the mean prediction per draw.
    """
    try:
        ch = posterior["channel_contribution"].sum(dim="channel").values
        ctrl = posterior["control_contribution"].sum(dim="control").values
        fourier = posterior["yearly_seasonality_contribution"].values
        # intercept_contribution is the per-draw intercept (constant across dates)
        intercept = posterior["intercept_contribution"].values  # (chain, draw)
        mu = ch + ctrl + fourier + intercept[:, :, np.newaxis]
        return mu.reshape(-1, n_obs) * target_scale
    except (KeyError, Exception):
        return None


def _compute_diagnostics(mmm: MMM, data: PreparedData, pp) -> ModelDiagnostics:
    """Compute model fit diagnostics: R², MAPE, divergences.

    R² and MAPE use the posterior mean prediction (noise-free), following
    Gelman et al. (2019) Bayesian R² recommendations. Per-sample posterior
    predictive draws include observation noise which inflates residuals and
    deflates R²; the mean prediction removes this noise bias.

    HDI intervals are computed from per-draw noiseless predictions
    (reconstructed from contribution variables in the posterior).
    """
    warnings = []

    y_true = data.y.values
    n_obs = len(y_true)

    y_pred_samples = _extract_pp_samples(mmm, pp, data.target_variable, n_obs)
    if y_pred_samples is None:
        y_pred_samples = np.zeros_like(y_true).reshape(1, -1)
        warnings.append("Could not extract posterior predictive samples")

    if y_pred_samples.ndim == 1:
        y_pred_samples = y_pred_samples.reshape(1, -1)

    # Rescale from normalized to original scale — pymc-marketing internally
    # divides the target by max(y), so PP samples come back in that scale.
    target_scale = 1.0
    if hasattr(mmm, "scalers") and hasattr(mmm.scalers, "_target"):
        ts = float(mmm.scalers._target.values)
        if ts > 0:
            target_scale = ts
            y_pred_samples = y_pred_samples * target_scale

    # Mean prediction (noise averages out across posterior draws)
    y_pred_mean = y_pred_samples.mean(axis=0)

    ss_tot = ((y_true - y_true.mean()) ** 2).sum()

    # R² from mean prediction
    ss_res_mean = ((y_true - y_pred_mean) ** 2).sum()
    r2_mean = float(1.0 - ss_res_mean / ss_tot) if ss_tot > 0 else 0.0

    # R² HDI from per-draw noiseless predictions
    mu_draws = _reconstruct_mu_draws(mmm.idata.posterior, target_scale, n_obs)
    if mu_draws is not None:
        r2_draws = 1.0 - ((y_true - mu_draws) ** 2).sum(axis=1) / ss_tot
        r2_hdi = _hdi_numpy(r2_draws, 0.94)
    else:
        r2_hdi = (r2_mean, r2_mean)

    # MAPE from mean prediction
    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        ape_mean = np.abs(
            (y_true[nonzero_mask] - y_pred_mean[nonzero_mask])
            / y_true[nonzero_mask]
        )
        mape_mean = float(ape_mean.mean() * 100)

        # MAPE HDI from per-draw noiseless predictions
        if mu_draws is not None:
            ape_draws = np.abs(
                (y_true[nonzero_mask] - mu_draws[:, nonzero_mask])
                / y_true[nonzero_mask]
            )
            mape_draws = ape_draws.mean(axis=1) * 100
            mape_hdi = _hdi_numpy(mape_draws, 0.94)
        else:
            mape_hdi = (mape_mean, mape_mean)
    else:
        mape_mean = 0.0
        mape_hdi = (0.0, 0.0)

    # Divergences
    divergences = 0
    if "sample_stats" in mmm.idata:
        divergences = int(mmm.idata.sample_stats["diverging"].sum().values)

    # Warnings per CLAUDE.md thresholds
    if mape_mean > 15:
        warnings.append(
            f"MAPE is {mape_mean:.1f}% (threshold: 15%). "
            "Model fit may not be reliable."
        )
    if divergences > 0:
        warnings.append(
            f"{divergences} divergent transitions detected. "
            "Consider increasing target_accept or reparameterising."
        )

    return ModelDiagnostics(
        r_squared_mean=round(r2_mean, 4),
        r_squared_hdi_3=round(float(r2_hdi[0]), 4),
        r_squared_hdi_97=round(float(r2_hdi[1]), 4),
        mape_mean=round(mape_mean, 2),
        mape_hdi_3=round(float(mape_hdi[0]), 2),
        mape_hdi_97=round(float(mape_hdi[1]), 2),
        divergences=divergences,
        warnings=warnings,
    )


def save_idata(mmm: MMM, run_id: int) -> str:
    """Save InferenceData to disk. Returns the file path."""
    artifacts_dir = Path(MODEL_ARTIFACTS_DIR)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / f"run_{run_id}.nc"
    mmm.idata.to_netcdf(str(path))
    logger.info("Saved InferenceData to %s", path)
    return str(path)
