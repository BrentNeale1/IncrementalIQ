"""Seed the database with realistic mock model results for dashboard demo."""
import json
import datetime
from sqlalchemy.orm import Session
from backend.db.config import engine, Base, SessionLocal
from backend.db.models import ModelRun

# Realistic posteriors for 5 channels on the sample dataset
CHANNEL_POSTERIORS = [
    {
        "channel": "spend_google_search",
        "beta_mean": 0.0042,
        "beta_sd": 0.0008,
        "beta_hdi_3": 0.0028,
        "beta_hdi_97": 0.0057,
        "adstock_alpha_mean": 0.35,
        "adstock_alpha_sd": 0.08,
        "adstock_alpha_hdi_3": 0.21,
        "adstock_alpha_hdi_97": 0.49,
        "saturation_lam_mean": 0.62,
        "saturation_lam_sd": 0.12,
        "saturation_lam_hdi_3": 0.40,
        "saturation_lam_hdi_97": 0.83,
        "contribution_mean": 285400.0,
        "contribution_pct": 28.5,
        "contribution_hdi_3": 22.1,
        "contribution_hdi_97": 35.2,
    },
    {
        "channel": "spend_google_shopping",
        "beta_mean": 0.0031,
        "beta_sd": 0.0009,
        "beta_hdi_3": 0.0015,
        "beta_hdi_97": 0.0048,
        "adstock_alpha_mean": 0.28,
        "adstock_alpha_sd": 0.09,
        "adstock_alpha_hdi_3": 0.12,
        "adstock_alpha_hdi_97": 0.44,
        "saturation_lam_mean": 0.55,
        "saturation_lam_sd": 0.14,
        "saturation_lam_hdi_3": 0.30,
        "saturation_lam_hdi_97": 0.79,
        "contribution_mean": 178200.0,
        "contribution_pct": 17.8,
        "contribution_hdi_3": 11.4,
        "contribution_hdi_97": 24.9,
    },
    {
        "channel": "spend_google_pmax",
        "beta_mean": 0.0027,
        "beta_sd": 0.0011,
        "beta_hdi_3": 0.0008,
        "beta_hdi_97": 0.0047,
        "adstock_alpha_mean": 0.41,
        "adstock_alpha_sd": 0.12,
        "adstock_alpha_hdi_3": 0.19,
        "adstock_alpha_hdi_97": 0.62,
        "saturation_lam_mean": 0.48,
        "saturation_lam_sd": 0.15,
        "saturation_lam_hdi_3": 0.22,
        "saturation_lam_hdi_97": 0.76,
        "contribution_mean": 142500.0,
        "contribution_pct": 14.2,
        "contribution_hdi_3": 6.8,
        "contribution_hdi_97": 22.4,
    },
    {
        "channel": "spend_meta_feed",
        "beta_mean": 0.0022,
        "beta_sd": 0.0010,
        "beta_hdi_3": 0.0005,
        "beta_hdi_97": 0.0041,
        "adstock_alpha_mean": 0.22,
        "adstock_alpha_sd": 0.10,
        "adstock_alpha_hdi_3": 0.05,
        "adstock_alpha_hdi_97": 0.40,
        "saturation_lam_mean": 0.70,
        "saturation_lam_sd": 0.13,
        "saturation_lam_hdi_3": 0.47,
        "saturation_lam_hdi_97": 0.91,
        "contribution_mean": 98600.0,
        "contribution_pct": 9.9,
        "contribution_hdi_3": 3.2,
        "contribution_hdi_97": 17.1,
    },
    {
        "channel": "spend_google_youtube",
        "beta_mean": 0.0015,
        "beta_sd": 0.0012,
        "beta_hdi_3": 0.0001,
        "beta_hdi_97": 0.0038,
        "adstock_alpha_mean": 0.52,
        "adstock_alpha_sd": 0.14,
        "adstock_alpha_hdi_3": 0.27,
        "adstock_alpha_hdi_97": 0.76,
        "saturation_lam_mean": 0.38,
        "saturation_lam_sd": 0.16,
        "saturation_lam_hdi_3": 0.12,
        "saturation_lam_hdi_97": 0.68,
        "contribution_mean": 52300.0,
        "contribution_pct": 5.2,
        "contribution_hdi_3": 0.4,
        "contribution_hdi_97": 12.8,
    },
]

DIAGNOSTICS = {
    "r_squared_mean": 0.91,
    "r_squared_hdi_3": 0.87,
    "r_squared_hdi_97": 0.94,
    "mape_mean": 0.082,
    "mape_hdi_3": 0.061,
    "mape_hdi_97": 0.108,
    "divergences": 3,
    "warnings": [],
}

BASELINE_PCT = 24.4  # remaining % after channel contributions

RESULTS_JSON = {
    "channel_posteriors": CHANNEL_POSTERIORS,
    "baseline_contribution_pct": BASELINE_PCT,
    "diagnostics": DIAGNOSTICS,
}

CONFIG = {
    "adstock_l_max": 8,
    "yearly_seasonality": 2,
    "chains": 2,
    "tune": 500,
    "draws": 500,
    "target_accept": 0.9,
    "random_seed": 42,
}

CHANNELS_USED = [
    "spend_google_search",
    "spend_google_shopping",
    "spend_google_pmax",
    "spend_meta_feed",
    "spend_google_youtube",
]

db = SessionLocal()
try:
    run = ModelRun(
        upload_id=1,
        status="completed",
        target_variable="revenue",
        channels_used=json.dumps(CHANNELS_USED),
        config_json=json.dumps(CONFIG),
        results_json=json.dumps(RESULTS_JSON),
        diagnostics_json=json.dumps(DIAGNOSTICS),
        idata_path=None,
        error_message=None,
        validation_json=None,
    )
    db.add(run)
    db.commit()
    print(f"Seeded ModelRun id={run.id}, status=completed, upload_id=1")
    print(f"Channels: {', '.join(CHANNELS_USED)}")
    print(f"Baseline: {BASELINE_PCT}%, R2: {DIAGNOSTICS['r_squared_mean']}, MAPE: {DIAGNOSTICS['mape_mean']}")
finally:
    db.close()
