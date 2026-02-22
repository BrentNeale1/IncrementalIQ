# IncrementIQ

A **local-first web application** for small-to-mid-tier digital advertisers ($5k-$30k/month ad spend) that measures the **incremental causal impact** of advertising channels on business outcomes using Bayesian Media Mix Modelling.

## Why IncrementIQ?

Platform-reported metrics (ROAS, CPA) tell you what the platform *claims* it drove. IncrementIQ tells you what **actually** happened by modelling the causal relationship between ad spend and business outcomes like revenue and orders.

- **Not attribution** - no touchpoint credit assignment
- **Not platform ROAS** - platform conversions are treated as a noisy covariate, never ground truth
- **Observational causal inference** - Bayesian MMM provides directional evidence, not RCT-level certainty

## Tech Stack

| Layer | Choice |
|---|---|
| Modelling | [PyMC](https://www.pymc.io/) via [pymc-marketing](https://github.com/pymc-labs/pymc-marketing) |
| Backend | FastAPI (Python) |
| Frontend | React |
| Database | SQLite via SQLAlchemy (PostgreSQL-ready via config swap) |
| Data Validation | Pandera |
| Visualisation | Plotly |
| Experiment Analysis | CausalImpact (Python) |

## Features

### Data Ingestion
- CSV upload with automatic schema validation and channel name standardisation
- API integrations: Google Ads, Meta (Facebook/Instagram), GA4, Shopify, WooCommerce
- Multi-source merge pipeline (combine ad spend + sessions + revenue from different platforms)
- Data quality reporting: history length checks, gap detection, spike detection, spend variance analysis

### Bayesian MMM
- Adstock transformation with geometric decay (carryover) and Hill saturation curves
- Half-Normal priors on channel coefficients (spend can only increase outcomes)
- Full posterior distributions - uncertainty is never hidden
- Channel contribution decomposition with credible intervals

### Experiment Validation
- **Spend-scaling experiments** via CausalImpact - tag a deliberate spend change and measure incremental impact
- **Product-level experiments** (Google Ads) - compare advertised vs. non-advertised product performance, with PMax/auto-bidding safety guard
- **Cross-correlation lag detection** - find the delay between spend and outcome per channel

### Output Views
- **Simple** - plain-language channel contributions with High/Moderate/Low confidence labels
- **Intermediate** - confidence intervals, adstock decay curves, saturation curves
- **Advanced** - full posterior distributions, model diagnostics, validation results

### Trust Score
Every report includes a composite trust score based on data quality, model fit, and experiment calibration:
- **Reliable** - good data, strong fit, calibration within tolerance
- **Use with caution** - moderate issues, directionally useful
- **Insufficient data** - recommend running an experiment before acting

## Quick Start

### Prerequisites
- Python 3.11+

### Installation

```bash
pip install -r requirements.txt
```

### Run the API server

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Run tests

```bash
# All tests except slow MCMC fitting tests
python -m pytest tests/ -m "not slow" -v

# Full suite including MCMC tests
python -m pytest tests/ -v
```

## API Endpoints

### Data Ingestion
- `POST /api/upload` - Upload a CSV file
- `GET /api/uploads` - List uploads
- `GET /api/uploads/{id}/quality` - Get data quality report

### API Integrations
- `POST /api/connections` - Register an API connection (Google Ads, Meta, GA4, Shopify, WooCommerce)
- `GET /api/connections` - List connections (credentials redacted)
- `POST /api/connections/{id}/test` - Test connectivity
- `DELETE /api/connections/{id}` - Deactivate a connection
- `POST /api/sync` - Sync data from a connection
- `GET /api/syncs` - List sync history
- `POST /api/merge` - Merge multiple uploads into a unified dataset

### Model
- `POST /api/model/run` - Fit a Bayesian MMM
- `GET /api/model/runs` - List model runs
- `GET /api/model/runs/{id}` - Get full model results
- `GET /api/model/runs/{id}/simple` - Simple output view
- `GET /api/model/runs/{id}/intermediate` - Intermediate output view
- `GET /api/model/runs/{id}/advanced` - Advanced output view
- `POST /api/model/validate` - Run holdout validation and posterior predictive checks

### Experiments
- `POST /api/experiments/lag-detection` - Detect spend-to-outcome lag per channel
- `POST /api/experiments/spend-scaling` - Run a spend-scaling experiment via CausalImpact
- `POST /api/experiments/product` - Run a product-level experiment (Google Ads only)

## Project Structure

```
backend/
  main.py                  # FastAPI app with all endpoints
  ingest/                  # CSV parsing, schema validation, data quality
  db/                      # SQLAlchemy models and config
  models/                  # Bayesian MMM (pymc-marketing), data prep, validation
  experiments/             # Lag detection, spend-scaling, product experiments
  integrations/            # API connectors (Google Ads, Meta, GA4, Shopify, WooCommerce)
  outputs/                 # Trust score, simple/intermediate/advanced views
tests/                     # 157 tests across all modules
```

## Known Limitations

- **Collinearity:** Channels that always run together produce wide posterior intervals. This is correct behaviour, not a bug.
- **PMax spend breakdown:** Google Ads API does not expose clean channel-level spend for Performance Max campaigns.
- **Cross-channel interactions:** V1 does not model interaction effects between channels (e.g. Search lift when YouTube is running).
- **Low spend variance:** If spend is too stable over time, adstock decay rates cannot be reliably estimated.
- **Observational inference:** MMM is not RCT-level certainty. Results should be validated with experiments.

## License

Proprietary. All rights reserved.
