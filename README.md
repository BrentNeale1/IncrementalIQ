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

## Current Status

| Component | Status |
|---|---|
| CSV ingestion + validation | Done |
| Data quality reporting | Done |
| Bayesian MMM (pymc-marketing) | Done |
| Simple / Intermediate / Advanced views | Done |
| Holdout validation + sensitivity analysis | Done |
| Experiments (lag detection, spend-scaling, product) | Done |
| API integrations (Google Ads, Meta, GA4, Shopify, WooCommerce) | Done |
| React frontend SPA | Done |
| **Testing with real data** | **Next** |

### What's Next

1. Upload real advertiser data (Google Ads + Shopify) using the CSV template (`data_template.csv`)
2. Run the model on real data and validate accuracy
3. Tune model settings if needed (adstock, seasonality, priors)
4. Run spend-scaling experiments to calibrate model predictions against real-world evidence

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+

### Installation

```bash
pip install -r requirements.txt
cd frontend && npm install
```

### Run the app

```bash
# Terminal 1: Backend
uvicorn backend.main:app --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

### Seed demo data (optional)

```bash
python generate_sample.py      # Creates sample_data.csv (730 days, 5 channels)
python seed_mock_results.py    # Inserts mock model results into the database
```

### Run tests

```bash
# All tests except slow MCMC fitting tests
python -m pytest tests/ -m "not slow" -v

# Full suite including MCMC tests
python -m pytest tests/ -v
```

## Data Requirements

### CSV template

A template is included at `data_template.csv`. Each row is one channel/campaign per day.

| Column | Source | Notes |
|---|---|---|
| `date` | - | YYYY-MM-DD, daily granularity |
| `channel` | Google Ads campaign type | `google_search`, `google_shopping`, `google_pmax`, `google_youtube` |
| `campaign` | Google Ads campaign name | Free text |
| `spend` | Google Ads: Cost | Daily per campaign |
| `impressions` | Google Ads: Impressions | Daily per campaign |
| `clicks` | Google Ads: Clicks | Daily per campaign |
| `in_platform_conversions` | Google Ads: Conversions | Covariate only, never outcome |
| `revenue` | Shopify: Total sales | **Same value for all rows on a given date** (store-wide) |
| `orders` | Shopify: Total orders | **Same value for all rows on a given date** (store-wide) |
| `sessions_organic` | GA4 (optional, 0 if unavailable) | Site-wide daily |
| `sessions_direct` | GA4 (optional, 0 if unavailable) | Site-wide daily |
| `sessions_email` | GA4 (optional, 0 if unavailable) | Site-wide daily |
| `sessions_referral` | GA4 (optional, 0 if unavailable) | Site-wide daily |

### Minimum data
- **9 months** daily data = hard minimum
- **18+ months** recommended for reliable results
- Revenue must be from your store (Shopify), **not** platform-reported revenue

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
frontend/
  src/
    components/            # Layout, charts (Plotly), stat cards, channel table
    views/                 # Dashboard, Upload, Model Run, Results, Experiments, Connections
    hooks/                 # useApi, useMutation data fetching
    styles/                # Full design system (cream/paper aesthetic, no CSS framework)
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
