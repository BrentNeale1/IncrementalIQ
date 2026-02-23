# IncrementIQ — Claude Code Project Brief

## What This Tool Is

IncrementIQ is a **local-first web application** for small-to-mid-tier digital advertisers ($5k–$30k/month ad spend). It measures the **incremental causal impact** of advertising channels on business outcomes (revenue, orders, leads) using Bayesian Media Mix Modelling.

Users are digital advertising account managers. They need to make budget allocation decisions grounded in causal evidence, not platform-reported metrics.

## What This Tool Is NOT — Read This Before Making Any Suggestions

- **Not an attribution tool.** We do not assign conversion credit to touchpoints. Do not suggest last-click, data-driven, or any touchpoint attribution logic.
- **Not a platform efficiency tool.** We do not calculate CAC or ROAS. Do not suggest these as outputs or metrics.
- **Not a replacement for controlled experiments.** The model provides observational causal inference. Always caveat this in UI copy and documentation.
- **Not a reporting layer on top of platform data.** Platform-reported conversions are treated as a noisy covariate, never as an outcome variable or ground truth.

---

## Tech Stack — These Decisions Are Final

Do not suggest alternatives to these unless explicitly asked.

| Layer | Choice | Notes |
|---|---|---|
| Modeling | PyMC via pymc-marketing | Bayesian MMM only. Not Stan, not Prophet, not sklearn. |
| Backend | FastAPI | Python. Async. |
| Frontend | React | Standard component library. |
| Database | SQLite via SQLAlchemy ORM | Local-first. ORM layer must support PostgreSQL migration via config change only — no rewrites. |
| Data validation | pandera or great_expectations | Schema enforcement, gap detection, spike detection. |
| Visualisation | Plotly | Interactive charts in the web app. |
| Experiment analysis | causalimpact (Python port) | Used for spend-scaling and geo-holdout validation only. |
| Cross-correlation | statsmodels.tsa.stattools.ccf | Lag detection preprocessing. |

---

## Data Schema

### Standard ingestion schema (CSV and API)

```
date                      | YYYY-MM-DD, daily granularity
channel                   | string (e.g. "google_search", "meta_feed", "google_pmax")
campaign                  | string, campaign name
spend                     | float, local currency
impressions               | integer
clicks                    | integer
in_platform_conversions   | float — COVARIATE ONLY, never outcome variable
revenue                   | float, local currency — PRIMARY OUTCOME VARIABLE
orders                    | integer — SECONDARY OUTCOME VARIABLE
sessions_organic          | integer
sessions_direct           | integer
sessions_email            | integer
sessions_referral         | integer
```

### Channel name conventions (standardise on ingest)

```
google_search
google_shopping
google_pmax
google_youtube
meta_feed
meta_instagram
meta_stories
```

### Data quality thresholds

- `< 9 months` history → hard reject, do not run model
- `< 18 months` history → warn user, proceed with caution flag
- `≥ 18 months` → proceed normally
- MAPE > 15% on holdout → surface warning
- Holdout coverage < 80% → surface warning
- Spend-scaling experiment discrepancy > 30% vs MMM prediction → trigger recalibration recommendation

---

## Modeling — Critical Constraints

### Model structure

```
Revenue(t) = baseline(t) + Σ [β_c × adstock_c(spend_c(t))] + Σ [γ_o × organic_o(t)] + ε(t)
```

- `baseline(t)` — trend + seasonality via Fourier terms or Gaussian Process
- `adstock_c` — carryover + saturation transformation on spend before model entry
- `β_c` — channel coefficient, posterior distribution
- `organic_o` — organic, direct, email, referral as covariates
- `in_platform_conversions_c` — additional covariate per channel, weakly informative prior, never outcome
- `ε(t)` — observation noise

### Adstock transformation

Carryover (Koyck geometric decay):
```
adstock(t) = spend(t) + λ × adstock(t-1)
```

Saturation (Hill function):
```
saturated(x) = 1 - exp(-α × x)
```

Both `λ` and `α` are estimated as **posterior distributions** per channel — not fixed values.

### Priors — do not change without explicit instruction

- Channel coefficients (`β_c`): **Half-Normal** — spend can only increase outcomes, never decrease. This prevents nonsensical negative contributions.
- Saturation curve shape: Beta distribution
- Seasonality: informed by data periodicity
- In-platform conversions: weakly informative prior to prevent domination

### Contribution decomposition

Channel percentage contributions = posterior mean of (adstock-transformed spend × coefficient) / total explained revenue. Uncertainty propagated from posterior.

### What the model must never do

- Use in-platform conversions as the dependent variable
- Produce negative channel contribution coefficients (enforced by Half-Normal prior)
- Present point estimates without uncertainty quantification
- Hide wide posterior intervals — collinearity between channels produces wide intervals and this is correct, honest behaviour. Surface it to the user.

---

## Experiment Features — Constraints

### Spend-scaling experiments (primary validation path)

- User tags a date range + campaign as a deliberate spend change
- CausalImpact runs on that window using other channels as controls
- Output: incremental impact estimate with confidence interval
- Result is used to calibrate MMM posterior for that channel

### Product-level experiment (Google Ads only)

- User uploads advertised product list from Google Ads product reports
- Tool compares in-campaign vs. out-of-campaign product revenue/orders
- **Hard constraint:** Only valid for manually-controlled product selection. If campaign type is PMax or uses auto-bidding, the tool must refuse to run this test and explain why (selection bias — products are chosen based on conversion probability, not randomly).
- Enforce this check in the UI. Do not make it bypassable.

### Geo-holdout experiments

- Recommended for: YouTube, awareness/prospecting campaigns, high-uncertainty channels, new channels with insufficient history
- **Not a general default.** Do not surface geo-holdouts as the first recommendation for standard always-on campaigns.
- Primary recommendation for most users is spend-scaling experiments.

---

## Output Layer — Three Views

### Simple view (default for account managers)
- Channel contribution bar chart, plain-language labels
- Recommendation engine: flag underperforming channels, recommend spend-scaling test if intervals are wide
- No statistical terminology visible
- Uncertainty expressed as: "High confidence / Moderate confidence / Low confidence — more data needed"

### Intermediate view
- Contribution chart with visible confidence intervals
- Adstock decay curves per channel
- Saturation curves
- Lag summary (days between spend and impact)
- In-platform conversions vs. modelled contribution comparison

### Advanced view
- Full posterior distribution summaries (mean, SD, 94% HDI — not p-values)
- Model fit diagnostics: R², MAPE, posterior predictive check plots
- Adstock parameter estimates with credible intervals
- Data quality flags
- CausalImpact experiment results alongside MMM posterior updates

### Design rule — never violate this
Uncertainty is represented at every output level. Even the simple view has a confidence indicator. The tool must never present results as certain when they are not.

---

## Trust Score

Displayed on every report. Composite of:
- Data quality score (history length, gap frequency, spend variance)
- Model fit score (holdout MAPE, posterior predictive check result)
- Calibration score (experiment vs. MMM prediction agreement)

Three tiers:
- **Model results are reliable** — good data, strong fit, calibration within tolerance
- **Use with caution** — moderate data or fit issues, directionally useful
- **Insufficient data** — history too short, too gappy, or fit too poor. Recommend running an experiment before acting.

---

## Known Limitations — Document These in the UI

- **Collinearity:** Channels that always run together produce wide posterior intervals. This is correct behaviour, not a bug.
- **PMax spend breakdown:** Google Ads API does not expose clean campaign-level spend breakdown by channel type for PMax. Document this explicitly wherever PMax data is displayed.
- **Cross-channel interactions:** V1 does not model interaction effects between channels (e.g. Search lift when YouTube is running). Known V2 candidate.
- **Low spend variance:** If spend is too stable over time, adstock decay rates cannot be reliably estimated. Flag this when spend variance is low.
- **Observational inference:** MMM is not RCT-level certainty. This caveat must appear consistently throughout the UI.

---

## API Integrations (Phase 1B — build after model is validated)

Priority order:
1. Google Ads API — Search, Shopping, PMax, YouTube
2. Meta Marketing API — Facebook, Instagram
3. GA4 — organic, direct, referral sessions
4. Shopify / WooCommerce — revenue, orders

Phase 1A uses CSV ingestion only. Do not build API connectors until the core model is validated end-to-end on CSV data.

---

## Build Sequence

| Phase | What | Status |
|---|---|---|
| 1A | CSV ingestion, schema validation, data quality report, SQLite storage | **Complete** |
| 1A+ | Wide-format CSV auto-detection, Excel multi-sheet ingestion, SQLite WAL mode | **Complete** |
| 2 | Core Bayesian MMM (pymc-marketing) | **Complete** |
| 2+ | Channel filtering/merging (smart defaults, API endpoint, frontend UI), R²/MAPE diagnostics fix (mean-prediction Bayesian R²), seasonality bump (6 Fourier modes) | **Complete** |
| 4 (basic) | Simple + intermediate output views | **Complete** |
| 5 | Internal validation (holdout, posterior predictive checks, sensitivity) | **Complete** |
| 3 | Adstock, lag detection, spend-scaling experiments, product experiment | **Complete** |
| 4 (advanced) | Advanced output view | **Complete** |
| 1B | API integrations (Google Ads, Meta, GA4, Shopify, WooCommerce) | **Complete** |

**Current phase:** All phases complete (1A, 1A+, 2, 2+, 3, 4, 5, 1B).

---

## File Structure

Update this section as the project grows.

```
incrementiq/
├── CLAUDE.md                  # This file
├── requirements.txt           # Python dependencies (openpyxl for Excel support)
├── data_template.xlsx         # Template Excel workbook for data ingestion
├── backend/
│   ├── __init__.py
│   ├── main.py                # FastAPI entrypoint — ingestion, model, views, experiment, channel analysis endpoints
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── schema.py          # Pandera schema, channel aliases, valid channels (incl. audience_network, display, messenger, threads, unknown)
│   │   ├── csv_reader.py      # CSV parsing, channel standardisation, validation
│   │   ├── wide_reader.py     # Wide-format CSV auto-detection, wide→long conversion, Excel multi-sheet reader
│   │   ├── quality.py         # Data quality report (history length, gaps, spikes, spend variance)
│   │   └── service.py         # Orchestrator: detect format → read → validate → quality → store (CSV + Excel)
│   ├── db/
│   │   ├── __init__.py
│   │   ├── config.py          # SQLAlchemy engine, session, Base (SQLite WAL mode + busy_timeout → PostgreSQL via env var)
│   │   └── models.py          # ORM: Upload, DailyRecord, QualityReport, ModelRun, ExperimentResult, ApiConnection, ApiSync
│   ├── models/
│   │   ├── __init__.py
│   │   ├── data_prep.py       # DB → model-ready DataFrame (pivot, aggregate, controls, channel filtering/merging, recommend_channel_config)
│   │   ├── mmm.py             # MMM config, build, fit, result extraction, mean-prediction R²/MAPE diagnostics (pymc-marketing)
│   │   ├── service.py         # Orchestrator: data prep → fit → extract → store (accepts channel_config)
│   │   └── validation.py      # Holdout testing, posterior predictive checks, sensitivity analysis
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── lag_detection.py   # Cross-correlation lag detection (statsmodels ccf)
│   │   ├── spend_scaling.py   # Spend-scaling experiment via CausalImpact
│   │   └── product_experiment.py  # Product-level experiment with PMax/auto-bidding guard
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseConnector ABC, ConnectorConfig, FetchResult
│   │   ├── registry.py          # Connector factory + registration
│   │   ├── service.py           # Orchestration: sync_connection(), merge_sources()
│   │   ├── google_ads.py        # Google Ads connector (google-ads SDK)
│   │   ├── meta_ads.py          # Meta Marketing API connector (facebook-business SDK)
│   │   ├── ga4.py               # GA4 connector (google-analytics-data SDK)
│   │   ├── shopify.py           # Shopify connector (httpx)
│   │   └── woocommerce.py       # WooCommerce connector (httpx)
│   └── outputs/
│       ├── __init__.py
│       ├── trust_score.py     # Composite trust score (data quality + model fit + calibration)
│       └── views.py           # Simple, Intermediate, and Advanced output views
├── frontend/
│   ├── index.html               # HTML entry point (Google Fonts: Playfair Display, DM Sans, DM Mono)
│   ├── package.json             # React 18, react-router-dom, react-plotly.js, Vite
│   ├── vite.config.js           # Vite config with /api proxy to localhost:8000
│   └── src/
│       ├── main.jsx             # ReactDOM + BrowserRouter
│       ├── App.jsx              # Route definitions (/, /upload, /connections, /model/run, /model/runs/:runId/results, /experiments)
│       ├── api.js               # Fetch wrapper: apiGet, apiPost, apiPostFile, apiDelete
│       ├── constants.js         # Channel colors/names, trust tiers, Plotly defaults, caveat text
│       ├── styles/
│       │   └── index.css        # Full design system — cream/paper aesthetic, all component styles
│       ├── hooks/
│       │   └── useApi.js        # useApi(url) data fetching + useMutation() hooks
│       ├── components/
│       │   ├── Layout.jsx       # App shell: dark topbar + sidebar (channels, data stats, caveat) + Outlet
│       │   ├── TrustBadge.jsx   # Green/amber/red trust pill badge
│       │   ├── StatStrip.jsx    # 4-column stat cards (incremental rev, baseline, trust, actions)
│       │   ├── ActionCard.jsx   # Recommendation card (scale/test/reduce/geo variants)
│       │   ├── ChannelTable.jsx # Channel performance matrix with inline bars, CI ranges, confidence pips
│       │   ├── QualityReport.jsx # Data quality display grid
│       │   ├── FileDropZone.jsx # CSV/Excel drag-drop upload area (.csv, .xlsx, .xls)
│       │   ├── LoadingSpinner.jsx # Loading state
│       │   └── charts/
│       │       ├── RevenueDecomposition.jsx  # Stacked bar chart (Plotly)
│       │       ├── ContributionDonut.jsx     # Donut chart with baseline (Plotly)
│       │       ├── ContributionBars.jsx      # Horizontal bars with optional error bars (Plotly)
│       │       ├── AdstockCurves.jsx         # Decay curves per channel (Plotly)
│       │       └── SaturationCurves.jsx      # Saturation response curves (Plotly)
│       └── views/
│           ├── DashboardPage.jsx       # Stat strip + action cards + charts + channel table
│           ├── UploadPage.jsx          # CSV/Excel upload + wide/long format docs + quality report
│           ├── ConnectionsPage.jsx     # API connection CRUD + sync + merge
│           ├── ModelRunPage.jsx        # Configure + run MMM + channel selection UI (smart defaults) + previous runs
│           ├── ResultsPage.jsx         # Tabbed: Simple / Intermediate / Advanced
│           ├── SimpleView.jsx          # Contribution bars + confidence labels + recommendations
│           ├── IntermediateView.jsx    # Error bars + adstock/saturation curves + parameter tables
│           ├── AdvancedView.jsx        # Full posteriors + diagnostics + validation + experiments
│           └── ExperimentsPage.jsx     # Lag detection + spend-scaling + product experiment forms
├── pytest.ini                 # Test configuration (slow marker)
└── tests/
    ├── __init__.py
    ├── test_ingest.py         # 37 tests: channel standardisation, CSV reader, quality report, wide format, Excel multi-sheet, full pipeline
    ├── test_model.py          # 25 tests: data prep, model construction, MCMC fitting, result extraction, service, channel filtering/merging
    ├── test_outputs.py        # 40 tests: trust score, confidence labels, simple/intermediate/advanced views, recommendations
    ├── test_validation.py     # 10 tests: holdout validation, posterior predictive check, sensitivity analysis
    ├── test_experiments.py    # 35 tests: lag detection, product experiment (PMax guard), spend-scaling data prep
    └── test_integrations.py   # 42 tests: connectors, registry, sync service, merge pipeline, DB models
```

---

## Key References

- pymc-labs/pymc-marketing — primary MMM library
- Brodersen et al. (2015) — CausalImpact (Annals of Applied Statistics)
- Jin et al. (2017) — Bayesian Methods for Media Mix Modelling with Carryover and Shape Effects (Google Research)
- PyMC documentation — pymc.io