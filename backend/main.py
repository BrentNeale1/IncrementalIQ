import io
import json
from dataclasses import asdict
from fastapi import FastAPI, UploadFile, Depends, HTTPException
from pydantic import BaseModel
import pandas as pd
from sqlalchemy.orm import Session
from backend.db.config import get_db, init_db
from backend.db.models import Upload, QualityReport, ModelRun, ExperimentResult, ApiConnection, ApiSync
from backend.ingest.csv_reader import ValidationError
from backend.ingest.service import ingest_csv
from backend.integrations.service import sync_connection, merge_sources
from backend.integrations.registry import get_connector, list_platforms
from backend.integrations.base import ConnectorConfig
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="IncrementIQ", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                    "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelRunRequest(BaseModel):
    upload_id: int
    target: str = "revenue"
    adstock_l_max: int = 8
    yearly_seasonality: int = 6
    chains: int = 4
    tune: int = 1500
    draws: int = 1000
    target_accept: float = 0.9
    channel_config: dict | None = None


@app.on_event("startup")
def on_startup():
    init_db()


@app.post("/api/upload")
async def upload_csv(file: UploadFile, db: Session = Depends(get_db)):
    """Upload a CSV or Excel file for ingestion, validation, and quality assessment."""
    allowed_extensions = (".csv", ".xlsx", ".xls")
    if not file.filename or not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Only .csv and .xlsx files are accepted.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = ingest_csv(db, file.filename, contents)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": exc.message, "errors": exc.details},
        )

    return {
        "upload_id": result.upload_id,
        "status": result.status,
        "rows_stored": result.rows_stored,
        "quality": {
            "history_days": result.quality.history_days,
            "history_months": result.quality.history_months,
            "history_status": result.quality.history_status,
            "date_range": f"{result.quality.date_range_start} to {result.quality.date_range_end}",
            "channels": [asdict(c) for c in result.quality.channels],
            "gap_count": result.quality.gap_count,
            "spike_count": result.quality.spike_count,
            "low_variance_channels": result.quality.low_variance_channels,
        },
        "warnings": result.warnings,
    }


@app.get("/api/uploads")
def list_uploads(db: Session = Depends(get_db)):
    """List all previous uploads."""
    uploads = db.query(Upload).order_by(Upload.uploaded_at.desc()).all()
    return [
        {
            "id": u.id,
            "filename": u.filename,
            "uploaded_at": u.uploaded_at.isoformat(),
            "row_count": u.row_count,
            "status": u.status,
        }
        for u in uploads
    ]


@app.get("/api/uploads/{upload_id}/quality")
def get_quality_report(upload_id: int, db: Session = Depends(get_db)):
    """Retrieve the data quality report for a specific upload."""
    report = db.query(QualityReport).filter_by(upload_id=upload_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Quality report not found.")
    return json.loads(report.report_json)


@app.get("/api/uploads/{upload_id}/channels")
def get_channel_analysis(upload_id: int, db: Session = Depends(get_db)):
    """Analyze channels in an upload and return a recommended config for model fitting."""
    from backend.models.data_prep import query_records, recommend_channel_config

    upload = db.query(Upload).filter_by(id=upload_id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found.")

    try:
        df = query_records(db, upload_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    rec = recommend_channel_config(df)

    return {
        "channels": rec["channels_detail"],
        "recommended_config": {
            "channels": rec["channels"],
            "merge": rec["merge"],
            "dropped": rec["dropped"],
            "reasons": rec["reasons"],
        },
    }


# --- Model endpoints ---

@app.post("/api/model/run")
def start_model_run(req: ModelRunRequest, db: Session = Depends(get_db)):
    """Fit a Bayesian MMM on an uploaded dataset.

    This is a synchronous endpoint — the model will be fitted before returning.
    For large datasets, consider running asynchronously.
    """
    from backend.models.mmm import ModelConfig
    from backend.models.service import run_model

    config = ModelConfig(
        adstock_l_max=req.adstock_l_max,
        yearly_seasonality=req.yearly_seasonality,
        chains=req.chains,
        tune=req.tune,
        draws=req.draws,
        target_accept=req.target_accept,
    )

    try:
        model_run, results = run_model(
            db, req.upload_id, target=req.target, config=config,
            channel_config=req.channel_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model fitting failed: {exc}")

    response = {
        "run_id": model_run.id,
        "status": model_run.status,
        "target_variable": model_run.target_variable,
    }

    if results:
        response["channel_contributions"] = [
            asdict(cp) for cp in results.channel_posteriors
        ]
        response["baseline_contribution_pct"] = results.baseline_contribution_pct
        response["diagnostics"] = asdict(results.diagnostics)

    return response


@app.get("/api/model/runs")
def list_model_runs(db: Session = Depends(get_db)):
    """List all model runs."""
    runs = db.query(ModelRun).order_by(ModelRun.created_at.desc()).all()
    return [
        {
            "id": r.id,
            "upload_id": r.upload_id,
            "status": r.status,
            "target_variable": r.target_variable,
            "created_at": r.created_at.isoformat(),
        }
        for r in runs
    ]


@app.get("/api/model/runs/{run_id}")
def get_model_run(run_id: int, db: Session = Depends(get_db)):
    """Get full results for a specific model run."""
    run = db.query(ModelRun).filter_by(id=run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found.")

    response = {
        "id": run.id,
        "upload_id": run.upload_id,
        "status": run.status,
        "target_variable": run.target_variable,
        "created_at": run.created_at.isoformat(),
        "channels_used": json.loads(run.channels_used),
        "config": json.loads(run.config_json),
    }

    if run.results_json:
        response["results"] = json.loads(run.results_json)
    if run.diagnostics_json:
        response["diagnostics"] = json.loads(run.diagnostics_json)
    if run.error_message:
        response["error"] = run.error_message

    return response


# --- Output view endpoints ---

@app.get("/api/model/runs/{run_id}/simple")
def get_simple_view(run_id: int, db: Session = Depends(get_db)):
    """Simple output view — plain-language channel contributions with confidence labels."""
    from backend.models.mmm import ModelResults, ChannelPosterior, ModelDiagnostics
    from backend.outputs.trust_score import compute_trust_score
    from backend.outputs.views import build_simple_view

    run = db.query(ModelRun).filter_by(id=run_id).first()
    if not run or not run.results_json:
        raise HTTPException(status_code=404, detail="Model run results not found.")

    results_data = json.loads(run.results_json)
    diagnostics_data = json.loads(run.diagnostics_json) if run.diagnostics_json else {}

    # Reconstruct ModelResults from stored JSON
    channel_posteriors = [ChannelPosterior(**cp) for cp in results_data["channel_posteriors"]]
    diagnostics = ModelDiagnostics(**results_data["diagnostics"])
    results = ModelResults(
        channel_posteriors=channel_posteriors,
        baseline_contribution_pct=results_data["baseline_contribution_pct"],
        diagnostics=diagnostics,
    )

    # Get quality data for trust score
    quality_report = db.query(QualityReport).filter_by(upload_id=run.upload_id).first()
    qr = json.loads(quality_report.report_json) if quality_report else {}

    trust = compute_trust_score(
        history_status=qr.get("history_status", "caution"),
        gap_count=qr.get("gap_count", 0),
        total_days=qr.get("history_days", 0),
        low_variance_channels=qr.get("low_variance_channels", []),
        total_channels=len(channel_posteriors),
        mape_mean=diagnostics.mape_mean,
        r_squared_mean=diagnostics.r_squared_mean,
        divergences=diagnostics.divergences,
        total_draws=json.loads(run.config_json).get("draws", 1000) * json.loads(run.config_json).get("chains", 4),
    )

    view = build_simple_view(results, trust)
    return view.to_dict()


@app.get("/api/model/runs/{run_id}/intermediate")
def get_intermediate_view(run_id: int, db: Session = Depends(get_db)):
    """Intermediate view — confidence intervals, adstock/saturation curves."""
    from backend.models.mmm import ModelResults, ChannelPosterior, ModelDiagnostics
    from backend.outputs.trust_score import compute_trust_score
    from backend.outputs.views import build_intermediate_view

    run = db.query(ModelRun).filter_by(id=run_id).first()
    if not run or not run.results_json:
        raise HTTPException(status_code=404, detail="Model run results not found.")

    results_data = json.loads(run.results_json)
    channel_posteriors = [ChannelPosterior(**cp) for cp in results_data["channel_posteriors"]]
    diagnostics = ModelDiagnostics(**results_data["diagnostics"])
    results = ModelResults(
        channel_posteriors=channel_posteriors,
        baseline_contribution_pct=results_data["baseline_contribution_pct"],
        diagnostics=diagnostics,
    )

    quality_report = db.query(QualityReport).filter_by(upload_id=run.upload_id).first()
    qr = json.loads(quality_report.report_json) if quality_report else {}
    config_data = json.loads(run.config_json)

    trust = compute_trust_score(
        history_status=qr.get("history_status", "caution"),
        gap_count=qr.get("gap_count", 0),
        total_days=qr.get("history_days", 0),
        low_variance_channels=qr.get("low_variance_channels", []),
        total_channels=len(channel_posteriors),
        mape_mean=diagnostics.mape_mean,
        r_squared_mean=diagnostics.r_squared_mean,
        divergences=diagnostics.divergences,
        total_draws=config_data.get("draws", 1000) * config_data.get("chains", 4),
    )

    view = build_intermediate_view(
        results, trust,
        l_max=config_data.get("adstock_l_max", 8),
    )
    return view.to_dict()


@app.get("/api/model/runs/{run_id}/advanced")
def get_advanced_view(run_id: int, db: Session = Depends(get_db)):
    """Advanced view — full posterior distributions, diagnostics, validation, experiments."""
    from backend.models.mmm import ModelResults, ChannelPosterior, ModelDiagnostics
    from backend.outputs.trust_score import compute_trust_score
    from backend.outputs.views import build_advanced_view

    run = db.query(ModelRun).filter_by(id=run_id).first()
    if not run or not run.results_json:
        raise HTTPException(status_code=404, detail="Model run results not found.")

    results_data = json.loads(run.results_json)
    channel_posteriors = [ChannelPosterior(**cp) for cp in results_data["channel_posteriors"]]
    diagnostics = ModelDiagnostics(**results_data["diagnostics"])
    results = ModelResults(
        channel_posteriors=channel_posteriors,
        baseline_contribution_pct=results_data["baseline_contribution_pct"],
        diagnostics=diagnostics,
    )

    quality_report = db.query(QualityReport).filter_by(upload_id=run.upload_id).first()
    qr = json.loads(quality_report.report_json) if quality_report else {}
    config_data = json.loads(run.config_json)

    total_draws = config_data.get("draws", 1000) * config_data.get("chains", 4)

    trust = compute_trust_score(
        history_status=qr.get("history_status", "caution"),
        gap_count=qr.get("gap_count", 0),
        total_days=qr.get("history_days", 0),
        low_variance_channels=qr.get("low_variance_channels", []),
        total_channels=len(channel_posteriors),
        mape_mean=diagnostics.mape_mean,
        r_squared_mean=diagnostics.r_squared_mean,
        divergences=diagnostics.divergences,
        total_draws=total_draws,
    )

    # Load validation report if it was previously run
    validation = json.loads(run.validation_json) if run.validation_json else None

    # Load experiment results linked to this upload
    experiment_records = (
        db.query(ExperimentResult)
        .filter_by(upload_id=run.upload_id)
        .order_by(ExperimentResult.created_at.desc())
        .all()
    )
    experiments = [json.loads(er.result_json) for er in experiment_records]

    view = build_advanced_view(
        results, trust,
        total_draws=total_draws,
        validation=validation,
        experiments=experiments,
    )
    return view.to_dict()


# --- Validation endpoints ---

class ValidationRequest(BaseModel):
    run_id: int
    holdout_fraction: float = 0.2
    run_sensitivity: bool = False


@app.post("/api/model/validate")
def validate_model(req: ValidationRequest, db: Session = Depends(get_db)):
    """Run holdout validation and posterior predictive checks on a fitted model."""
    from backend.models.mmm import ModelConfig, ModelResults, ChannelPosterior, ModelDiagnostics
    from backend.models.data_prep import prepare_from_db
    from backend.models.validation import (
        run_holdout_validation,
        run_posterior_predictive_check,
        run_sensitivity_analysis,
        ValidationReport,
    )

    run = db.query(ModelRun).filter_by(id=req.run_id).first()
    if not run or run.status != "completed":
        raise HTTPException(status_code=400, detail="Model run not found or not completed.")

    config_data = json.loads(run.config_json)
    config = ModelConfig(**config_data)
    data = prepare_from_db(db, run.upload_id, target=run.target_variable)

    report = ValidationReport()

    try:
        report.holdout = run_holdout_validation(data, config, req.holdout_fraction)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Holdout validation failed: {exc}")

    if req.run_sensitivity:
        results_data = json.loads(run.results_json)
        base_betas = {
            cp["channel"]: cp["beta_mean"]
            for cp in results_data["channel_posteriors"]
        }
        try:
            report.sensitivity = run_sensitivity_analysis(data, base_betas, config)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Sensitivity analysis failed: {exc}")

    # Persist validation results on the model run
    run.validation_json = report.to_json()
    db.commit()

    return report.to_dict()


# --- Experiment endpoints (Phase 3) ---

class LagDetectionRequest(BaseModel):
    upload_id: int
    target: str = "revenue"
    max_lag: int = 21


class SpendScalingRequest(BaseModel):
    upload_id: int
    target_channel: str
    pre_period_start: str
    pre_period_end: str
    post_period_start: str
    post_period_end: str
    outcome_column: str = "revenue"
    alpha: float = 0.05
    mmm_predicted_impact: float | None = None


class ProductExperimentRequest(BaseModel):
    advertised_products: list[str]
    campaign_name: str
    campaign_type: str
    bidding_strategy: str


@app.post("/api/experiments/lag-detection")
def run_lag_detection(req: LagDetectionRequest, db: Session = Depends(get_db)):
    """Detect optimal lag (days) between each channel's spend and outcome."""
    from backend.models.data_prep import prepare_from_db
    from backend.experiments.lag_detection import detect_lags_from_prepared_data

    try:
        data = prepare_from_db(db, req.upload_id, target=req.target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    report = detect_lags_from_prepared_data(
        data.X, data.y,
        channel_columns=data.channel_columns,
        max_lag=req.max_lag,
    )
    return report.to_dict()


@app.post("/api/experiments/spend-scaling")
def run_spend_scaling(req: SpendScalingRequest, db: Session = Depends(get_db)):
    """Run a spend-scaling experiment via CausalImpact."""
    from backend.models.data_prep import prepare_from_db
    from backend.experiments.spend_scaling import run_spend_scaling_experiment

    try:
        data = prepare_from_db(db, req.upload_id, target=req.outcome_column)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Reconstruct a daily DataFrame with date, outcome, and channel spend columns
    daily_df = data.X.copy()
    daily_df[data.y.name] = data.y.values
    daily_df["date"] = daily_df.index

    try:
        result = run_spend_scaling_experiment(
            daily_data=daily_df,
            target_channel=req.target_channel,
            pre_period=(req.pre_period_start, req.pre_period_end),
            post_period=(req.post_period_start, req.post_period_end),
            outcome_column=req.outcome_column,
            channel_columns=data.channel_columns,
            alpha=req.alpha,
            mmm_predicted_impact=req.mmm_predicted_impact,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CausalImpact analysis failed: {exc}")

    # Persist experiment result
    experiment_record = ExperimentResult(
        upload_id=req.upload_id,
        experiment_type="spend_scaling",
        channel=req.target_channel,
        result_json=result.to_json(),
    )
    db.add(experiment_record)
    db.commit()

    response = result.to_dict()
    response["experiment_id"] = experiment_record.id
    return response


@app.post("/api/experiments/product")
def run_product_exp(req: ProductExperimentRequest, file: UploadFile = None, db: Session = Depends(get_db)):
    """Run a product-level experiment (Google Ads only).

    Upload a CSV with columns: product_id, revenue, orders.
    HARD CONSTRAINT: PMax and auto-bidding campaigns are refused.
    """
    from backend.experiments.product_experiment import (
        run_product_experiment,
        ProductExperimentError,
    )

    if file is None:
        raise HTTPException(status_code=400, detail="Product data CSV file is required.")

    contents = file.file.read()
    try:
        product_df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}")

    try:
        result = run_product_experiment(
            product_data=product_df,
            advertised_products=req.advertised_products,
            campaign_name=req.campaign_name,
            campaign_type=req.campaign_type,
            bidding_strategy=req.bidding_strategy,
        )
    except ProductExperimentError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return result.to_dict()


# --- API Integration endpoints (Phase 1B) ---

class ConnectionCreateRequest(BaseModel):
    platform: str
    display_name: str
    credentials: dict
    config: dict


class SyncRequest(BaseModel):
    connection_id: int
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD


class MergeRequest(BaseModel):
    upload_ids: list[int]


@app.post("/api/connections")
def create_connection(req: ConnectionCreateRequest, db: Session = Depends(get_db)):
    """Register a new API connection."""
    available = list_platforms()
    if req.platform not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown platform '{req.platform}'. Available: {available}",
        )

    conn = ApiConnection(
        platform=req.platform,
        display_name=req.display_name,
        credentials_json=json.dumps(req.credentials),
        config_json=json.dumps(req.config),
        is_active=True,
    )
    db.add(conn)
    db.commit()

    return {
        "id": conn.id,
        "platform": conn.platform,
        "display_name": conn.display_name,
        "is_active": conn.is_active,
        "created_at": conn.created_at.isoformat(),
    }


@app.get("/api/connections")
def list_connections(db: Session = Depends(get_db)):
    """List all API connections (credentials redacted)."""
    connections = db.query(ApiConnection).order_by(ApiConnection.created_at.desc()).all()
    return [
        {
            "id": c.id,
            "platform": c.platform,
            "display_name": c.display_name,
            "is_active": c.is_active,
            "created_at": c.created_at.isoformat(),
            "last_sync_status": c.last_sync_status,
            "config": json.loads(c.config_json),
        }
        for c in connections
    ]


@app.post("/api/connections/{connection_id}/test")
def test_connection(connection_id: int, db: Session = Depends(get_db)):
    """Test connectivity for an API connection."""
    conn = db.query(ApiConnection).filter_by(id=connection_id).first()
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found.")

    config = ConnectorConfig(
        platform=conn.platform,
        credentials=json.loads(conn.credentials_json),
        config=json.loads(conn.config_json),
    )

    try:
        connector = get_connector(config)
        success = connector.test_connection()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"connection_id": connection_id, "success": success}


@app.delete("/api/connections/{connection_id}")
def deactivate_connection(connection_id: int, db: Session = Depends(get_db)):
    """Deactivate an API connection."""
    conn = db.query(ApiConnection).filter_by(id=connection_id).first()
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found.")

    conn.is_active = False
    db.commit()
    return {"connection_id": connection_id, "is_active": False}


@app.post("/api/sync")
def run_sync(req: SyncRequest, db: Session = Depends(get_db)):
    """Run a data sync for an API connection over a date range."""
    import datetime as dt
    try:
        start = dt.date.fromisoformat(req.start_date)
        end = dt.date.fromisoformat(req.end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    if start > end:
        raise HTTPException(status_code=400, detail="start_date must be <= end_date.")

    try:
        sync = sync_connection(db, req.connection_id, start, end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {
        "sync_id": sync.id,
        "connection_id": sync.connection_id,
        "upload_id": sync.upload_id,
        "status": sync.status,
        "rows_fetched": sync.rows_fetched,
        "date_range": f"{sync.date_range_start} to {sync.date_range_end}",
    }


@app.get("/api/syncs")
def list_syncs(db: Session = Depends(get_db)):
    """List sync history."""
    syncs = db.query(ApiSync).order_by(ApiSync.started_at.desc()).all()
    return [
        {
            "id": s.id,
            "connection_id": s.connection_id,
            "upload_id": s.upload_id,
            "status": s.status,
            "rows_fetched": s.rows_fetched,
            "date_range": f"{s.date_range_start} to {s.date_range_end}",
            "started_at": s.started_at.isoformat(),
            "completed_at": s.completed_at.isoformat() if s.completed_at else None,
            "error_message": s.error_message,
        }
        for s in syncs
    ]


@app.post("/api/merge")
def merge_uploads(req: MergeRequest, db: Session = Depends(get_db)):
    """Merge multiple uploads into a unified dataset."""
    try:
        upload = merge_sources(db, req.upload_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Load quality report
    qr = db.query(QualityReport).filter_by(upload_id=upload.id).first()
    quality_data = json.loads(qr.report_json) if qr else {}

    return {
        "upload_id": upload.id,
        "status": upload.status,
        "rows_stored": upload.row_count,
        "source_uploads": req.upload_ids,
        "quality": quality_data,
    }
