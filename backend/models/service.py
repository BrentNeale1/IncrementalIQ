"""Model service: orchestrates data prep → model build → fit → extract → store."""
import json
import logging
import traceback
from sqlalchemy.orm import Session
from backend.db.models import ModelRun, Upload
from backend.models.data_prep import prepare_from_db, PreparedData
from backend.models.mmm import (
    ModelConfig,
    ModelResults,
    build_mmm,
    fit_mmm,
    extract_results,
    save_idata,
)

logger = logging.getLogger(__name__)


def run_model(
    db: Session,
    upload_id: int,
    target: str = "revenue",
    config: ModelConfig | None = None,
) -> tuple[ModelRun, ModelResults | None]:
    """Full pipeline: prepare data, fit model, extract results, store in DB.

    Parameters
    ----------
    db : SQLAlchemy session
    upload_id : ID of the upload to model
    target : "revenue" or "orders"
    config : Optional model configuration overrides

    Returns
    -------
    (ModelRun, ModelResults or None if failed)
    """
    config = config or ModelConfig()

    # Verify upload exists and wasn't rejected
    upload = db.query(Upload).filter_by(id=upload_id).first()
    if not upload:
        raise ValueError(f"Upload {upload_id} not found")
    if upload.status == "rejected":
        raise ValueError(
            f"Upload {upload_id} was rejected due to insufficient data. "
            "At least 9 months of history required."
        )

    # Prepare data
    data = prepare_from_db(db, upload_id, target=target)

    # Create model run record
    model_run = ModelRun(
        upload_id=upload_id,
        status="running",
        target_variable=target,
        channels_used=json.dumps(data.channel_columns),
        config_json=config.to_json(),
    )
    db.add(model_run)
    db.flush()

    try:
        # Build and fit
        mmm = build_mmm(data, config)
        mmm = fit_mmm(mmm, data, config)

        # Extract results
        results = extract_results(mmm, data)

        # Save artifacts
        idata_path = save_idata(mmm, model_run.id)

        # Update DB
        model_run.status = "completed"
        model_run.results_json = results.to_json()
        model_run.diagnostics_json = json.dumps({
            "r_squared_mean": results.diagnostics.r_squared_mean,
            "mape_mean": results.diagnostics.mape_mean,
            "divergences": results.diagnostics.divergences,
            "warnings": results.diagnostics.warnings,
        })
        model_run.idata_path = idata_path
        db.commit()

        logger.info("Model run %d completed successfully", model_run.id)
        return model_run, results

    except Exception as exc:
        model_run.status = "failed"
        model_run.error_message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        db.commit()
        logger.error("Model run %d failed: %s", model_run.id, exc)
        raise
