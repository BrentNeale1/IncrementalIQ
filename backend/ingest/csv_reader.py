from io import BytesIO
import pandas as pd
import pandera.pandas
from backend.ingest.schema import (
    CHANNEL_ALIASES,
    REQUIRED_COLUMNS,
    VALID_CHANNELS,
    ingestion_schema,
)


class ValidationError(Exception):
    def __init__(self, message: str, details: list[str] | None = None):
        self.message = message
        self.details = details or []
        super().__init__(message)


def standardise_channel(name: str) -> str:
    """Map a channel name to the canonical form."""
    normalised = name.strip().lower().replace(" ", "_")
    if normalised in VALID_CHANNELS:
        return normalised
    lookup_key = name.strip().lower()
    if lookup_key in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[lookup_key]
    if normalised in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[normalised]
    return normalised


def read_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read raw CSV bytes into a DataFrame, validate schema, standardise channels.

    Returns the cleaned DataFrame ready for storage.
    Raises ValidationError on missing columns or schema failures.
    """
    df = pd.read_csv(BytesIO(file_bytes))
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(
            "Missing required columns",
            details=[f"Missing: {', '.join(missing)}"],
        )

    df["channel"] = df["channel"].apply(standardise_channel)

    unknown_channels = set(df["channel"].unique()) - set(VALID_CHANNELS)
    unknown_warnings: list[str] = []
    if unknown_channels:
        unknown_warnings = [
            f"Unknown channel '{ch}' — will be stored but may not be modelled"
            for ch in sorted(unknown_channels)
        ]

    try:
        df = ingestion_schema.validate(df, lazy=True)
    except pandera.pandas.errors.SchemaErrors as exc:
        details = []
        for _, row in exc.failure_cases.iterrows():
            details.append(
                f"Column '{row.get('column', '?')}': {row.get('check', '?')} "
                f"(index {row.get('index', '?')})"
            )
        raise ValidationError("Schema validation failed", details=details) from exc

    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df, unknown_warnings
