"""Base connector ABC and shared types for API integrations."""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from backend.ingest.schema import REQUIRED_COLUMNS


@dataclass
class ConnectorConfig:
    """Credentials and settings for an API connector."""
    platform: str
    credentials: dict
    config: dict  # account IDs, property IDs, etc.


@dataclass
class FetchResult:
    """Result of a connector fetch operation."""
    data: pd.DataFrame  # 13-column standard schema
    warnings: list[str] = field(default_factory=list)
    rows_fetched: int = 0

    def __post_init__(self):
        self.rows_fetched = len(self.data)


ZERO_FILL_COLUMNS = {
    "spend": 0.0,
    "impressions": 0,
    "clicks": 0,
    "in_platform_conversions": 0.0,
    "revenue": 0.0,
    "orders": 0,
    "sessions_organic": 0,
    "sessions_direct": 0,
    "sessions_email": 0,
    "sessions_referral": 0,
}


def build_empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the standard 13-column schema."""
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has all 13 required columns, zero-filling missing ones."""
    for col, default in ZERO_FILL_COLUMNS.items():
        if col not in df.columns:
            df[col] = default
    # Enforce column order
    return df[REQUIRED_COLUMNS]


class BaseConnector(ABC):
    """Abstract base class for all API connectors."""

    platform: str = ""

    def __init__(self, config: ConnectorConfig):
        self.config = config
        self._client = None

    @abstractmethod
    def authenticate(self) -> None:
        """Initialise the API client with stored credentials.

        Raises ConnectionError if credentials are invalid or missing.
        """

    @abstractmethod
    def fetch(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> FetchResult:
        """Fetch data for the given date range and transform to 13-column schema.

        Returns FetchResult with the DataFrame and any warnings.
        Raises ConnectionError or ValueError on failure.
        """

    def test_connection(self) -> bool:
        """Verify that credentials are valid and the API is reachable.

        Returns True on success, False on failure.
        """
        try:
            self.authenticate()
            return True
        except Exception:
            return False
