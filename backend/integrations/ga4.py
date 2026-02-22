"""GA4 (Google Analytics 4) connector.

Fetches session data by default channel group, mapping to sessions_organic,
sessions_direct, sessions_email, sessions_referral. Can optionally fetch
revenue and orders from ecommerce events.
"""

from __future__ import annotations

import datetime

import pandas as pd

from backend.integrations.base import (
    BaseConnector,
    ConnectorConfig,
    FetchResult,
    ensure_schema,
)
from backend.integrations.registry import register_connector

# GA4 sessionDefaultChannelGroup → sessions column mapping
CHANNEL_GROUP_MAP = {
    "organic search": "sessions_organic",
    "organic": "sessions_organic",
    "direct": "sessions_direct",
    "email": "sessions_email",
    "referral": "sessions_referral",
}


class GA4Connector(BaseConnector):
    platform = "ga4"

    def authenticate(self) -> None:
        creds = self.config.credentials
        property_id = self.config.config.get("property_id")
        if not property_id:
            raise ConnectionError("Missing property_id in config")

        if "service_account_json" not in creds and "refresh_token" not in creds:
            raise ConnectionError(
                "GA4 requires either service_account_json or refresh_token in credentials"
            )

        try:
            from google.analytics.data_v1beta import BetaAnalyticsDataClient
        except ImportError:
            raise ConnectionError(
                "google-analytics-data package not installed. "
                "Run: pip install google-analytics-data"
            )

        if "service_account_json" in creds:
            self._client = BetaAnalyticsDataClient.from_service_account_info(
                creds["service_account_json"]
            )
        else:
            from google.oauth2.credentials import Credentials
            credentials = Credentials(
                token=None,
                refresh_token=creds["refresh_token"],
                client_id=creds.get("client_id", ""),
                client_secret=creds.get("client_secret", ""),
                token_uri="https://oauth2.googleapis.com/token",
            )
            self._client = BetaAnalyticsDataClient(credentials=credentials)

    def fetch(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> FetchResult:
        self.authenticate()
        property_id = self.config.config["property_id"]

        # Build request using SDK types (imported lazily to allow mocking)
        try:
            from google.analytics.data_v1beta.types import (
                RunReportRequest,
                DateRange,
                Dimension,
                Metric,
            )
            request = RunReportRequest(
                property=f"properties/{property_id}",
                date_ranges=[DateRange(
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                )],
                dimensions=[
                    Dimension(name="date"),
                    Dimension(name="sessionDefaultChannelGroup"),
                ],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="purchaseRevenue"),
                    Metric(name="ecommercePurchases"),
                ],
            )
        except ImportError:
            # Fallback for environments without the SDK (e.g. testing with mocked client)
            request = {
                "property": f"properties/{property_id}",
                "date_ranges": [{"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}],
                "dimensions": [{"name": "date"}, {"name": "sessionDefaultChannelGroup"}],
                "metrics": [{"name": "sessions"}, {"name": "purchaseRevenue"}, {"name": "ecommercePurchases"}],
            }

        response = self._client.run_report(request)

        # Aggregate sessions by date and channel group
        date_data: dict[str, dict] = {}
        warnings = []

        for row in response.rows:
            date_str = row.dimension_values[0].value
            # GA4 returns date as YYYYMMDD
            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            channel_group = row.dimension_values[1].value.lower()
            sessions = int(row.metric_values[0].value)
            revenue = float(row.metric_values[1].value)
            orders = int(float(row.metric_values[2].value))

            if date_formatted not in date_data:
                date_data[date_formatted] = {
                    "date": date_formatted,
                    "channel": "organic_traffic",
                    "campaign": "ga4_sessions",
                    "sessions_organic": 0,
                    "sessions_direct": 0,
                    "sessions_email": 0,
                    "sessions_referral": 0,
                    "revenue": 0.0,
                    "orders": 0,
                }

            col = CHANNEL_GROUP_MAP.get(channel_group)
            if col:
                date_data[date_formatted][col] += sessions

            date_data[date_formatted]["revenue"] += revenue
            date_data[date_formatted]["orders"] += orders

        if not date_data:
            warnings.append("No data returned from GA4 for the given date range.")
            df = pd.DataFrame(columns=["date", "channel", "campaign",
                                        "sessions_organic", "sessions_direct",
                                        "sessions_email", "sessions_referral",
                                        "revenue", "orders"])
        else:
            df = pd.DataFrame(list(date_data.values()))

        df = ensure_schema(df)
        return FetchResult(data=df, warnings=warnings)


register_connector("ga4", GA4Connector)
