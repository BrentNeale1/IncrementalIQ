"""Google Ads API connector.

Fetches campaign-level spend, clicks, and conversions.
Maps Google Ads campaign types to standard channel names.
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

# Google Ads advertising_channel_type → standard channel name
CHANNEL_TYPE_MAP = {
    "SEARCH": "google_search",
    "SHOPPING": "google_shopping",
    "VIDEO": "google_youtube",
    "PERFORMANCE_MAX": "google_pmax",
}


class GoogleAdsConnector(BaseConnector):
    platform = "google_ads"

    def authenticate(self) -> None:
        creds = self.config.credentials
        required = ["developer_token", "client_id", "client_secret", "refresh_token"]
        missing = [k for k in required if not creds.get(k)]
        if missing:
            raise ConnectionError(f"Missing Google Ads credentials: {missing}")

        customer_id = self.config.config.get("customer_id")
        if not customer_id:
            raise ConnectionError("Missing customer_id in config")

        try:
            from google.ads.googleads.client import GoogleAdsClient
        except ImportError:
            raise ConnectionError(
                "google-ads package not installed. Run: pip install google-ads"
            )

        self._client = GoogleAdsClient.load_from_dict({
            "developer_token": creds["developer_token"],
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
            "refresh_token": creds["refresh_token"],
            "login_customer_id": customer_id,
            "use_proto_plus": True,
        })

    def fetch(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> FetchResult:
        self.authenticate()
        customer_id = self.config.config["customer_id"]

        ga_service = self._client.get_service("GoogleAdsService")

        query = f"""
            SELECT
                segments.date,
                campaign.name,
                campaign.advertising_channel_type,
                metrics.cost_micros,
                metrics.clicks,
                metrics.conversions
            FROM campaign
            WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY segments.date
        """

        rows = []
        warnings = []
        response = ga_service.search_stream(customer_id=customer_id, query=query)

        for batch in response:
            for row in batch.results:
                channel_type = row.campaign.advertising_channel_type.name
                channel = CHANNEL_TYPE_MAP.get(channel_type)
                if channel is None:
                    warnings.append(
                        f"Unmapped Google Ads channel type: {channel_type}"
                    )
                    channel = f"google_{channel_type.lower()}"

                rows.append({
                    "date": row.segments.date,
                    "channel": channel,
                    "campaign": row.campaign.name,
                    "spend": row.metrics.cost_micros / 1_000_000,
                    "clicks": row.metrics.clicks,
                    "in_platform_conversions": row.metrics.conversions,
                })

        if not rows:
            warnings.append("No data returned from Google Ads for the given date range.")
            df = pd.DataFrame(columns=["date", "channel", "campaign", "spend",
                                        "clicks", "in_platform_conversions"])
        else:
            df = pd.DataFrame(rows)

        df = ensure_schema(df)
        return FetchResult(data=df, warnings=warnings)


register_connector("google_ads", GoogleAdsConnector)
