"""Meta Marketing API connector.

Fetches ad-set level spend, clicks, and conversions from
Facebook/Instagram campaigns. Maps publisher platforms to standard channels.
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

# Meta publisher_platform → standard channel name
PLATFORM_MAP = {
    "facebook": "meta_feed",
    "instagram": "meta_instagram",
}

# Placement-level override for stories
STORIES_POSITIONS = {"story", "stories", "instagram_stories", "facebook_stories"}


class MetaAdsConnector(BaseConnector):
    platform = "meta"

    def authenticate(self) -> None:
        creds = self.config.credentials
        required = ["access_token"]
        missing = [k for k in required if not creds.get(k)]
        if missing:
            raise ConnectionError(f"Missing Meta credentials: {missing}")

        ad_account_id = self.config.config.get("ad_account_id")
        if not ad_account_id:
            raise ConnectionError("Missing ad_account_id in config")

        try:
            from facebook_business.api import FacebookAdsApi
            from facebook_business.adobjects.adaccount import AdAccount
        except ImportError:
            raise ConnectionError(
                "facebook-business package not installed. Run: pip install facebook-business"
            )

        FacebookAdsApi.init(
            app_id=creds.get("app_id", ""),
            app_secret=creds.get("app_secret", ""),
            access_token=creds["access_token"],
        )
        self._account = AdAccount(f"act_{ad_account_id}")

    def fetch(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> FetchResult:
        self.authenticate()

        params = {
            "time_range": {
                "since": start_date.isoformat(),
                "until": end_date.isoformat(),
            },
            "time_increment": 1,  # daily
            "level": "campaign",
            "breakdowns": ["publisher_platform"],
        }
        fields = [
            "date_start",
            "campaign_name",
            "publisher_platform",
            "spend",
            "clicks",
            "actions",
        ]

        insights = self._account.get_insights(params=params, fields=fields)

        rows = []
        warnings = []
        for insight in insights:
            publisher = (insight.get("publisher_platform") or "facebook").lower()
            channel = PLATFORM_MAP.get(publisher, "meta_feed")

            conversions = 0.0
            actions = insight.get("actions") or []
            for action in actions:
                if action.get("action_type") in (
                    "offsite_conversion.fb_pixel_purchase",
                    "purchase",
                    "omni_purchase",
                ):
                    conversions += float(action.get("value", 0))

            rows.append({
                "date": insight.get("date_start"),
                "channel": channel,
                "campaign": insight.get("campaign_name", ""),
                "spend": float(insight.get("spend", 0)),
                "clicks": int(insight.get("clicks", 0)),
                "in_platform_conversions": conversions,
            })

        if not rows:
            warnings.append("No data returned from Meta for the given date range.")
            df = pd.DataFrame(columns=["date", "channel", "campaign", "spend",
                                        "clicks", "in_platform_conversions"])
        else:
            df = pd.DataFrame(rows)

        df = ensure_schema(df)
        return FetchResult(data=df, warnings=warnings)


register_connector("meta", MetaAdsConnector)
