"""Shopify connector.

Fetches daily revenue and order counts via the Shopify Admin REST API.
Uses httpx for HTTP requests (no SDK dependency).
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


class ShopifyConnector(BaseConnector):
    platform = "shopify"

    def authenticate(self) -> None:
        creds = self.config.credentials
        required = ["access_token"]
        missing = [k for k in required if not creds.get(k)]
        if missing:
            raise ConnectionError(f"Missing Shopify credentials: {missing}")

        shop_domain = self.config.config.get("shop_domain")
        if not shop_domain:
            raise ConnectionError("Missing shop_domain in config")

        try:
            import httpx  # noqa: F401
        except ImportError:
            raise ConnectionError(
                "httpx package not installed. Run: pip install httpx"
            )

    def fetch(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> FetchResult:
        self.authenticate()
        import httpx

        shop_domain = self.config.config["shop_domain"]
        access_token = self.config.credentials["access_token"]
        api_version = self.config.config.get("api_version", "2024-01")

        base_url = f"https://{shop_domain}/admin/api/{api_version}"
        headers = {
            "X-Shopify-Access-Token": access_token,
            "Content-Type": "application/json",
        }

        # Fetch orders within the date range
        all_orders = []
        params = {
            "created_at_min": f"{start_date}T00:00:00Z",
            "created_at_max": f"{end_date}T23:59:59Z",
            "status": "any",
            "limit": 250,
        }

        warnings = []
        url = f"{base_url}/orders.json"

        with httpx.Client(timeout=30) as client:
            while url:
                resp = client.get(url, headers=headers, params=params)
                if resp.status_code != 200:
                    raise ConnectionError(
                        f"Shopify API error {resp.status_code}: {resp.text}"
                    )
                data = resp.json()
                all_orders.extend(data.get("orders", []))

                # Pagination via Link header
                link = resp.headers.get("link", "")
                url = None
                params = None  # only use params on first request
                if 'rel="next"' in link:
                    for part in link.split(","):
                        if 'rel="next"' in part:
                            url = part.split(";")[0].strip().strip("<>")
                            break

        # Aggregate by date
        date_totals: dict[str, dict] = {}
        for order in all_orders:
            created = order.get("created_at", "")[:10]  # YYYY-MM-DD
            if created not in date_totals:
                date_totals[created] = {"revenue": 0.0, "orders": 0}
            date_totals[created]["revenue"] += float(
                order.get("total_price", 0)
            )
            date_totals[created]["orders"] += 1

        if not date_totals:
            warnings.append("No orders returned from Shopify for the given date range.")
            df = pd.DataFrame(columns=["date", "channel", "campaign", "revenue", "orders"])
        else:
            rows = [
                {
                    "date": date,
                    "channel": "ecommerce",
                    "campaign": "shopify_orders",
                    "revenue": totals["revenue"],
                    "orders": totals["orders"],
                }
                for date, totals in sorted(date_totals.items())
            ]
            df = pd.DataFrame(rows)

        df = ensure_schema(df)
        return FetchResult(data=df, warnings=warnings)


register_connector("shopify", ShopifyConnector)
