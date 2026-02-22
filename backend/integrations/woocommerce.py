"""WooCommerce connector.

Fetches daily revenue and order counts via the WooCommerce REST API.
Uses httpx for HTTP requests with HMAC authentication.
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


class WooCommerceConnector(BaseConnector):
    platform = "woocommerce"

    def authenticate(self) -> None:
        creds = self.config.credentials
        required = ["consumer_key", "consumer_secret"]
        missing = [k for k in required if not creds.get(k)]
        if missing:
            raise ConnectionError(f"Missing WooCommerce credentials: {missing}")

        store_url = self.config.config.get("store_url")
        if not store_url:
            raise ConnectionError("Missing store_url in config")

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

        store_url = self.config.config["store_url"].rstrip("/")
        consumer_key = self.config.credentials["consumer_key"]
        consumer_secret = self.config.credentials["consumer_secret"]

        all_orders = []
        page = 1
        warnings = []

        with httpx.Client(timeout=30) as client:
            while True:
                resp = client.get(
                    f"{store_url}/wp-json/wc/v3/orders",
                    params={
                        "after": f"{start_date}T00:00:00Z",
                        "before": f"{end_date}T23:59:59Z",
                        "per_page": 100,
                        "page": page,
                        "status": "completed,processing",
                    },
                    auth=(consumer_key, consumer_secret),
                )
                if resp.status_code != 200:
                    raise ConnectionError(
                        f"WooCommerce API error {resp.status_code}: {resp.text}"
                    )
                orders = resp.json()
                if not orders:
                    break
                all_orders.extend(orders)
                page += 1

        # Aggregate by date
        date_totals: dict[str, dict] = {}
        for order in all_orders:
            created = order.get("date_created", "")[:10]
            if created not in date_totals:
                date_totals[created] = {"revenue": 0.0, "orders": 0}
            date_totals[created]["revenue"] += float(order.get("total", 0))
            date_totals[created]["orders"] += 1

        if not date_totals:
            warnings.append("No orders returned from WooCommerce for the given date range.")
            df = pd.DataFrame(columns=["date", "channel", "campaign", "revenue", "orders"])
        else:
            rows = [
                {
                    "date": date,
                    "channel": "ecommerce",
                    "campaign": "woocommerce_orders",
                    "revenue": totals["revenue"],
                    "orders": totals["orders"],
                }
                for date, totals in sorted(date_totals.items())
            ]
            df = pd.DataFrame(rows)

        df = ensure_schema(df)
        return FetchResult(data=df, warnings=warnings)


register_connector("woocommerce", WooCommerceConnector)
