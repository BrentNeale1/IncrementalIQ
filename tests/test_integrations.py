"""Tests for Phase 1B API integrations.

All connector tests use mocked SDK responses — no live API calls.
Tests verify correct 13-column schema output, channel mapping, zero-fill,
authentication error handling, merge logic, and pandera schema validation.
"""

import datetime
import json
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.config import Base
from backend.db.models import (
    ApiConnection,
    ApiSync,
    DailyRecord,
    QualityReport,
    Upload,
)
from backend.ingest.schema import REQUIRED_COLUMNS
from backend.integrations.base import (
    BaseConnector,
    ConnectorConfig,
    FetchResult,
    ensure_schema,
    build_empty_dataframe,
)
from backend.integrations.registry import get_connector, list_platforms, _REGISTRY
from backend.integrations.service import sync_connection, merge_sources


# ---- fixtures ----

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _make_config(platform, credentials=None, config=None):
    return ConnectorConfig(
        platform=platform,
        credentials=credentials or {},
        config=config or {},
    )


# ---- Base / utility tests ----

class TestBaseInfrastructure:
    def test_build_empty_dataframe_has_all_columns(self):
        df = build_empty_dataframe()
        assert list(df.columns) == REQUIRED_COLUMNS

    def test_ensure_schema_fills_missing_columns(self):
        df = pd.DataFrame({
            "date": ["2024-01-01"],
            "channel": ["google_search"],
            "campaign": ["test"],
            "spend": [100.0],
        })
        result = ensure_schema(df)
        assert list(result.columns) == REQUIRED_COLUMNS
        assert result["clicks"].iloc[0] == 0
        assert result["revenue"].iloc[0] == 0.0
        assert result["sessions_organic"].iloc[0] == 0

    def test_ensure_schema_preserves_existing_values(self):
        df = pd.DataFrame({
            "date": ["2024-01-01"],
            "channel": ["google_search"],
            "campaign": ["test"],
            "spend": [100.0],
            "clicks": [200],
            "in_platform_conversions": [10.0],
            "revenue": [500.0],
            "orders": [5],
            "sessions_organic": [300],
            "sessions_direct": [150],
            "sessions_email": [50],
            "sessions_referral": [25],
        })
        result = ensure_schema(df)
        assert result["spend"].iloc[0] == 100.0
        assert result["clicks"].iloc[0] == 200

    def test_fetch_result_rows_fetched(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = FetchResult(data=df)
        assert result.rows_fetched == 3

    def test_fetch_result_empty(self):
        df = pd.DataFrame()
        result = FetchResult(data=df)
        assert result.rows_fetched == 0


class TestRegistry:
    def test_list_platforms_returns_all_five(self):
        platforms = list_platforms()
        assert "google_ads" in platforms
        assert "meta" in platforms
        assert "ga4" in platforms
        assert "shopify" in platforms
        assert "woocommerce" in platforms

    def test_get_connector_unknown_platform_raises(self):
        config = _make_config("unknown_platform")
        with pytest.raises(ValueError, match="Unknown platform"):
            get_connector(config)

    def test_get_connector_returns_correct_type(self):
        from backend.integrations.google_ads import GoogleAdsConnector
        config = _make_config("google_ads", credentials={}, config={})
        connector = get_connector(config)
        assert isinstance(connector, GoogleAdsConnector)


# ---- Google Ads connector tests ----

class TestGoogleAdsConnector:
    def test_authenticate_missing_credentials(self):
        from backend.integrations.google_ads import GoogleAdsConnector
        config = _make_config("google_ads", credentials={}, config={})
        connector = GoogleAdsConnector(config)
        with pytest.raises(ConnectionError, match="Missing Google Ads credentials"):
            connector.authenticate()

    def test_authenticate_missing_customer_id(self):
        from backend.integrations.google_ads import GoogleAdsConnector
        config = _make_config("google_ads", credentials={
            "developer_token": "tok",
            "client_id": "cid",
            "client_secret": "cs",
            "refresh_token": "rt",
        }, config={})
        connector = GoogleAdsConnector(config)
        with pytest.raises(ConnectionError, match="Missing customer_id"):
            connector.authenticate()

    @patch("backend.integrations.google_ads.GoogleAdsConnector.authenticate")
    def test_fetch_transforms_to_schema(self, mock_auth):
        from backend.integrations.google_ads import GoogleAdsConnector

        # Create mock response
        mock_row = MagicMock()
        mock_row.segments.date = "2024-06-01"
        mock_row.campaign.name = "Brand Search"
        mock_row.campaign.advertising_channel_type.name = "SEARCH"
        mock_row.metrics.cost_micros = 50_000_000  # $50
        mock_row.metrics.clicks = 500
        mock_row.metrics.conversions = 25.0

        mock_batch = MagicMock()
        mock_batch.results = [mock_row]

        config = _make_config("google_ads", credentials={
            "developer_token": "tok", "client_id": "cid",
            "client_secret": "cs", "refresh_token": "rt",
        }, config={"customer_id": "123"})
        connector = GoogleAdsConnector(config)

        mock_service = MagicMock()
        mock_service.search_stream.return_value = [mock_batch]
        connector._client = MagicMock()
        connector._client.get_service.return_value = mock_service

        result = connector.fetch(datetime.date(2024, 6, 1), datetime.date(2024, 6, 30))

        assert list(result.data.columns) == REQUIRED_COLUMNS
        assert len(result.data) == 1
        assert result.data["channel"].iloc[0] == "google_search"
        assert result.data["spend"].iloc[0] == 50.0
        assert result.data["revenue"].iloc[0] == 0.0  # zero-filled
        assert result.data["sessions_organic"].iloc[0] == 0  # zero-filled

    def test_channel_type_mapping(self):
        from backend.integrations.google_ads import CHANNEL_TYPE_MAP
        assert CHANNEL_TYPE_MAP["SEARCH"] == "google_search"
        assert CHANNEL_TYPE_MAP["SHOPPING"] == "google_shopping"
        assert CHANNEL_TYPE_MAP["VIDEO"] == "google_youtube"
        assert CHANNEL_TYPE_MAP["PERFORMANCE_MAX"] == "google_pmax"

    @patch("backend.integrations.google_ads.GoogleAdsConnector.authenticate")
    def test_fetch_unmapped_channel_type_warns(self, mock_auth):
        from backend.integrations.google_ads import GoogleAdsConnector

        mock_row = MagicMock()
        mock_row.segments.date = "2024-06-01"
        mock_row.campaign.name = "Display"
        mock_row.campaign.advertising_channel_type.name = "DISPLAY"
        mock_row.metrics.cost_micros = 10_000_000
        mock_row.metrics.clicks = 100
        mock_row.metrics.conversions = 5.0

        mock_batch = MagicMock()
        mock_batch.results = [mock_row]

        config = _make_config("google_ads", credentials={
            "developer_token": "tok", "client_id": "cid",
            "client_secret": "cs", "refresh_token": "rt",
        }, config={"customer_id": "123"})
        connector = GoogleAdsConnector(config)

        mock_service = MagicMock()
        mock_service.search_stream.return_value = [mock_batch]
        connector._client = MagicMock()
        connector._client.get_service.return_value = mock_service

        result = connector.fetch(datetime.date(2024, 6, 1), datetime.date(2024, 6, 30))

        assert any("Unmapped" in w for w in result.warnings)
        assert result.data["channel"].iloc[0] == "google_display"


# ---- Meta Ads connector tests ----

class TestMetaAdsConnector:
    def test_authenticate_missing_access_token(self):
        from backend.integrations.meta_ads import MetaAdsConnector
        config = _make_config("meta", credentials={}, config={"ad_account_id": "123"})
        connector = MetaAdsConnector(config)
        with pytest.raises(ConnectionError, match="Missing Meta credentials"):
            connector.authenticate()

    def test_authenticate_missing_ad_account_id(self):
        from backend.integrations.meta_ads import MetaAdsConnector
        config = _make_config("meta", credentials={"access_token": "tok"}, config={})
        connector = MetaAdsConnector(config)
        with pytest.raises(ConnectionError, match="Missing ad_account_id"):
            connector.authenticate()

    @patch("backend.integrations.meta_ads.MetaAdsConnector.authenticate")
    def test_fetch_transforms_to_schema(self, mock_auth):
        from backend.integrations.meta_ads import MetaAdsConnector

        mock_insight = {
            "date_start": "2024-06-01",
            "campaign_name": "Summer Sale",
            "publisher_platform": "facebook",
            "spend": "75.50",
            "clicks": "800",
            "actions": [
                {"action_type": "purchase", "value": "12"},
            ],
        }

        config = _make_config("meta", credentials={"access_token": "tok"},
                              config={"ad_account_id": "123"})
        connector = MetaAdsConnector(config)

        mock_account = MagicMock()
        mock_account.get_insights.return_value = [mock_insight]
        connector._account = mock_account

        result = connector.fetch(datetime.date(2024, 6, 1), datetime.date(2024, 6, 30))

        assert list(result.data.columns) == REQUIRED_COLUMNS
        assert len(result.data) == 1
        assert result.data["channel"].iloc[0] == "meta_feed"
        assert result.data["spend"].iloc[0] == 75.50
        assert result.data["in_platform_conversions"].iloc[0] == 12.0
        assert result.data["revenue"].iloc[0] == 0.0  # zero-filled

    @patch("backend.integrations.meta_ads.MetaAdsConnector.authenticate")
    def test_fetch_instagram_platform(self, mock_auth):
        from backend.integrations.meta_ads import MetaAdsConnector

        mock_insight = {
            "date_start": "2024-06-01",
            "campaign_name": "IG Promo",
            "publisher_platform": "instagram",
            "spend": "30.0",
            "clicks": "200",
            "actions": [],
        }

        config = _make_config("meta", credentials={"access_token": "tok"},
                              config={"ad_account_id": "123"})
        connector = MetaAdsConnector(config)
        mock_account = MagicMock()
        mock_account.get_insights.return_value = [mock_insight]
        connector._account = mock_account

        result = connector.fetch(datetime.date(2024, 6, 1), datetime.date(2024, 6, 30))
        assert result.data["channel"].iloc[0] == "meta_instagram"

    @patch("backend.integrations.meta_ads.MetaAdsConnector.authenticate")
    def test_fetch_empty_result_warns(self, mock_auth):
        from backend.integrations.meta_ads import MetaAdsConnector

        config = _make_config("meta", credentials={"access_token": "tok"},
                              config={"ad_account_id": "123"})
        connector = MetaAdsConnector(config)
        mock_account = MagicMock()
        mock_account.get_insights.return_value = []
        connector._account = mock_account

        result = connector.fetch(datetime.date(2024, 6, 1), datetime.date(2024, 6, 30))
        assert any("No data" in w for w in result.warnings)
        assert result.rows_fetched == 0


# ---- GA4 connector tests ----

class TestGA4Connector:
    def test_authenticate_missing_property_id(self):
        from backend.integrations.ga4 import GA4Connector
        config = _make_config("ga4", credentials={"refresh_token": "rt"}, config={})
        connector = GA4Connector(config)
        with pytest.raises(ConnectionError, match="Missing property_id"):
            connector.authenticate()

    def test_authenticate_missing_credentials(self):
        from backend.integrations.ga4 import GA4Connector
        config = _make_config("ga4", credentials={}, config={"property_id": "123"})
        connector = GA4Connector(config)
        with pytest.raises(ConnectionError, match="requires either"):
            connector.authenticate()

    @patch("backend.integrations.ga4.GA4Connector.authenticate")
    def test_fetch_transforms_to_schema(self, mock_auth):
        from backend.integrations.ga4 import GA4Connector

        # Mock GA4 report response
        mock_row1 = MagicMock()
        mock_row1.dimension_values = [
            MagicMock(value="20240601"),
            MagicMock(value="Organic Search"),
        ]
        mock_row1.metric_values = [
            MagicMock(value="500"),    # sessions
            MagicMock(value="1200.50"),  # revenue
            MagicMock(value="15"),     # orders
        ]

        mock_row2 = MagicMock()
        mock_row2.dimension_values = [
            MagicMock(value="20240601"),
            MagicMock(value="Direct"),
        ]
        mock_row2.metric_values = [
            MagicMock(value="300"),
            MagicMock(value="800.00"),
            MagicMock(value="10"),
        ]

        mock_response = MagicMock()
        mock_response.rows = [mock_row1, mock_row2]

        config = _make_config("ga4", credentials={"refresh_token": "rt"},
                              config={"property_id": "123456"})
        connector = GA4Connector(config)
        connector._client = MagicMock()
        connector._client.run_report.return_value = mock_response

        result = connector.fetch(datetime.date(2024, 6, 1), datetime.date(2024, 6, 30))

        assert list(result.data.columns) == REQUIRED_COLUMNS
        assert len(result.data) == 1  # aggregated by date
        assert result.data["sessions_organic"].iloc[0] == 500
        assert result.data["sessions_direct"].iloc[0] == 300
        assert result.data["spend"].iloc[0] == 0  # zero-filled
        assert result.data["channel"].iloc[0] == "organic_traffic"

    def test_channel_group_mapping(self):
        from backend.integrations.ga4 import CHANNEL_GROUP_MAP
        assert CHANNEL_GROUP_MAP["organic search"] == "sessions_organic"
        assert CHANNEL_GROUP_MAP["direct"] == "sessions_direct"
        assert CHANNEL_GROUP_MAP["email"] == "sessions_email"
        assert CHANNEL_GROUP_MAP["referral"] == "sessions_referral"


# ---- Shopify connector tests ----

class TestShopifyConnector:
    def test_authenticate_missing_access_token(self):
        from backend.integrations.shopify import ShopifyConnector
        config = _make_config("shopify", credentials={}, config={"shop_domain": "test.myshopify.com"})
        connector = ShopifyConnector(config)
        with pytest.raises(ConnectionError, match="Missing Shopify credentials"):
            connector.authenticate()

    def test_authenticate_missing_shop_domain(self):
        from backend.integrations.shopify import ShopifyConnector
        config = _make_config("shopify", credentials={"access_token": "tok"}, config={})
        connector = ShopifyConnector(config)
        with pytest.raises(ConnectionError, match="Missing shop_domain"):
            connector.authenticate()

    @patch("backend.integrations.shopify.ShopifyConnector.authenticate")
    def test_fetch_transforms_to_schema(self, mock_auth):
        from backend.integrations.shopify import ShopifyConnector

        mock_orders = [
            {"created_at": "2024-06-01T10:00:00Z", "total_price": "99.99"},
            {"created_at": "2024-06-01T14:30:00Z", "total_price": "149.99"},
            {"created_at": "2024-06-02T09:00:00Z", "total_price": "75.00"},
        ]

        config = _make_config("shopify", credentials={"access_token": "tok"},
                              config={"shop_domain": "test.myshopify.com"})
        connector = ShopifyConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"orders": mock_orders}
        mock_response.headers = {}

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = connector.fetch(datetime.date(2024, 6, 1), datetime.date(2024, 6, 30))

        assert list(result.data.columns) == REQUIRED_COLUMNS
        assert len(result.data) == 2  # 2 unique dates
        assert result.data["channel"].iloc[0] == "ecommerce"
        assert result.data["campaign"].iloc[0] == "shopify_orders"
        # June 1: 99.99 + 149.99 = 249.98
        day1 = result.data[result.data["date"] == "2024-06-01"]
        assert day1["revenue"].iloc[0] == pytest.approx(249.98)
        assert day1["orders"].iloc[0] == 2
        # Zero-filled ad columns
        assert result.data["spend"].iloc[0] == 0

    @patch("backend.integrations.shopify.ShopifyConnector.authenticate")
    def test_fetch_empty_warns(self, mock_auth):
        from backend.integrations.shopify import ShopifyConnector

        config = _make_config("shopify", credentials={"access_token": "tok"},
                              config={"shop_domain": "test.myshopify.com"})
        connector = ShopifyConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"orders": []}
        mock_response.headers = {}

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = connector.fetch(datetime.date(2024, 6, 1), datetime.date(2024, 6, 30))

        assert any("No orders" in w for w in result.warnings)


# ---- WooCommerce connector tests ----

class TestWooCommerceConnector:
    def test_authenticate_missing_consumer_key(self):
        from backend.integrations.woocommerce import WooCommerceConnector
        config = _make_config("woocommerce", credentials={}, config={"store_url": "https://store.com"})
        connector = WooCommerceConnector(config)
        with pytest.raises(ConnectionError, match="Missing WooCommerce credentials"):
            connector.authenticate()

    def test_authenticate_missing_store_url(self):
        from backend.integrations.woocommerce import WooCommerceConnector
        config = _make_config("woocommerce",
                              credentials={"consumer_key": "ck", "consumer_secret": "cs"},
                              config={})
        connector = WooCommerceConnector(config)
        with pytest.raises(ConnectionError, match="Missing store_url"):
            connector.authenticate()

    @patch("backend.integrations.woocommerce.WooCommerceConnector.authenticate")
    def test_fetch_transforms_to_schema(self, mock_auth):
        from backend.integrations.woocommerce import WooCommerceConnector

        mock_orders = [
            {"date_created": "2024-06-01T10:00:00", "total": "120.00"},
            {"date_created": "2024-06-01T15:00:00", "total": "80.00"},
        ]

        config = _make_config("woocommerce",
                              credentials={"consumer_key": "ck", "consumer_secret": "cs"},
                              config={"store_url": "https://store.com"})
        connector = WooCommerceConnector(config)

        # First call returns orders, second call returns empty (pagination stop)
        mock_resp_with_data = MagicMock()
        mock_resp_with_data.status_code = 200
        mock_resp_with_data.json.return_value = mock_orders

        mock_resp_empty = MagicMock()
        mock_resp_empty.status_code = 200
        mock_resp_empty.json.return_value = []

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = [mock_resp_with_data, mock_resp_empty]
            mock_client_cls.return_value = mock_client

            result = connector.fetch(datetime.date(2024, 6, 1), datetime.date(2024, 6, 30))

        assert list(result.data.columns) == REQUIRED_COLUMNS
        assert len(result.data) == 1
        assert result.data["channel"].iloc[0] == "ecommerce"
        assert result.data["revenue"].iloc[0] == 200.0
        assert result.data["orders"].iloc[0] == 2
        assert result.data["spend"].iloc[0] == 0  # zero-filled


# ---- Integration service tests ----

class TestSyncConnection:
    def _create_connection(self, db_session, platform="google_ads"):
        conn = ApiConnection(
            platform=platform,
            display_name="Test Connection",
            credentials_json=json.dumps({
                "developer_token": "tok", "client_id": "cid",
                "client_secret": "cs", "refresh_token": "rt",
            }),
            config_json=json.dumps({"customer_id": "123"}),
            is_active=True,
        )
        db_session.add(conn)
        db_session.commit()
        return conn

    def test_sync_inactive_connection_raises(self, db_session):
        conn = self._create_connection(db_session)
        conn.is_active = False
        db_session.commit()

        with pytest.raises(ValueError, match="not found or inactive"):
            sync_connection(db_session, conn.id,
                            datetime.date(2024, 6, 1), datetime.date(2024, 6, 30))

    def test_sync_nonexistent_connection_raises(self, db_session):
        with pytest.raises(ValueError, match="not found or inactive"):
            sync_connection(db_session, 999,
                            datetime.date(2024, 6, 1), datetime.date(2024, 6, 30))

    @patch("backend.integrations.service.get_connector")
    def test_sync_creates_upload_and_records(self, mock_get_connector, db_session):
        conn = self._create_connection(db_session)

        mock_df = pd.DataFrame({
            "date": ["2024-06-01"],
            "channel": ["google_search"],
            "campaign": ["Brand"],
            "spend": [100.0],
            "clicks": [200],
            "in_platform_conversions": [10.0],
            "revenue": [0.0],
            "orders": [0],
            "sessions_organic": [0],
            "sessions_direct": [0],
            "sessions_email": [0],
            "sessions_referral": [0],
        })

        mock_connector = MagicMock()
        mock_connector.fetch.return_value = FetchResult(data=mock_df)
        mock_get_connector.return_value = mock_connector

        sync = sync_connection(db_session, conn.id,
                                datetime.date(2024, 6, 1), datetime.date(2024, 6, 30))

        assert sync.status == "completed"
        assert sync.rows_fetched == 1

        # Verify Upload was created
        upload = db_session.query(Upload).filter_by(id=sync.upload_id).first()
        assert upload is not None
        assert upload.row_count == 1

        # Verify DailyRecord was stored
        records = db_session.query(DailyRecord).filter_by(upload_id=upload.id).all()
        assert len(records) == 1
        assert records[0].channel == "google_search"
        assert records[0].spend == 100.0

    @patch("backend.integrations.service.get_connector")
    def test_sync_failure_records_error(self, mock_get_connector, db_session):
        conn = self._create_connection(db_session)

        mock_connector = MagicMock()
        mock_connector.fetch.side_effect = ConnectionError("API timeout")
        mock_get_connector.return_value = mock_connector

        with pytest.raises(ConnectionError):
            sync_connection(db_session, conn.id,
                            datetime.date(2024, 6, 1), datetime.date(2024, 6, 30))

        # Verify sync was recorded as failed
        sync = db_session.query(ApiSync).first()
        assert sync.status == "failed"
        assert "API timeout" in sync.error_message


class TestMergeSources:
    def _create_upload_with_records(self, db_session, filename, rows):
        upload = Upload(filename=filename, row_count=len(rows), status="success")
        db_session.add(upload)
        db_session.flush()

        for row in rows:
            record = DailyRecord(upload_id=upload.id, **row)
            db_session.add(record)
        db_session.commit()
        return upload

    def test_merge_requires_at_least_two_uploads(self, db_session):
        with pytest.raises(ValueError, match="At least 2"):
            merge_sources(db_session, [1])

    def test_merge_ad_and_ecommerce_data(self, db_session):
        ad_upload = self._create_upload_with_records(db_session, "ads.csv", [
            {
                "date": datetime.date(2024, 6, 1),
                "channel": "google_search", "campaign": "Brand",
                "spend": 100.0, "clicks": 200,
                "in_platform_conversions": 10.0, "revenue": 0.0, "orders": 0,
                "sessions_organic": 0, "sessions_direct": 0,
                "sessions_email": 0, "sessions_referral": 0,
            },
        ])

        ecom_upload = self._create_upload_with_records(db_session, "shopify.csv", [
            {
                "date": datetime.date(2024, 6, 1),
                "channel": "ecommerce", "campaign": "shopify_orders",
                "spend": 0.0, "clicks": 0,
                "in_platform_conversions": 0.0, "revenue": 1500.0, "orders": 20,
                "sessions_organic": 0, "sessions_direct": 0,
                "sessions_email": 0, "sessions_referral": 0,
            },
        ])

        merged = merge_sources(db_session, [ad_upload.id, ecom_upload.id])

        records = db_session.query(DailyRecord).filter_by(upload_id=merged.id).all()
        assert len(records) == 1
        assert records[0].channel == "google_search"
        assert records[0].spend == 100.0
        assert records[0].revenue == 1500.0
        assert records[0].orders == 20

    def test_merge_ad_and_sessions_data(self, db_session):
        ad_upload = self._create_upload_with_records(db_session, "ads.csv", [
            {
                "date": datetime.date(2024, 6, 1),
                "channel": "google_search", "campaign": "Brand",
                "spend": 100.0, "clicks": 200,
                "in_platform_conversions": 10.0, "revenue": 0.0, "orders": 0,
                "sessions_organic": 0, "sessions_direct": 0,
                "sessions_email": 0, "sessions_referral": 0,
            },
        ])

        ga4_upload = self._create_upload_with_records(db_session, "ga4.csv", [
            {
                "date": datetime.date(2024, 6, 1),
                "channel": "organic_traffic", "campaign": "ga4_sessions",
                "spend": 0.0, "clicks": 0,
                "in_platform_conversions": 0.0, "revenue": 0.0, "orders": 0,
                "sessions_organic": 500, "sessions_direct": 300,
                "sessions_email": 50, "sessions_referral": 25,
            },
        ])

        merged = merge_sources(db_session, [ad_upload.id, ga4_upload.id])

        records = db_session.query(DailyRecord).filter_by(upload_id=merged.id).all()
        assert len(records) == 1
        assert records[0].sessions_organic == 500
        assert records[0].sessions_direct == 300
        assert records[0].spend == 100.0

    def test_merge_three_sources(self, db_session):
        ad_upload = self._create_upload_with_records(db_session, "ads.csv", [
            {
                "date": datetime.date(2024, 6, 1),
                "channel": "google_search", "campaign": "Brand",
                "spend": 100.0, "clicks": 200,
                "in_platform_conversions": 10.0, "revenue": 0.0, "orders": 0,
                "sessions_organic": 0, "sessions_direct": 0,
                "sessions_email": 0, "sessions_referral": 0,
            },
        ])

        ga4_upload = self._create_upload_with_records(db_session, "ga4.csv", [
            {
                "date": datetime.date(2024, 6, 1),
                "channel": "organic_traffic", "campaign": "ga4_sessions",
                "spend": 0.0, "clicks": 0,
                "in_platform_conversions": 0.0, "revenue": 0.0, "orders": 0,
                "sessions_organic": 500, "sessions_direct": 300,
                "sessions_email": 50, "sessions_referral": 25,
            },
        ])

        ecom_upload = self._create_upload_with_records(db_session, "shopify.csv", [
            {
                "date": datetime.date(2024, 6, 1),
                "channel": "ecommerce", "campaign": "shopify_orders",
                "spend": 0.0, "clicks": 0,
                "in_platform_conversions": 0.0, "revenue": 2000.0, "orders": 30,
                "sessions_organic": 0, "sessions_direct": 0,
                "sessions_email": 0, "sessions_referral": 0,
            },
        ])

        merged = merge_sources(db_session, [ad_upload.id, ga4_upload.id, ecom_upload.id])

        records = db_session.query(DailyRecord).filter_by(upload_id=merged.id).all()
        assert len(records) == 1
        assert records[0].spend == 100.0
        assert records[0].sessions_organic == 500
        assert records[0].revenue == 2000.0
        assert records[0].orders == 30

    def test_merge_creates_quality_report(self, db_session):
        u1 = self._create_upload_with_records(db_session, "a.csv", [
            {
                "date": datetime.date(2024, 6, 1),
                "channel": "google_search", "campaign": "Brand",
                "spend": 100.0, "clicks": 200,
                "in_platform_conversions": 10.0, "revenue": 500.0, "orders": 5,
                "sessions_organic": 100, "sessions_direct": 50,
                "sessions_email": 10, "sessions_referral": 5,
            },
        ])
        u2 = self._create_upload_with_records(db_session, "b.csv", [
            {
                "date": datetime.date(2024, 6, 2),
                "channel": "ecommerce", "campaign": "shopify",
                "spend": 0.0, "clicks": 0,
                "in_platform_conversions": 0.0, "revenue": 600.0, "orders": 8,
                "sessions_organic": 0, "sessions_direct": 0,
                "sessions_email": 0, "sessions_referral": 0,
            },
        ])

        merged = merge_sources(db_session, [u1.id, u2.id])

        qr = db_session.query(QualityReport).filter_by(upload_id=merged.id).first()
        assert qr is not None

    def test_merge_empty_upload_raises(self, db_session):
        u1 = Upload(filename="empty.csv", row_count=0, status="success")
        db_session.add(u1)
        u2 = Upload(filename="empty2.csv", row_count=0, status="success")
        db_session.add(u2)
        db_session.commit()

        with pytest.raises(ValueError, match="no records"):
            merge_sources(db_session, [u1.id, u2.id])


# ---- DB model tests ----

class TestDBModels:
    def test_api_connection_creation(self, db_session):
        conn = ApiConnection(
            platform="google_ads",
            display_name="My Google Ads",
            credentials_json=json.dumps({"key": "value"}),
            config_json=json.dumps({"customer_id": "123"}),
            is_active=True,
        )
        db_session.add(conn)
        db_session.commit()

        loaded = db_session.query(ApiConnection).first()
        assert loaded.platform == "google_ads"
        assert loaded.display_name == "My Google Ads"
        assert loaded.is_active is True

    def test_api_sync_creation(self, db_session):
        conn = ApiConnection(
            platform="meta",
            display_name="Meta Ads",
            credentials_json="{}",
            config_json="{}",
            is_active=True,
        )
        db_session.add(conn)
        db_session.flush()

        upload = Upload(filename="test.csv", row_count=10, status="success")
        db_session.add(upload)
        db_session.flush()

        sync = ApiSync(
            connection_id=conn.id,
            upload_id=upload.id,
            date_range_start=datetime.date(2024, 6, 1),
            date_range_end=datetime.date(2024, 6, 30),
            status="completed",
            rows_fetched=10,
        )
        db_session.add(sync)
        db_session.commit()

        loaded = db_session.query(ApiSync).first()
        assert loaded.status == "completed"
        assert loaded.rows_fetched == 10

    def test_daily_record_unique_constraint_allows_multiple_uploads(self, db_session):
        """The updated constraint allows same date/channel/campaign across different uploads."""
        u1 = Upload(filename="a.csv", row_count=1, status="success")
        u2 = Upload(filename="b.csv", row_count=1, status="success")
        db_session.add_all([u1, u2])
        db_session.flush()

        base = {
            "date": datetime.date(2024, 6, 1),
            "channel": "google_search",
            "campaign": "Brand",
            "spend": 100.0, "clicks": 200,
            "in_platform_conversions": 10.0, "revenue": 500.0, "orders": 5,
            "sessions_organic": 100, "sessions_direct": 50,
            "sessions_email": 10, "sessions_referral": 5,
        }

        r1 = DailyRecord(upload_id=u1.id, **base)
        r2 = DailyRecord(upload_id=u2.id, **base)
        db_session.add_all([r1, r2])
        db_session.commit()

        records = db_session.query(DailyRecord).all()
        assert len(records) == 2
