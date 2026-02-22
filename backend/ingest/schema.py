import pandera.pandas as pa
from pandera.pandas import Column, Check

VALID_CHANNELS = [
    "google_search",
    "google_shopping",
    "google_pmax",
    "google_youtube",
    "meta_feed",
    "meta_instagram",
    "meta_stories",
]

# Mapping of common variations to canonical channel names
CHANNEL_ALIASES = {
    "google search": "google_search",
    "googlesearch": "google_search",
    "google_ads_search": "google_search",
    "google shopping": "google_shopping",
    "googleshopping": "google_shopping",
    "google_ads_shopping": "google_shopping",
    "google pmax": "google_pmax",
    "googlepmax": "google_pmax",
    "google_performance_max": "google_pmax",
    "performance_max": "google_pmax",
    "google youtube": "google_youtube",
    "googleyoutube": "google_youtube",
    "youtube": "google_youtube",
    "meta feed": "meta_feed",
    "metafeed": "meta_feed",
    "facebook_feed": "meta_feed",
    "facebook": "meta_feed",
    "meta instagram": "meta_instagram",
    "metainstagram": "meta_instagram",
    "instagram": "meta_instagram",
    "meta stories": "meta_stories",
    "metastories": "meta_stories",
    "facebook_stories": "meta_stories",
    "instagram_stories": "meta_stories",
}

REQUIRED_COLUMNS = [
    "date",
    "channel",
    "campaign",
    "spend",
    "impressions",
    "clicks",
    "in_platform_conversions",
    "revenue",
    "orders",
    "sessions_organic",
    "sessions_direct",
    "sessions_email",
    "sessions_referral",
]

ingestion_schema = pa.DataFrameSchema(
    columns={
        "date": Column(pa.DateTime, coerce=True, nullable=False),
        "channel": Column(str, nullable=False),
        "campaign": Column(str, nullable=False),
        "spend": Column(float, Check.ge(0), coerce=True, nullable=False),
        "impressions": Column(int, Check.ge(0), coerce=True, nullable=False),
        "clicks": Column(int, Check.ge(0), coerce=True, nullable=False),
        "in_platform_conversions": Column(float, Check.ge(0), coerce=True, nullable=False),
        "revenue": Column(float, coerce=True, nullable=False),
        "orders": Column(int, Check.ge(0), coerce=True, nullable=False),
        "sessions_organic": Column(int, Check.ge(0), coerce=True, nullable=False),
        "sessions_direct": Column(int, Check.ge(0), coerce=True, nullable=False),
        "sessions_email": Column(int, Check.ge(0), coerce=True, nullable=False),
        "sessions_referral": Column(int, Check.ge(0), coerce=True, nullable=False),
    },
    strict=False,  # allow extra columns (they'll be ignored)
    coerce=True,
)
