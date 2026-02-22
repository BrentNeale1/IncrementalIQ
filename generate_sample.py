"""Generate a realistic sample CSV for IncrementIQ demo."""
import csv
import math
import random
from datetime import date, timedelta

random.seed(42)

START = date(2024, 1, 1)
DAYS = 730  # ~24 months → "sufficient" history status
END = START + timedelta(days=DAYS - 1)

CHANNELS = {
    # channel: (base_daily_spend, spend_noise_pct, ctr, conv_rate, cpc_range)
    "google_search":   (180, 0.30, 0.06, 0.035, (0.80, 2.50)),
    "google_shopping": (120, 0.35, 0.04, 0.028, (0.40, 1.20)),
    "google_pmax":     (100, 0.40, 0.03, 0.025, (0.50, 1.80)),
    "meta_feed":       (90,  0.35, 0.015, 0.018, (0.30, 0.90)),
    "google_youtube":  (60,  0.45, 0.008, 0.008, (0.10, 0.40)),
}

# Avg order value and revenue-per-conversion
AVG_ORDER_VALUE = 85.0

rows = []

for day_offset in range(DAYS):
    d = START + timedelta(days=day_offset)
    day_of_week = d.weekday()  # 0=Mon
    day_of_year = d.timetuple().tm_yday

    # Seasonality multipliers
    weekly_mult = 1.0 + 0.15 * math.sin(2 * math.pi * (day_of_week - 2) / 7)  # peak Thu-Fri
    yearly_mult = 1.0 + 0.25 * math.sin(2 * math.pi * (day_of_year - 80) / 365)  # peak ~spring
    # Black Friday / holiday bump (Nov-Dec)
    if d.month == 11 and d.day >= 20:
        yearly_mult *= 1.6
    elif d.month == 12 and d.day <= 24:
        yearly_mult *= 1.45
    elif d.month == 12 and d.day >= 25:
        yearly_mult *= 0.7  # post-Christmas dip

    # Slow growth trend over 2 years
    trend = 1.0 + 0.15 * (day_offset / DAYS)

    season = weekly_mult * yearly_mult * trend

    # Baseline organic metrics (not driven by ads)
    sessions_organic = max(0, int(320 * season * random.gauss(1.0, 0.12)))
    sessions_direct = max(0, int(180 * season * random.gauss(1.0, 0.15)))
    sessions_email = max(0, int(45 * random.gauss(1.0, 0.25)))
    sessions_referral = max(0, int(30 * random.gauss(1.0, 0.20)))

    # Baseline revenue from organic (not attributed to ads)
    baseline_orders = max(0, int((sessions_organic + sessions_direct) * 0.022 * random.gauss(1.0, 0.10)))
    baseline_revenue = baseline_orders * AVG_ORDER_VALUE * random.gauss(1.0, 0.08)

    total_ad_revenue = 0.0
    total_ad_orders = 0

    for channel, (base_spend, noise_pct, ctr, conv_rate, cpc_range) in CHANNELS.items():
        # Spend varies with season + channel-specific noise
        spend = base_spend * season * random.gauss(1.0, noise_pct)

        # Occasional spend scaling events (for interesting patterns)
        if channel == "google_search" and date(2025, 3, 1) <= d <= date(2025, 3, 31):
            spend *= 1.8  # deliberate spend increase
        if channel == "meta_feed" and date(2025, 6, 1) <= d <= date(2025, 6, 14):
            spend *= 0.3  # deliberate spend decrease

        spend = max(5.0, spend)
        cpc = random.uniform(*cpc_range)
        clicks = max(1, int(spend / cpc))
        impressions = max(clicks, int(clicks / max(0.005, ctr * random.gauss(1.0, 0.15))))

        # In-platform conversions (noisy, platform-inflated)
        true_convs = clicks * conv_rate * season * 0.3 * random.gauss(1.0, 0.20)
        in_platform_conversions = max(0, true_convs * random.uniform(1.1, 1.5))  # platform inflation

        # Actual incremental orders/revenue (what the model should find)
        incremental_orders = max(0, int(true_convs * random.gauss(1.0, 0.15)))
        incremental_revenue = incremental_orders * AVG_ORDER_VALUE * random.gauss(1.0, 0.10)

        total_ad_orders += incremental_orders
        total_ad_revenue += incremental_revenue

        # Revenue and orders are total (baseline + incremental), split evenly across channels
        # We'll add baseline portion to each channel's revenue proportionally
        channel_share = base_spend / sum(bs for bs, *_ in CHANNELS.values())
        revenue = incremental_revenue + baseline_revenue * channel_share
        orders = incremental_orders + int(baseline_orders * channel_share)

        rows.append({
            "date": d.isoformat(),
            "channel": channel,
            "campaign": f"{channel}_always_on",
            "spend": round(spend, 2),
            "impressions": impressions,
            "clicks": clicks,
            "in_platform_conversions": round(in_platform_conversions, 2),
            "revenue": round(revenue, 2),
            "orders": orders,
            "sessions_organic": sessions_organic,
            "sessions_direct": sessions_direct,
            "sessions_email": sessions_email,
            "sessions_referral": sessions_referral,
        })

out_path = "sample_data.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "date", "channel", "campaign", "spend", "impressions", "clicks",
        "in_platform_conversions", "revenue", "orders",
        "sessions_organic", "sessions_direct", "sessions_email", "sessions_referral",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} rows across {DAYS} days, {len(CHANNELS)} channels -> {out_path}")
print(f"Date range: {START} to {END}")
print(f"Channels: {', '.join(CHANNELS.keys())}")
