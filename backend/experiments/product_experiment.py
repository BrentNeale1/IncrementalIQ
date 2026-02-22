"""Product-level experiment: Google Ads only.

CLAUDE.md spec:
- User uploads advertised product list from Google Ads product reports
- Tool compares in-campaign vs. out-of-campaign product revenue/orders
- HARD CONSTRAINT: Only valid for manually-controlled product selection.
  If campaign type is PMax or uses auto-bidding, the tool MUST refuse to
  run this test and explain why (selection bias — products are chosen
  based on conversion probability, not randomly).
- This check is NOT bypassable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd


# Campaign types that invalidate the experiment due to selection bias
INVALID_CAMPAIGN_TYPES = {"pmax", "performance_max", "smart_shopping"}

# Bidding strategies that invalidate the experiment
INVALID_BIDDING_STRATEGIES = {
    "target_roas",
    "target_cpa",
    "maximize_conversions",
    "maximize_conversion_value",
}

SELECTION_BIAS_EXPLANATION = (
    "This product-level experiment cannot be run because the campaign uses "
    "automated product selection (Performance Max or auto-bidding). "
    "In these campaign types, Google's algorithm selects products based on "
    "their predicted conversion probability, not randomly. This creates "
    "selection bias: advertised products would have performed better "
    "regardless of the advertising. Any comparison between advertised and "
    "non-advertised products would overstate the true advertising effect. "
    "Consider running a spend-scaling experiment instead."
)


class ProductExperimentError(Exception):
    """Raised when the product experiment cannot be run."""
    pass


@dataclass
class ProductExperimentResult:
    """Output from a product-level experiment."""
    advertised_product_count: int
    non_advertised_product_count: int
    # Revenue comparison
    advertised_revenue_total: float
    non_advertised_revenue_total: float
    advertised_revenue_per_product: float
    non_advertised_revenue_per_product: float
    revenue_lift_pct: float
    # Orders comparison
    advertised_orders_total: int
    non_advertised_orders_total: int
    advertised_orders_per_product: float
    non_advertised_orders_per_product: float
    orders_lift_pct: float
    # Statistical test
    revenue_significant: bool
    orders_significant: bool
    p_value_revenue: float
    p_value_orders: float
    # Metadata
    campaign_name: str
    campaign_type: str
    bidding_strategy: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def validate_campaign_eligibility(
    campaign_type: str,
    bidding_strategy: str,
) -> None:
    """Check that the campaign is eligible for a product experiment.

    HARD CONSTRAINT — raises ProductExperimentError if PMax or auto-bidding.
    This check is not bypassable.
    """
    ct_lower = campaign_type.strip().lower().replace(" ", "_")
    bs_lower = bidding_strategy.strip().lower().replace(" ", "_")

    if ct_lower in INVALID_CAMPAIGN_TYPES:
        raise ProductExperimentError(SELECTION_BIAS_EXPLANATION)

    if bs_lower in INVALID_BIDDING_STRATEGIES:
        raise ProductExperimentError(SELECTION_BIAS_EXPLANATION)


def run_product_experiment(
    product_data: pd.DataFrame,
    advertised_products: list[str],
    campaign_name: str,
    campaign_type: str,
    bidding_strategy: str,
) -> ProductExperimentResult:
    """Compare in-campaign vs. out-of-campaign product performance.

    Parameters
    ----------
    product_data : DataFrame with columns: product_id, revenue, orders
                  Represents all products in the catalogue for the analysis period.
    advertised_products : list of product_id values that were in the campaign
    campaign_name : name of the campaign being tested
    campaign_type : type (e.g. "search", "shopping", "pmax")
    bidding_strategy : bidding strategy (e.g. "manual_cpc", "target_roas")

    Returns
    -------
    ProductExperimentResult with lift estimates

    Raises
    ------
    ProductExperimentError if campaign uses PMax or auto-bidding
    """
    # HARD CONSTRAINT: refuse PMax and auto-bidding
    validate_campaign_eligibility(campaign_type, bidding_strategy)

    required_cols = {"product_id", "revenue", "orders"}
    missing = required_cols - set(product_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    advertised_set = set(advertised_products)
    is_advertised = product_data["product_id"].isin(advertised_set)

    adv = product_data[is_advertised]
    non_adv = product_data[~is_advertised]

    if len(adv) == 0:
        raise ValueError("No advertised products found in the data.")
    if len(non_adv) == 0:
        raise ValueError("No non-advertised products found for comparison.")

    # Revenue metrics
    adv_rev_total = float(adv["revenue"].sum())
    non_adv_rev_total = float(non_adv["revenue"].sum())
    adv_rev_per = adv_rev_total / len(adv)
    non_adv_rev_per = non_adv_rev_total / len(non_adv)
    rev_lift = ((adv_rev_per - non_adv_rev_per) / non_adv_rev_per * 100) if non_adv_rev_per > 0 else 0

    # Orders metrics
    adv_ord_total = int(adv["orders"].sum())
    non_adv_ord_total = int(non_adv["orders"].sum())
    adv_ord_per = adv_ord_total / len(adv)
    non_adv_ord_per = non_adv_ord_total / len(non_adv)
    ord_lift = ((adv_ord_per - non_adv_ord_per) / non_adv_ord_per * 100) if non_adv_ord_per > 0 else 0

    # Welch's t-test for significance
    from scipy import stats
    if adv["revenue"].std() > 0 and non_adv["revenue"].std() > 0:
        t_rev, p_rev = stats.ttest_ind(adv["revenue"], non_adv["revenue"], equal_var=False)
    else:
        p_rev = 1.0
    if adv["orders"].std() > 0 and non_adv["orders"].std() > 0:
        t_ord, p_ord = stats.ttest_ind(adv["orders"], non_adv["orders"], equal_var=False)
    else:
        p_ord = 1.0

    warnings = []
    if len(adv) < 30:
        warnings.append(
            f"Only {len(adv)} advertised products — small sample may yield unreliable results."
        )
    if len(non_adv) < 30:
        warnings.append(
            f"Only {len(non_adv)} non-advertised products for comparison."
        )

    return ProductExperimentResult(
        advertised_product_count=len(adv),
        non_advertised_product_count=len(non_adv),
        advertised_revenue_total=round(adv_rev_total, 2),
        non_advertised_revenue_total=round(non_adv_rev_total, 2),
        advertised_revenue_per_product=round(adv_rev_per, 2),
        non_advertised_revenue_per_product=round(non_adv_rev_per, 2),
        revenue_lift_pct=round(rev_lift, 2),
        advertised_orders_total=adv_ord_total,
        non_advertised_orders_total=non_adv_ord_total,
        advertised_orders_per_product=round(adv_ord_per, 2),
        non_advertised_orders_per_product=round(non_adv_ord_per, 2),
        orders_lift_pct=round(ord_lift, 2),
        revenue_significant=bool(p_rev < 0.05),
        orders_significant=bool(p_ord < 0.05),
        p_value_revenue=round(float(p_rev), 4),
        p_value_orders=round(float(p_ord), 4),
        campaign_name=campaign_name,
        campaign_type=campaign_type,
        bidding_strategy=bidding_strategy,
        warnings=warnings,
    )
