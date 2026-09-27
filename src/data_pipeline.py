"""
Turns raw SEC EDGAR filings + daily prices into a flat, model-ready table.

One row per (symbol, quarterly report).

Earnings events:
    Every 10-Q / 10-K is one quarterly report. Its announcement time is the
    8-K "Item 2.02 Results of Operations" press release filed between the
    period end and the 10-Q/10-K (the real earnings release), falling back to
    the 10-Q/10-K acceptance time when the company didn't file one.

Features:
    Quarterly fundamentals from XBRL company facts -- as ratios, margins and
    growth rates so companies of different sizes are comparable -- plus
    valuation (price x reported shares), sector, how the quarter compares with
    the company's own recent trend, multi-year trend quality, pre-announcement
    price momentum/volatility, and the first-day reaction to the release.
    Each value is the one from the *original* filing for that period (later
    restatements are ignored).

Targets:
    Forward % return *in excess of the benchmark* (SPY) over the same days,
    so the model learns what the report says about the stock rather than
    which way the whole market went. The 1d target runs from the last close
    before the announcement to the next close (the market's reaction). The
    1w/1m/1y targets start at that first post-release close instead: those
    models get the first-day reaction as an input, so it can't also be part
    of what they predict.

    Each horizon's label only needs its own window of prices, so recent
    reports still train the short horizons before a year has passed.

Lookahead note:
    XBRL numbers come from the 10-Q/10-K, which is often filed a few days
    after the press release. The headline numbers are the same ones published
    in the release, so this is treated as available at announcement time.
"""

from __future__ import annotations

import logging
import math
import traceback
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from .config import config
from .price_client import PriceClient, PriceAPIError
from .sec_client import SECClient, SECAPIError

logger = logging.getLogger(__name__)

ET = "America/New_York"
MARKET_CLOSE_HOUR = 16
PERIODIC_FORMS = {"10-Q", "10-K"}

# Metric -> XBRL us-gaap tags, in preference order. Companies switch tags over
# time (e.g. SalesRevenueNet -> RevenueFromContractWithCustomer... in 2018), so
# each period uses the first tag that has a value for it.
DURATION_METRICS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",  # banks
    ],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
}
INSTANT_METRICS = {
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_liabilities": ["Liabilities"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "inventory": ["InventoryNet"],
    "receivables": ["AccountsReceivableNetCurrent"],
}
# Share counts are averages, so a missing quarter can't be derived as
# YTD minus the prior YTD the way flows are.
MIN_PLAUSIBLE_SHARES = 1_000_000
SHARE_TAGS = ["WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"]

# Broad industry groups from the SIC code in the EDGAR submissions header.
# First matching range wins, so narrow groups come before broad ones.
SECTOR_RANGES = [
    ("biopharma", [(2833, 2836), (8731, 8731)]),
    ("medtech_health", [(3840, 3851), (8000, 8099), (5047, 5047), (5122, 5122)]),
    ("software_internet", [(7370, 7379)]),
    ("semis_hardware", [(3570, 3579), (3600, 3699)]),
    ("energy", [(1300, 1389), (2900, 2999)]),
    ("materials", [(1000, 1299), (1400, 1499), (2600, 2699), (2800, 2832), (2837, 2899), (3300, 3399)]),
    ("banks", [(6000, 6199)]),
    ("insurance", [(6300, 6411)]),
    ("real_estate", [(6500, 6553), (6798, 6798)]),
    ("financial_other", [(6200, 6299), (6412, 6499), (6700, 6797), (6799, 6799)]),
    ("utilities", [(4900, 4999)]),
    ("telecom_media", [(4800, 4899), (2700, 2799), (7800, 7899)]),
    ("consumer_retail", [(5200, 5999), (2000, 2399), (3140, 3199), (3940, 3949), (7000, 7099)]),
    ("transport", [(4000, 4799)]),
    ("industrials", [(1500, 1799), (2400, 2599), (3400, 3569), (3580, 3599), (3700, 3839), (3852, 3939), (3950, 3999), (5000, 5199)]),
    ("services_other", [(7100, 7369), (7380, 7799), (7900, 8999)]),
]
SECTORS = [name for name, _ in SECTOR_RANGES] + ["other"]
SECTOR_FEATURES = [f"sector_{s}" for s in SECTORS]

FUNDAMENTAL_FEATURES = [
    "log_revenue", "log_assets",
    "gross_margin", "operating_margin", "net_margin",
    "rnd_to_revenue", "sga_to_revenue", "sbc_to_revenue",
    "ocf_margin", "fcf_margin", "capital_return_to_revenue",
    "revenue_growth_qoq", "revenue_growth_yoy",
    "operating_income_growth_yoy", "net_income_growth_yoy", "ocf_growth_yoy",
    "gross_margin_change_yoy", "operating_margin_change_yoy",
    "current_ratio", "liabilities_to_assets", "cash_to_assets",
    "long_term_debt_to_assets", "inventory_to_assets", "receivables_to_revenue",
    "return_on_assets",
]
VALUATION_FEATURES = ["log_market_cap", "sales_yield", "earnings_yield", "fcf_yield", "book_to_market"]
# This quarter vs. the company's own last four -- a stand-in for "surprise",
# since EDGAR has no analyst estimates. Used by every horizon.
VS_TREND_FEATURES = [
    "revenue_growth_accel", "revenue_growth_vs_trend",
    "gross_margin_vs_trend", "operating_margin_vs_trend",
]
# Multi-year trajectory; only the 1m/1y models get these.
TREND_QUALITY_FEATURES = [
    "revenue_growth_volatility_8q", "positive_growth_share_8q",
    "operating_margin_slope_8q", "revenue_cagr_2y", "accruals_ttm",
]
PRICE_FEATURES = [
    "report_lag_days",
    "pre_excess_return_21d", "pre_excess_return_63d", "pre_excess_return_252d",
    "pre_volatility_21d",
]
# The market's first-day reaction; only for horizons measured after it.
DAY1_FEATURES = ["day1_excess_return"]

BASE_FEATURES = FUNDAMENTAL_FEATURES + VALUATION_FEATURES + VS_TREND_FEATURES + PRICE_FEATURES + SECTOR_FEATURES
FEATURE_COLUMNS = BASE_FEATURES + DAY1_FEATURES + TREND_QUALITY_FEATURES

# Features also given as a percentile against every company that reported
# the same calendar quarter, which strips out economy-wide swings.
RANK_FEATURES = [
    "gross_margin", "operating_margin", "fcf_margin", "return_on_assets",
    "revenue_growth_yoy", "revenue_growth_accel",
    "sales_yield", "earnings_yield", "fcf_yield", "book_to_market",
    "pre_excess_return_63d",
]


# Inputs grouped by what they describe, for explaining models: shuffling a
# whole group in held-out data and measuring the lost accuracy shows how much
# a model leans on that kind of information. Together they cover every input.
FEATURE_GROUPS = {
    "profitability": ["gross_margin", "operating_margin", "net_margin", "return_on_assets",
                      "gross_margin_change_yoy", "operating_margin_change_yoy"],
    "growth": ["revenue_growth_qoq", "revenue_growth_yoy", "operating_income_growth_yoy",
               "net_income_growth_yoy", "ocf_growth_yoy"],
    "cost_structure": ["rnd_to_revenue", "sga_to_revenue", "sbc_to_revenue"],
    "cash_flow": ["ocf_margin", "fcf_margin", "capital_return_to_revenue"],
    "balance_sheet": ["current_ratio", "liabilities_to_assets", "cash_to_assets",
                      "long_term_debt_to_assets", "inventory_to_assets", "receivables_to_revenue"],
    "size": ["log_revenue", "log_assets"],
    "valuation": VALUATION_FEATURES,
    "vs_own_trend": VS_TREND_FEATURES,
    "trend_quality": TREND_QUALITY_FEATURES,
    "price_momentum": ["pre_excess_return_21d", "pre_excess_return_63d", "pre_excess_return_252d", "pre_volatility_21d"],
    "release_timing": ["report_lag_days"],
    "first_day_reaction": DAY1_FEATURES,
    "sector": SECTOR_FEATURES,
}
assert sorted(c for cols in FEATURE_GROUPS.values() for c in cols) == sorted(FEATURE_COLUMNS), \
    "FEATURE_GROUPS must cover every feature exactly once"


def horizon_features(horizon: str) -> list[str]:
    cols = list(BASE_FEATURES)
    if horizon != "1d":
        cols += DAY1_FEATURES
    if horizon in ("1m", "1y"):
        cols += TREND_QUALITY_FEATURES
    return cols


@dataclass
class PipelineEvent:
    level: str
    symbol: str
    stage: str
    message: str
    traceback: str | None = None


@dataclass
class PipelineResult:
    dataframe: pd.DataFrame
    events: list[PipelineEvent] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)


@dataclass
class EarningsEvent:
    period_end: pd.Timestamp
    announced_at: pd.Timestamp  # tz-aware, US/Eastern
    source_form: str            # "8-K" when the press release was found
    periodic_filing: dict = field(default_factory=dict)  # the 10-Q/10-K the numbers come from


@dataclass
class SymbolData:
    cik: int
    profile: dict
    earnings: list[EarningsEvent]
    table: pd.DataFrame
    releases: list[pd.Timestamp]


# ------------------------------------------------------------------ filings
def _acceptance_time(filing: dict) -> pd.Timestamp:
    # acceptanceDateTime is UTC ("...Z"). Without it, assume the filing date
    # before the open, which errs toward an earlier (safer) baseline price.
    raw = filing.get("acceptanceDateTime")
    if raw:
        try:
            return pd.Timestamp(raw).tz_convert(ET)
        except (ValueError, TypeError):
            pass
    return pd.Timestamp(filing["filingDate"]).tz_localize(ET) + pd.Timedelta(hours=9)


def _earnings_releases(filings: list[dict]) -> list[pd.Timestamp]:
    return sorted(
        _acceptance_time(f)
        for f in filings
        if f.get("form") == "8-K" and "2.02" in (f.get("items") or "").split(",")
    )


def _earnings_events(filings: list[dict]) -> list[EarningsEvent]:
    periodic: dict[pd.Timestamp, dict] = {}
    for f in filings:
        if f.get("form") not in PERIODIC_FORMS or not f.get("reportDate"):
            continue
        end = pd.Timestamp(f["reportDate"])
        # Keep the original filing for each period, not later duplicates.
        if end not in periodic or f["filingDate"] < periodic[end]["filingDate"]:
            periodic[end] = f

    releases = _earnings_releases(filings)
    events = []
    for end, f in sorted(periodic.items()):
        filed_at = _acceptance_time(f)
        # Latest press release between period end and the 10-Q/10-K. "Latest"
        # skips mid-quarter 2.02 filings such as preliminary-results warnings.
        matches = [r for r in releases if end.tz_localize(ET) < r <= filed_at]
        if matches:
            events.append(EarningsEvent(end, matches[-1], "8-K", f))
        else:
            events.append(EarningsEvent(end, filed_at, f["form"], f))
    return events


def sector_group(sic: int | None) -> str:
    if sic is None:
        return "other"
    for name, ranges in SECTOR_RANGES:
        if any(lo <= sic <= hi for lo, hi in ranges):
            return name
    return "other"


# -------------------------------------------------------------------- XBRL
def _original_values(facts: dict, tag: str, unit: str = "USD", namespace: str = "us-gaap") -> list[dict]:
    """Facts for a tag, keeping only the earliest-filed value per period so
    later restatements can't leak into historical features."""
    entries = facts.get("facts", {}).get(namespace, {}).get(tag, {}).get("units", {}).get(unit, [])
    best: dict[tuple, dict] = {}
    for e in entries:
        if e.get("form") not in PERIODIC_FORMS:
            continue
        key = (e.get("start"), e["end"])
        if key not in best or e["filed"] < best[key]["filed"]:
            best[key] = e
    return list(best.values())


def _days(start: str, end: str) -> int:
    return (pd.Timestamp(end) - pd.Timestamp(start)).days


def _quarterly_durations(entries: list[dict]) -> dict[str, float]:
    """period_end -> 3-month value. 10-Qs often only report year-to-date
    figures (always for cash flow) and 10-Ks only the full year, so a missing
    quarter is derived as YTD(end) - YTD(previous quarter end)."""
    by_end: dict[str, list[tuple[str, float]]] = {}
    by_start: dict[str, dict[str, float]] = {}
    for e in entries:
        if not e.get("start"):
            continue
        by_end.setdefault(e["end"], []).append((e["start"], e["val"]))
        by_start.setdefault(e["start"], {})[e["end"]] = e["val"]

    out: dict[str, float] = {}
    for end, spans in by_end.items():
        direct = [v for s, v in spans if 75 <= _days(s, end) <= 105]
        if direct:
            out[end] = direct[0]
            continue
        for start, ytd in sorted(spans, key=lambda sv: _days(sv[0], end)):
            if _days(start, end) <= 105:
                continue
            prior = [
                v for prev_end, v in by_start[start].items()
                if 75 <= _days(prev_end, end) <= 105
            ]
            if prior:
                out[end] = ytd - prior[0]
                break
    return out


def _share_counts(facts: dict) -> dict[str, float]:
    """period_end -> weighted-average share count: the 3-month figure, or the
    full-year one for a 10-K quarter that only reports the annual average."""
    out: dict[str, float] = {}
    annual: dict[str, float] = {}
    for tag in SHARE_TAGS:
        for e in _original_values(facts, tag, unit="shares"):
            if not e.get("start") or not e.get("val"):
                continue
            span = _days(e["start"], e["end"])
            if 75 <= span <= 105:
                out.setdefault(e["end"], e["val"])
            elif 350 <= span <= 380:
                annual.setdefault(e["end"], e["val"])
    for end, v in annual.items():
        out.setdefault(end, v)
    return out


def _fundamentals_table(facts: dict) -> pd.DataFrame:
    """One row per fiscal period end, one column per metric."""
    columns: dict[str, dict[str, float]] = {}
    for metric, tags in DURATION_METRICS.items():
        merged: dict[str, float] = {}
        for tag in tags:
            for end, v in _quarterly_durations(_original_values(facts, tag)).items():
                merged.setdefault(end, v)
        columns[metric] = merged
    for metric, tags in INSTANT_METRICS.items():
        merged = {}
        for tag in tags:
            for e in _original_values(facts, tag):
                if not e.get("start"):
                    merged.setdefault(e["end"], e["val"])
        columns[metric] = merged
    columns["shares"] = _share_counts(facts)

    df = pd.DataFrame(columns, dtype="float64")
    if df.empty:
        return df
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()

    # Filers occasionally tag share counts in thousands. Under a million
    # shares is essentially never real for a listed company.
    df.loc[df["shares"] < MIN_PLAUSIBLE_SHARES, "shares"] = np.nan

    # Fall back to the cover-page share count (dated a few weeks after the
    # period end) for quarters without a weighted average.
    cover = sorted(
        (pd.Timestamp(e["end"]), float(e["val"]))
        for e in _original_values(facts, "EntityCommonStockSharesOutstanding", unit="shares", namespace="dei")
        if (e.get("val") or 0) >= MIN_PLAUSIBLE_SHARES
    )
    if cover:
        for end in df.index[df["shares"].isna()]:
            match = [v for d, v in cover if end <= d <= end + pd.Timedelta(days=120)]
            if match:
                df.at[end, "shares"] = match[0]

    # Same kind of error in the other direction, or a short run the floor
    # above misses: drop counts wildly off their neighbours. Real splits (at
    # most ~20-for-1) stay well inside this band.
    neighbours = df["shares"].rolling(9, center=True, min_periods=3).median()
    ratio = df["shares"] / neighbours
    df.loc[(ratio < 1 / 30) | (ratio > 30), "shares"] = np.nan
    return df


def _row_near(table: pd.DataFrame, target: pd.Timestamp, tolerance_days: int) -> pd.Series | None:
    if table.empty:
        return None
    diffs = np.abs((table.index - target).days)
    i = int(np.argmin(diffs))
    return table.iloc[i] if diffs[i] <= tolerance_days else None


def _quarter_history(table: pd.DataFrame, period_end: pd.Timestamp, n: int) -> list[pd.Series]:
    """Rows for this quarter and the n-1 before it, newest first. Steps back
    from each quarter actually found, so 52/53-week fiscal years don't drift.
    A missing quarter is an empty Series."""
    empty = pd.Series(dtype="float64")
    cur = _row_near(table, period_end, 0)
    rows = [empty if cur is None else cur]
    anchor = period_end
    for _ in range(n - 1):
        target = anchor - pd.Timedelta(days=91)
        row = _row_near(table, target, 20)
        if row is None:
            rows.append(empty)
            anchor = target
        else:
            rows.append(row)
            anchor = row.name
    return rows


def _num(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def _ratio(a, b, limit: float = 50.0) -> float:
    a, b = _num(a), _num(b)
    if math.isnan(a) or math.isnan(b) or b == 0:
        return math.nan
    return float(np.clip(a / b, -limit, limit))


def _growth(cur, prev) -> float:
    cur, prev = _num(cur), _num(prev)
    if math.isnan(cur) or math.isnan(prev) or prev == 0:
        return math.nan
    return float(np.clip((cur - prev) / abs(prev), -10.0, 10.0))


def _log10(x) -> float:
    x = _num(x)
    return math.log10(x) if x > 0 else math.nan


def _mean(values, min_count: int = 1) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(np.mean(vals)) if len(vals) >= min_count else math.nan


def _gross(row: pd.Series) -> float:
    gross = _num(row.get("gross_profit"))
    if math.isnan(gross):
        gross = _num(row.get("revenue")) - _num(row.get("cost_of_revenue"))
    return gross


def _ttm(rows: list[pd.Series], metric: str, offset: int = 0) -> float:
    vals = [_num(r.get(metric)) for r in rows[offset: offset + 4]]
    if len(vals) < 4 or any(math.isnan(v) for v in vals):
        return math.nan
    return float(sum(vals))


def _slope(values: list[float], min_count: int = 4) -> float:
    """Least-squares slope per quarter; values are newest first."""
    pts = [(-i, v) for i, v in enumerate(values) if not math.isnan(v)]
    if len(pts) < min_count:
        return math.nan
    x, y = np.array(pts, dtype="float64").T
    return float(np.polyfit(x, y, 1)[0])


def _fundamental_features(table: pd.DataFrame, period_end: pd.Timestamp) -> tuple[dict, dict] | None:
    """(features, extras). `extras` holds the absolute TTM figures and share
    count that valuation needs once the price is known."""
    rows = _quarter_history(table, period_end, 12)
    cur, prev_q, prev_y = rows[0], rows[1], rows[4]
    if cur.empty or (math.isnan(_num(cur.get("revenue"))) and math.isnan(_num(cur.get("total_assets")))):
        return None

    rev = cur.get("revenue")
    gross = _gross(cur)
    ocf, capex = _num(cur.get("operating_cash_flow")), _num(cur.get("capex"))
    capital_return = np.nansum([_num(cur.get("buybacks")), _num(cur.get("dividends_paid"))])

    revs = [_num(r.get("revenue")) for r in rows]
    growth = [_growth(revs[k], revs[k + 4]) for k in range(8)]
    gross_m = [_ratio(_gross(r), r.get("revenue")) for r in rows[:8]]
    op_m = [_ratio(r.get("operating_income"), r.get("revenue")) for r in rows[:8]]

    ni_ttm = _ttm(rows, "net_income")
    ocf_ttm = _ttm(rows, "operating_cash_flow")
    rev_ttm = _ttm(rows, "revenue")
    rev_ttm_2y = _ttm(rows, "revenue", offset=8)
    capex_ttm = _ttm(rows, "capex")

    features = {
        "log_revenue": _log10(rev),
        "log_assets": _log10(cur.get("total_assets")),
        "gross_margin": gross_m[0],
        "operating_margin": op_m[0],
        "net_margin": _ratio(cur.get("net_income"), rev),
        "rnd_to_revenue": _ratio(cur.get("rnd"), rev),
        "sga_to_revenue": _ratio(cur.get("sga"), rev),
        "sbc_to_revenue": _ratio(cur.get("sbc"), rev),
        "ocf_margin": _ratio(ocf, rev),
        "fcf_margin": _ratio(ocf - capex, rev),
        "capital_return_to_revenue": _ratio(capital_return, rev),
        "revenue_growth_qoq": _growth(rev, prev_q.get("revenue")),
        "revenue_growth_yoy": growth[0],
        "operating_income_growth_yoy": _growth(cur.get("operating_income"), prev_y.get("operating_income")),
        "net_income_growth_yoy": _growth(cur.get("net_income"), prev_y.get("net_income")),
        "ocf_growth_yoy": _growth(ocf, prev_y.get("operating_cash_flow")),
        "gross_margin_change_yoy": gross_m[0] - _ratio(_gross(prev_y), prev_y.get("revenue")),
        "operating_margin_change_yoy": op_m[0] - _ratio(prev_y.get("operating_income"), prev_y.get("revenue")),
        "current_ratio": _ratio(cur.get("current_assets"), cur.get("current_liabilities")),
        "liabilities_to_assets": _ratio(cur.get("total_liabilities"), cur.get("total_assets")),
        "cash_to_assets": _ratio(cur.get("cash"), cur.get("total_assets")),
        "long_term_debt_to_assets": _ratio(cur.get("long_term_debt"), cur.get("total_assets")),
        "inventory_to_assets": _ratio(cur.get("inventory"), cur.get("total_assets")),
        "receivables_to_revenue": _ratio(cur.get("receivables"), rev),
        "return_on_assets": _ratio(cur.get("net_income"), cur.get("total_assets")),
        # vs. own trend
        "revenue_growth_accel": growth[0] - growth[1],
        "revenue_growth_vs_trend": growth[0] - _mean(growth[1:5], 2),
        "gross_margin_vs_trend": gross_m[0] - _mean(gross_m[1:5], 2),
        "operating_margin_vs_trend": op_m[0] - _mean(op_m[1:5], 2),
        # trend quality
        "revenue_growth_volatility_8q": (
            float(np.std([g for g in growth if not math.isnan(g)]))
            if sum(not math.isnan(g) for g in growth) >= 4 else math.nan
        ),
        "positive_growth_share_8q": (
            float(np.mean([g > 0 for g in growth if not math.isnan(g)]))
            if sum(not math.isnan(g) for g in growth) >= 4 else math.nan
        ),
        "operating_margin_slope_8q": _slope(op_m),
        "revenue_cagr_2y": (
            math.sqrt(rev_ttm / rev_ttm_2y) - 1.0
            if rev_ttm > 0 and rev_ttm_2y > 0 else math.nan
        ),
        "accruals_ttm": _ratio(ni_ttm - ocf_ttm, cur.get("total_assets")),
    }
    extras = {
        "revenue_ttm": rev_ttm,
        "net_income_ttm": ni_ttm,
        "fcf_ttm": ocf_ttm - capex_ttm,
        "book_equity": _num(cur.get("total_assets")) - _num(cur.get("total_liabilities")),
        "shares": _num(cur.get("shares")),
    }
    return features, extras


# ------------------------------------------------------------------ prices
def _baseline_position(prices: pd.DataFrame, announced_at: pd.Timestamp) -> int | None:
    """Index of the last close strictly before the announcement."""
    local = announced_at.tz_convert(ET).tz_localize(None)
    last_date = local.normalize()
    if local.hour < MARKET_CLOSE_HOUR:
        last_date -= pd.Timedelta(days=1)
    pos = int(prices.index.searchsorted(last_date, side="right")) - 1
    if pos < 0 or (last_date - prices.index[pos]).days > 7:
        return None  # no price near the announcement (pre-IPO, data gap)
    return pos


class _Returns:
    """Stock returns in excess of the benchmark over the same trading days."""

    def __init__(self, prices: pd.DataFrame, benchmark: pd.Series):
        self.stock = prices["adj_close"].to_numpy()
        self.bench = benchmark.reindex(prices.index, method="ffill").to_numpy()
        self.n = len(self.stock)

    def raw(self, a: int, b: int) -> float:
        if a < 0 or b >= self.n:
            return math.nan
        return (self.stock[b] / self.stock[a] - 1.0) * 100.0

    def excess(self, a: int, b: int) -> float:
        if a < 0 or b >= self.n:
            return math.nan
        bench = self.bench[b] / self.bench[a] - 1.0
        if not math.isfinite(bench):
            return math.nan
        return (self.stock[b] / self.stock[a] - 1.0 - bench) * 100.0


def _price_features(returns: _Returns, prices: pd.DataFrame, pos: int) -> dict:
    window = prices["adj_close"].iloc[max(0, pos - 21): pos + 1].pct_change().dropna()
    return {
        "pre_excess_return_21d": returns.excess(pos - 21, pos),
        "pre_excess_return_63d": returns.excess(pos - 63, pos),
        "pre_excess_return_252d": returns.excess(pos - 252, pos),
        "pre_volatility_21d": float(window.std() * 100.0) if len(window) > 5 else math.nan,
        "day1_excess_return": returns.excess(pos, pos + 1),
        # Raw momentum, only for models trained before excess returns.
        "pre_return_21d": returns.raw(pos - 21, pos),
        "pre_return_63d": returns.raw(pos - 63, pos),
    }


def _split_factor(splits: pd.Series, after: pd.Timestamp) -> float:
    """How many of today's shares one share at `after` became. Yahoo's
    close is back-adjusted for these splits; the filing's share count isn't."""
    if splits.empty:
        return 1.0
    return float(splits[splits.index > after].prod())


def _valuation_features(extras: dict, raw_close: float, split_factor: float) -> tuple[dict, float]:
    market_cap = _num(raw_close) * split_factor * extras["shares"]
    if not market_cap > 0:
        market_cap = math.nan
    return {
        "log_market_cap": _log10(market_cap),
        "sales_yield": _ratio(extras["revenue_ttm"], market_cap),
        "earnings_yield": _ratio(extras["net_income_ttm"], market_cap),
        "fcf_yield": _ratio(extras["fcf_ttm"], market_cap),
        "book_to_market": _ratio(extras["book_equity"], market_cap),
    }, market_cap


def _window(label: str, pos: int) -> tuple[int, int]:
    """(start, end) price positions of a horizon's window: the 1d window is
    the release reaction, the rest start at the first post-release close."""
    start = pos if label == "1d" else pos + 1
    return start, start + config.HORIZONS[label]


def _targets(returns: _Returns, prices: pd.DataFrame, pos: int) -> dict:
    """target_<h> (excess %, or None when the window isn't complete yet) and
    label_end_<h> (date the label is measured on) for every horizon."""
    out = {}
    for label in config.HORIZONS:
        start, end = _window(label, pos)
        if end < returns.n:
            out[f"target_{label}"] = returns.excess(start, end)
            out[f"label_end_{label}"] = prices.index[end].strftime("%Y-%m-%d")
        else:
            out[f"target_{label}"] = None
            out[f"label_end_{label}"] = None
    return out


# ---------------------------------------------------------------- assembly
@dataclass
class _EventResult:
    features: dict
    pos: int
    market_cap: float


def _event_features(
    event: EarningsEvent,
    data: SymbolData,
    prices: pd.DataFrame,
    returns: _Returns,
    splits: pd.Series,
) -> _EventResult | None:
    fundamentals = _fundamental_features(data.table, event.period_end)
    if fundamentals is None:
        return None
    pos = _baseline_position(prices, event.announced_at)
    if pos is None:
        return None
    features, extras = fundamentals
    announced = event.announced_at.tz_localize(None)
    valuation, market_cap = _valuation_features(
        extras, prices["close"].iloc[pos], _split_factor(splits, announced)
    )
    sector = sector_group(data.profile.get("sic"))
    features = {
        **features,
        **valuation,
        "report_lag_days": float((announced.normalize() - event.period_end).days),
        **_price_features(returns, prices, pos),
        **{f"sector_{s}": float(s == sector) for s in SECTORS},
    }
    return _EventResult(features, pos, market_cap)


def _load_symbol(symbol: str, sec: SECClient) -> SymbolData:
    cik = sec.get_cik(symbol)
    filings = sec.get_filings(cik)
    table = _fundamentals_table(sec.get_company_facts(cik))
    return SymbolData(cik, sec.get_company_profile(cik), _earnings_events(filings), table, _earnings_releases(filings))


def _splits_or_none(symbol: str, price_client: PriceClient, log) -> pd.Series:
    try:
        return price_client.get_splits(symbol)
    except PriceAPIError as exc:
        log("warning", symbol, "fetch_splits", f"No split history, assuming none: {exc}", exc)
        return pd.Series(dtype="float64")


def get_benchmark(price_client: PriceClient) -> pd.Series:
    tomorrow = (pd.Timestamp.now(tz=ET).tz_localize(None) + timedelta(days=1)).strftime("%Y-%m-%d")
    bench = price_client.get_adjusted_closes(config.BENCHMARK_SYMBOL, "2007-01-01", tomorrow)
    if bench.empty:
        raise PriceAPIError(f"No price history for benchmark {config.BENCHMARK_SYMBOL}")
    return bench


def period_quarter(period_end: pd.Timestamp) -> str:
    return str(pd.Timestamp(period_end).to_period("Q"))


def build_training_dataset(
    symbols: list[str],
    sec: SECClient,
    price_client: PriceClient,
    progress_cb=None,
) -> PipelineResult:
    events_log: list[PipelineEvent] = []
    all_rows: list[dict] = []

    def log(level, symbol, stage, message, exc=None):
        events_log.append(
            PipelineEvent(level, symbol, stage, message, traceback.format_exc() if exc else None)
        )
        {"error": logger.error, "warning": logger.warning}.get(level, logger.info)(
            "[%s] %s: %s", symbol, stage, message
        )

    benchmark = get_benchmark(price_client)

    for i, symbol in enumerate(symbols):
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        if progress_cb:
            progress_cb(f"Fetching {symbol} ({i + 1}/{len(symbols)})", (i + 1) / len(symbols))

        try:
            data = _load_symbol(symbol, sec)
        except SECAPIError as exc:
            log("error", symbol, "fetch_filings", str(exc), exc)
            continue
        if not data.earnings or data.table.empty:
            log("warning", symbol, "fetch_filings", "No 10-Q/10-K filings with XBRL financials found")
            continue

        # 252-day momentum needs ~a year of prices before the first report.
        first = min(e.announced_at for e in data.earnings).tz_localize(None) - timedelta(days=400)
        last = max(e.announced_at for e in data.earnings).tz_localize(None) + timedelta(days=400)
        try:
            prices = price_client.get_prices(symbol, first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"))
        except PriceAPIError as exc:
            log("error", symbol, "fetch_prices", str(exc), exc)
            continue
        if prices.empty:
            log("warning", symbol, "fetch_prices", "No price history returned")
            continue
        splits = _splits_or_none(symbol, price_client, log)
        returns = _Returns(prices, benchmark)
        sector = sector_group(data.profile.get("sic"))

        matched = skipped_no_features = skipped_no_targets = 0
        for event in data.earnings:
            result = _event_features(event, data, prices, returns, splits)
            if result is None:
                skipped_no_features += 1
                continue
            targets = _targets(returns, prices, result.pos)
            if targets["target_1d"] is None:
                skipped_no_targets += 1
                continue

            all_rows.append({
                "symbol": symbol,
                "sector": sector,
                "earnings_date": event.announced_at.strftime("%Y-%m-%d"),
                "period_end": event.period_end.strftime("%Y-%m-%d"),
                "period_quarter": period_quarter(event.period_end),
                "baseline_price_date": prices.index[result.pos].strftime("%Y-%m-%d"),
                **result.features,
                **targets,
            })
            matched += 1

        log(
            "info", symbol, "assemble",
            f"{matched} usable earnings events, "
            f"{skipped_no_features} skipped (no XBRL financials or no price at announcement), "
            f"{skipped_no_targets} skipped (no price after the announcement yet)",
        )

    if not all_rows:
        return PipelineResult(dataframe=pd.DataFrame(), events=events_log, feature_columns=[])

    df = pd.DataFrame(all_rows)
    feature_columns = [c for c in FEATURE_COLUMNS if c in df and df[c].notna().any()]
    return PipelineResult(dataframe=df, events=events_log, feature_columns=feature_columns)


METRIC_LABELS = {
    "revenue": "Revenue",
    "cost_of_revenue": "Cost of revenue",
    "gross_profit": "Gross profit",
    "operating_income": "Operating income",
    "net_income": "Net income",
    "rnd": "R&D expense",
    "sga": "SG&A expense",
    "sbc": "Stock-based compensation",
    "operating_cash_flow": "Operating cash flow",
    "capex": "Capital expenditure",
    "buybacks": "Share buybacks",
    "dividends_paid": "Dividends paid",
    "total_assets": "Total assets",
    "current_assets": "Current assets",
    "current_liabilities": "Current liabilities",
    "total_liabilities": "Total liabilities",
    "cash": "Cash & equivalents",
    "long_term_debt": "Long-term debt",
    "inventory": "Inventory",
    "receivables": "Receivables",
    "shares": "Diluted shares (avg)",
}


def _report_source(
    symbol: str,
    data: SymbolData,
    event: EarningsEvent,
    prices: pd.DataFrame,
    returns: _Returns,
    result: _EventResult,
) -> dict:
    """The filing a prediction is based on, plus the raw XBRL figures behind
    its features (this quarter vs. the same quarter a year earlier)."""
    f = event.periodic_filing
    accession = f.get("accessionNumber", "")
    url = None
    if accession and f.get("primaryDocument"):
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{data.cik}/"
            f"{accession.replace('-', '')}/{f['primaryDocument']}"
        )
    filed_at = _acceptance_time(f) if f else None

    cur = _row_near(data.table, event.period_end, 0)
    prev_y = _row_near(data.table, event.period_end - pd.Timedelta(days=364), 15)
    reported = []
    for metric, label in METRIC_LABELS.items():
        value = _num(cur.get(metric)) if cur is not None else math.nan
        prior = _num(prev_y.get(metric)) if prev_y is not None else math.nan
        if math.isnan(value) and math.isnan(prior):
            continue
        reported.append({
            "metric": metric,
            "label": label,
            "kind": "shares" if metric == "shares" else "duration" if metric in DURATION_METRICS else "instant",
            "value": None if math.isnan(value) else value,
            "prior_year_value": None if math.isnan(prior) else prior,
        })

    pos = result.pos
    day1_date = prices.index[pos + 1].strftime("%Y-%m-%d") if pos + 1 < len(prices) else None
    return {
        "horizon_windows": _horizon_windows(returns, prices, pos),
        "symbol": symbol,
        "company": data.profile.get("name"),
        "cik": data.cik,
        "sic": data.profile.get("sic"),
        "sic_description": data.profile.get("sic_description"),
        "sector": sector_group(data.profile.get("sic")),
        "period_end": event.period_end.strftime("%Y-%m-%d"),
        "period_quarter": period_quarter(event.period_end),
        "prior_year_period_end": prev_y.name.strftime("%Y-%m-%d") if prev_y is not None else None,
        "form": f.get("form"),
        "filed_date": f.get("filingDate"),
        "filed_at": filed_at.strftime("%Y-%m-%d %H:%M %Z") if filed_at is not None else None,
        "accession_number": accession or None,
        "filing_url": url,
        "announced_at": event.announced_at.strftime("%Y-%m-%d %H:%M %Z"),
        "announced_via": "8-K Item 2.02 press release" if event.source_form == "8-K" else f"{event.source_form} filing",
        "baseline_price_date": prices.index[pos].strftime("%Y-%m-%d"),
        "day1_date": day1_date,
        "market_cap": None if math.isnan(result.market_cap) else result.market_cap,
        "benchmark": config.BENCHMARK_SYMBOL,
        "reported": reported,
    }


def _horizon_windows(returns: _Returns, prices: pd.DataFrame, pos: int) -> dict:
    """Per horizon: whether its window has already closed, the date it
    closes (estimated in business days while still open), and, once closed,
    what the stock actually did vs. the benchmark."""
    out = {}
    last = len(prices) - 1
    for label in config.HORIZONS:
        start, end = _window(label, pos)
        if end <= last:
            actual = returns.excess(start, end)
            out[label] = {
                "closed": True,
                "window_end": prices.index[end].strftime("%Y-%m-%d"),
                "actual_excess_pct": None if math.isnan(actual) else round(actual, 3),
            }
        else:
            out[label] = {
                "closed": False,
                "window_end": (prices.index[last] + pd.offsets.BDay(end - last)).strftime("%Y-%m-%d"),
                "actual_excess_pct": None,
            }
    return out


def estimate_window_end(earnings_date: str, horizon: str) -> str:
    """Approximate close of a horizon's window from just the release date,
    for predictions saved before windows were recorded."""
    first_close = pd.Timestamp(earnings_date) + pd.offsets.BDay(1)
    extra = 0 if horizon == "1d" else config.HORIZONS[horizon]
    return (first_close + pd.offsets.BDay(extra)).strftime("%Y-%m-%d")


def annotate_windows(decisions: dict, earnings_date: str | None) -> dict:
    """Marks each decision `closed` if its window has passed as of today,
    filling in an estimated window end for older predictions that didn't
    record one. Returns the same dict."""
    today = pd.Timestamp.now(tz=ET).strftime("%Y-%m-%d")
    for h, d in (decisions or {}).items():
        if not d.get("window_end") and earnings_date and h in config.HORIZONS:
            d["window_end"] = estimate_window_end(earnings_date, h)
        d["closed"] = bool(d.get("window_closed")) or bool(d.get("window_end") and d["window_end"] <= today)
    return decisions


def fetch_latest_report_features(
    symbol: str,
    sec: SECClient,
    price_client: PriceClient,
) -> tuple[str | None, dict, str | None, dict | None]:
    """Features for the most recent quarterly report that has XBRL financials.

    Returns (earnings_date, features, note, source). `note` warns when a newer
    earnings release exists whose 10-Q/10-K hasn't been filed yet (so the
    prediction uses the previous report), or when the first trading day after
    the release hasn't closed yet. `source` describes the filing and the raw
    figures the features were built from.
    """
    symbol = symbol.strip().upper()
    data = _load_symbol(symbol, sec)
    if not data.earnings or data.table.empty:
        return None, {}, None, None

    today = pd.Timestamp.now(tz=ET).tz_localize(None).normalize()
    prices = price_client.get_prices(
        symbol,
        (today - timedelta(days=520)).strftime("%Y-%m-%d"),
        (today + timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if prices.empty:
        raise PriceAPIError(f"No recent price history for {symbol}")
    splits = price_client.get_splits(symbol)
    returns = _Returns(prices, get_benchmark(price_client))

    for event in reversed(data.earnings):
        result = _event_features(event, data, prices, returns, splits)
        if result is None:
            continue
        notes = []
        latest = data.earnings[-1]
        if latest.period_end > event.period_end:
            notes.append(
                f"{symbol} filed its {latest.periodic_filing.get('form', '10-Q')} for the quarter ended "
                f"{latest.period_end.strftime('%Y-%m-%d')} on {latest.periodic_filing.get('filingDate')}, but "
                f"EDGAR's XBRL data for it isn't available yet, so this uses the previous report."
            )
        elif data.releases and data.releases[-1] > event.announced_at:
            notes.append(
                f"{symbol} released earnings on {data.releases[-1].strftime('%Y-%m-%d')} but hasn't filed "
                f"the 10-Q/10-K with XBRL financials yet, so this uses the previous report."
            )
        if math.isnan(result.features["day1_excess_return"]):
            notes.append(
                "The first trading day after this release hasn't closed yet, so the 1w/1m/1y "
                "calls use the typical first-day reaction instead of the real one."
            )
        source = _report_source(symbol, data, event, prices, returns, result)
        return event.announced_at.strftime("%Y-%m-%d"), result.features, " ".join(notes) or None, source

    return None, {}, None, None
