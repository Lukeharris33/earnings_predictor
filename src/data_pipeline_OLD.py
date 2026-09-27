"""
Turns raw FMP responses into a flat, model-ready table:

    one row per (symbol, earnings report date)
    columns  = every numeric field from the earnings + income statement +
               balance sheet + cash flow statement (the full earnings report)
    targets  = forward % price change at 1d / 1w / 1m / 1y (trading days)

No lookahead bias: features come only from the statement filed at/around the
earnings date; the price baseline is the first trading day on/after the
earnings date (the earliest point you could actually act on the numbers).
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from .config import config
from .fmp_client import FMPClient, FMPAPIError

logger = logging.getLogger(__name__)

# Columns that are identifiers / metadata rather than model features, even
# though some of them are technically numeric-looking (e.g. "cik").
NON_FEATURE_COLUMNS = {
    "date", "symbol", "reportedCurrency", "cik", "fillingDate", "filingDate",
    "acceptedDate", "calendarYear", "period", "link", "finalLink", "lastUpdated",
}


@dataclass
class PipelineEvent:
    """One log line describing something that happened while building the dataset."""
    level: str  # "info" | "warning" | "error"
    symbol: str
    stage: str
    message: str
    traceback: str | None = None


@dataclass
class PipelineResult:
    dataframe: pd.DataFrame
    events: list[PipelineEvent] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)


def _parse_date(s: str | None):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _merge_statement_rows(income: dict, balance: dict | None, cashflow: dict | None) -> dict:
    merged = dict(income)
    for extra in (balance, cashflow):
        if extra:
            for k, v in extra.items():
                if k not in merged:
                    merged[k] = v
    return merged


def _index_statements_by_filing_date(rows: list[dict]) -> list[tuple[datetime, dict]]:
    indexed = []
    for row in rows:
        filing = _parse_date(row.get("fillingDate") or row.get("filingDate") or row.get("date"))
        if filing:
            indexed.append((filing, row))
    return indexed


def _find_nearest_statement(target_date: datetime, indexed_statements, window_days: int):
    best = None
    best_diff = None
    for filing_date, row in indexed_statements:
        diff = abs((filing_date - target_date).days)
        if diff <= window_days and (best_diff is None or diff < best_diff):
            best, best_diff = row, diff
    return best


def _build_price_series(prices: list[dict]) -> pd.Series:
    if not prices:
        return pd.Series(dtype="float64")
    rows = []
    for p in prices:
        d = _parse_date(p.get("date"))
        price = p.get("adjClose", p.get("close"))
        if d is not None and price is not None:
            rows.append((d, float(price)))
    if not rows:
        return pd.Series(dtype="float64")
    rows.sort(key=lambda r: r[0])
    idx, vals = zip(*rows)
    return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()


def _forward_returns(earnings_date: datetime, price_series: pd.Series):
    """Returns (baseline_date, {horizon_label: pct_change or None})."""
    if price_series.empty:
        return None, {h: None for h in config.HORIZONS}
    on_or_after = price_series.index[price_series.index >= earnings_date]
    if len(on_or_after) == 0:
        return None, {h: None for h in config.HORIZONS}
    baseline_date = on_or_after[0]
    baseline_pos = price_series.index.get_loc(baseline_date)
    baseline_price = price_series.iloc[baseline_pos]

    results = {}
    for label, trading_days in config.HORIZONS.items():
        target_pos = baseline_pos + trading_days
        if target_pos < len(price_series):
            future_price = price_series.iloc[target_pos]
            results[label] = (future_price - baseline_price) / baseline_price * 100.0
        else:
            results[label] = None
    return baseline_date, results


def build_training_dataset(symbols: list[str], fmp: FMPClient, progress_cb=None) -> PipelineResult:
    events: list[PipelineEvent] = []
    all_rows: list[dict] = []

    def log(level, symbol, stage, message, exc=None):
        events.append(PipelineEvent(level, symbol, stage, message, traceback.format_exc() if exc else None))
        (logger.error if level == "error" else logger.warning if level == "warning" else logger.info)(
            "[%s] %s: %s", symbol, stage, message
        )

    for i, symbol in enumerate(symbols):
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        if progress_cb:
            progress_cb(f"Fetching {symbol} ({i + 1}/{len(symbols)})")

        try:
            earnings = fmp.get_earnings(symbol)
        except FMPAPIError as exc:
            log("error", symbol, "fetch_earnings", str(exc), exc)
            continue
        if not earnings:
            log("warning", symbol, "fetch_earnings", "No earnings history returned")
            continue

        try:
            income = fmp.get_income_statement(symbol)
            balance = fmp.get_balance_sheet(symbol)
            cashflow = fmp.get_cash_flow(symbol)
        except FMPAPIError as exc:
            log("error", symbol, "fetch_statements", str(exc), exc)
            continue

        balance_by_date = {r.get("date"): r for r in balance}
        cashflow_by_date = {r.get("date"): r for r in cashflow}
        merged_statements = [
            _merge_statement_rows(r, balance_by_date.get(r.get("date")), cashflow_by_date.get(r.get("date")))
            for r in income
        ]
        indexed_statements = _index_statements_by_filing_date(merged_statements)
        if not indexed_statements:
            log("warning", symbol, "fetch_statements", "No usable financial statements returned")
            continue

        earnings_dates = [d for d in (_parse_date(e.get("date")) for e in earnings) if d]
        if not earnings_dates:
            continue
        price_from = (min(earnings_dates) - timedelta(days=10)).strftime("%Y-%m-%d")
        price_to = (max(earnings_dates) + timedelta(days=380)).strftime("%Y-%m-%d")
        try:
            prices = fmp.get_historical_prices(symbol, price_from, price_to)
        except FMPAPIError as exc:
            log("error", symbol, "fetch_prices", str(exc), exc)
            continue
        price_series = _build_price_series(prices)
        if price_series.empty:
            log("warning", symbol, "fetch_prices", "No price history returned")
            continue

        matched, skipped_no_statement, skipped_no_targets = 0, 0, 0
        for e in earnings:
            e_date = _parse_date(e.get("date"))
            if not e_date:
                continue
            statement = _find_nearest_statement(e_date, indexed_statements, config.STATEMENT_MATCH_WINDOW_DAYS)
            if not statement:
                skipped_no_statement += 1
                continue

            baseline_date, targets = _forward_returns(e_date, price_series)
            if baseline_date is None or any(v is None for v in targets.values()):
                skipped_no_targets += 1
                continue

            eps_actual = e.get("epsActual")
            eps_estimated = e.get("epsEstimated")
            rev_actual = e.get("revenueActual")
            rev_estimated = e.get("revenueEstimated")

            row = {
                "symbol": symbol,
                "earnings_date": e_date.strftime("%Y-%m-%d"),
                "baseline_price_date": baseline_date.strftime("%Y-%m-%d"),
                "epsActual": eps_actual,
                "epsEstimated": eps_estimated,
                "epsSurprise": (eps_actual - eps_estimated) if (eps_actual is not None and eps_estimated is not None) else None,
                "revenueActual": rev_actual,
                "revenueEstimated": rev_estimated,
                "revenueSurprise": (rev_actual - rev_estimated) if (rev_actual is not None and rev_estimated is not None) else None,
            }
            for k, v in statement.items():
                if k in NON_FEATURE_COLUMNS or k in row:
                    continue
                row[k] = v
            for label in config.HORIZONS:
                row[f"target_{label}"] = targets[label]

            all_rows.append(row)
            matched += 1

        log(
            "info", symbol, "assemble",
            f"{matched} usable earnings events, {skipped_no_statement} skipped (no matching statement), "
            f"{skipped_no_targets} skipped (incomplete forward price history)",
        )

    if not all_rows:
        return PipelineResult(dataframe=pd.DataFrame(), events=events, feature_columns=[])

    df = pd.DataFrame(all_rows)
    target_cols = [f"target_{label}" for label in config.HORIZONS]
    meta_cols = {"symbol", "earnings_date", "baseline_price_date"}
    numeric_df = df.drop(columns=list(meta_cols)).apply(pd.to_numeric, errors="coerce")
    feature_columns = [c for c in numeric_df.columns if c not in target_cols]
    # Drop columns that are entirely null -- they carry zero signal and would
    # otherwise just get imputed to a constant everywhere.
    feature_columns = [c for c in feature_columns if numeric_df[c].notna().any()]

    df = pd.concat([df[list(meta_cols)], numeric_df], axis=1)
    return PipelineResult(dataframe=df, events=events, feature_columns=feature_columns)


def fetch_latest_report_features(symbol: str, fmp: FMPClient) -> tuple[str | None, dict]:
    """Same statement-matching logic as build_training_dataset, but for a
    single symbol's most recent earnings report -- used at prediction time,
    when there's no future price to look up yet."""
    symbol = symbol.strip().upper()
    earnings = fmp.get_earnings(symbol, limit=4)
    if not earnings:
        return None, {}
    latest = max(earnings, key=lambda e: e.get("date") or "")
    e_date = _parse_date(latest.get("date"))
    if not e_date:
        return None, {}

    income = fmp.get_income_statement(symbol, limit=4)
    balance = fmp.get_balance_sheet(symbol, limit=4)
    cashflow = fmp.get_cash_flow(symbol, limit=4)
    balance_by_date = {r.get("date"): r for r in balance}
    cashflow_by_date = {r.get("date"): r for r in cashflow}
    merged_statements = [
        _merge_statement_rows(r, balance_by_date.get(r.get("date")), cashflow_by_date.get(r.get("date")))
        for r in income
    ]
    indexed_statements = _index_statements_by_filing_date(merged_statements)
    statement = _find_nearest_statement(e_date, indexed_statements, config.STATEMENT_MATCH_WINDOW_DAYS)
    if not statement:
        return e_date.strftime("%Y-%m-%d"), {}

    eps_actual, eps_estimated = latest.get("epsActual"), latest.get("epsEstimated")
    rev_actual, rev_estimated = latest.get("revenueActual"), latest.get("revenueEstimated")
    row = {
        "epsActual": eps_actual,
        "epsEstimated": eps_estimated,
        "epsSurprise": (eps_actual - eps_estimated) if (eps_actual is not None and eps_estimated is not None) else None,
        "revenueActual": rev_actual,
        "revenueEstimated": rev_estimated,
        "revenueSurprise": (rev_actual - rev_estimated) if (rev_actual is not None and rev_estimated is not None) else None,
    }
    for k, v in statement.items():
        if k in NON_FEATURE_COLUMNS or k in row:
            continue
        row[k] = v
    return e_date.strftime("%Y-%m-%d"), row
