"""
Data for the testing page: a batch's predictions next to what the stocks
have actually done since each release, with the latest traded prices mixed
in so open windows track in (near) real time.

Prices here are split-adjusted closes (not dividend-adjusted) so they match
live quotes; over the short windows this page tracks, dividends barely move
the comparison. Closed windows on predictions made since window tracking was
added also carry the dividend-adjusted actual recorded at prediction time.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from .config import config
from .data_pipeline import ET, annotate_windows
from .price_client import PriceClient
from .supabase_client import SupabaseLogger


def market_open(now: datetime | None = None) -> bool:
    """Regular US session, Mon-Fri 9:30-16:00 ET (exchange holidays aren't checked)."""
    now = now or pd.Timestamp.now(tz=ET)
    minutes = now.hour * 60 + now.minute
    return now.weekday() < 5 and 9 * 60 + 30 <= minutes < 16 * 60


def live_payload(symbols: list[str], price_client: PriceClient) -> dict:
    return {
        "quotes": price_client.get_live_quotes(list(symbols) + [config.BENCHMARK_SYMBOL]),
        "market_open": market_open(),
        "as_of": pd.Timestamp.now(tz=ET).isoformat(),
    }


def _series(prices: pd.Series, quote: dict | None) -> dict:
    """{dates, closes}, with today's point replaced/extended by the live quote."""
    prices = prices.dropna()
    dates = [d.strftime("%Y-%m-%d") for d in prices.index]
    closes = [round(float(v), 4) for v in prices.to_numpy()]
    if quote:
        qdate = quote["time"][:10]
        if dates and dates[-1] == qdate:
            closes[-1] = round(quote["price"], 4)
        elif not dates or qdate > dates[-1]:
            dates.append(qdate)
            closes.append(round(quote["price"], 4))
    return {"dates": dates, "closes": closes}


def batch_payload(batch_id: str, db: SupabaseLogger, price_client: PriceClient) -> dict:
    batch = db.get_batch(batch_id)
    predictions = []
    for p in batch["predictions"]:
        details = p.get("details") or {}
        source = details.get("source") or {}
        predictions.append({
            "symbol": p["symbol"],
            "company": source.get("company"),
            "earnings_date": p.get("earnings_date"),
            "baseline_date": source.get("baseline_price_date") or p.get("earnings_date"),
            "day1_date": source.get("day1_date"),
            "decisions": annotate_windows(p.get("decisions"), p.get("earnings_date")),
            "note": details.get("note"),
        })

    symbols = sorted({p["symbol"] for p in predictions})
    baselines = [p["baseline_date"] for p in predictions if p["baseline_date"]]
    start = (pd.Timestamp(min(baselines)) - timedelta(days=10)).strftime("%Y-%m-%d") if baselines else None
    tomorrow = (pd.Timestamp.now(tz=ET) + timedelta(days=1)).strftime("%Y-%m-%d")

    live = live_payload(symbols, price_client) if symbols else {"quotes": {}, "market_open": market_open(), "as_of": None}
    prices, missing = {}, []
    if start:
        for sym in symbols:
            try:
                prices[sym] = _series(price_client.get_prices(sym, start, tomorrow)["close"], live["quotes"].get(sym))
            except Exception as exc:  # one bad ticker shouldn't sink the page
                missing.append({"symbol": sym, "error": str(exc)})
        bench = price_client.get_prices(config.BENCHMARK_SYMBOL, start, tomorrow)["close"]
        benchmark = _series(bench, live["quotes"].get(config.BENCHMARK_SYMBOL))
    else:
        benchmark = {"dates": [], "closes": []}

    return {
        "batch": {
            "id": batch["id"],
            "name": batch["name"],
            "model": (batch.get("models") or {}).get("name"),
            "created_at": batch.get("created_at"),
            "status": batch.get("status"),
        },
        "predictions": predictions,
        "prices": prices,
        "missing_prices": missing,
        "benchmark": {"symbol": config.BENCHMARK_SYMBOL, **benchmark},
        "horizons": config.HORIZONS,
        **{k: live[k] for k in ("market_open", "as_of")},
    }
