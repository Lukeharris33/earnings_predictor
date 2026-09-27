"""
Batch predictions: score many tickers with one model under a single group
name, on a background thread the UI can poll. Every prediction is saved to
Supabase with the batch's id and name, and the batch row gets a summary
(BUY/SELL/HOLD counts per horizon, failures) when it finishes.
"""
from __future__ import annotations

import logging
import threading
import traceback
import uuid
from collections import Counter

import numpy as np

from .predictor import predict_for_symbol
from .price_client import PriceClient
from .sec_client import SECClient
from .supabase_client import SupabaseLogger

logger = logging.getLogger(__name__)

JOBS: dict[str, dict] = {}
_lock = threading.Lock()


def _set_status(job_id: str, **fields):
    with _lock:
        JOBS.setdefault(job_id, {}).update(fields)


def get_status(job_id: str) -> dict | None:
    with _lock:
        job = JOBS.get(job_id)
        return None if job is None else {**job, "results": list(job.get("results", []))}


def start_batch(model_id: str, symbols: list[str], name: str) -> str:
    job_id = str(uuid.uuid4())
    _set_status(job_id, done=False, progress=0.0, message="Starting...", results=[], failures=[])
    threading.Thread(target=_run, args=(job_id, model_id, symbols, name), daemon=True).start()
    return job_id


def run_batch_sync(model_id: str, symbols: list[str], name: str) -> dict:
    job_id = str(uuid.uuid4())
    _set_status(job_id, done=False, progress=0.0, message="Starting...", results=[], failures=[])
    _run(job_id, model_id, symbols, name)
    return get_status(job_id)


def summarize(results: list[dict], failures: list[dict], n_symbols: int) -> dict:
    """BUY/SELL/HOLD counts per horizon over the windows still open; horizons
    whose window had already closed are counted as CLOSED instead."""
    counts: dict[str, Counter] = {}
    moves: dict[str, list[float]] = {}
    for r in results:
        for h, d in r["decisions"].items():
            if d.get("window_closed"):
                counts.setdefault(h, Counter())["CLOSED"] += 1
                continue
            counts.setdefault(h, Counter())[d["action"]] += 1
            moves.setdefault(h, []).append(d["predicted_pct_change"])
    return {
        "n_symbols": n_symbols,
        "n_ok": len(results),
        "n_failed": len(failures),
        "failures": failures,
        "counts": {h: {a: c.get(a, 0) for a in ("BUY", "SELL", "HOLD", "CLOSED")} for h, c in counts.items()},
        "mean_predicted_pct": {h: round(float(np.mean(v)), 3) for h, v in moves.items()},
    }


def _run(job_id: str, model_id: str, symbols: list[str], name: str):
    db = batch_id = None
    results, failures = [], []
    try:
        db = SupabaseLogger()
        batch_id = db.create_batch(name, model_id, symbols)
        _set_status(job_id, batch_id=batch_id, name=name)
        sec, prices = SECClient(), PriceClient()

        for i, symbol in enumerate(symbols):
            _set_status(job_id, progress=i / len(symbols), message=f"Scoring {symbol} ({i + 1}/{len(symbols)})")
            try:
                r = predict_for_symbol(model_id, symbol, name=name, batch_id=batch_id, db=db, sec=sec, price_client=prices)
                results.append({"symbol": r["symbol"], "earnings_date": r["earnings_date"],
                                "decisions": r["decisions"], "note": r["note"]})
            except Exception as exc:
                logger.warning("Batch %s: %s failed: %s", name, symbol, exc)
                failures.append({"symbol": symbol, "error": str(exc)})
                try:
                    db.log_error(None, symbol, "batch_predict", f"[{name}] {exc}", traceback.format_exc())
                except Exception:
                    logger.exception("Failed to log batch error to Supabase")
            _set_status(job_id, results=list(results), failures=list(failures))

        summary = summarize(results, failures, len(symbols))
        db.finish_batch(batch_id, "completed" if results else "failed", summary)
        _set_status(
            job_id, done=True, progress=1.0, summary=summary, error=not results,
            message=f"{name}: {len(results)} scored, {len(failures)} failed.",
        )
    except Exception as exc:
        logger.exception("Batch %s failed", name)
        if db and batch_id:
            try:
                db.finish_batch(batch_id, "failed", summarize(results, failures, len(symbols)) | {"error": str(exc)})
            except Exception:
                logger.exception("Also failed to record the batch failure")
        _set_status(job_id, done=True, error=True, message=str(exc))
