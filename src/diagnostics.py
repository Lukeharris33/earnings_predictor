"""
Model diagnostics: what a model was trained on (its profile) and which kinds
of input it actually relies on (shuffle importance per feature group, on the
walk-forward years it hadn't seen).

New models get this at training time. Older v2 models can backfill it: the
training rows saved in Supabase are replayed through the same walk-forward
evaluation (no EDGAR/price downloads). Two things are approximated in a
backfill because the stored rows don't carry them: each label's end date
(business days from the release instead of exchange trading days) and the
fiscal quarter (release date minus the reporting lag). Both only affect
which rows are purged at fold edges and which peers a row is ranked
against, so the replayed metrics land close to, not exactly on, the
originals. The saved model's headline metrics are never overwritten.
"""
from __future__ import annotations

import logging
import threading
import traceback
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import config
from .data_pipeline import SECTOR_FEATURES
from .supabase_client import SupabaseLogger

logger = logging.getLogger(__name__)

THIN_TICKER_ROWS = 8

JOBS: dict[str, dict] = {}
_lock = threading.Lock()


def _set_status(job_id: str, **fields):
    with _lock:
        JOBS.setdefault(job_id, {}).update(fields)


def get_status(job_id: str) -> dict | None:
    with _lock:
        return dict(JOBS[job_id]) if job_id in JOBS else None


# ------------------------------------------------------------------ profile
def training_profile(df: pd.DataFrame, symbols: list[str], run_config: dict, features_by_horizon: dict) -> dict:
    """What the model learned from: breadth, history, company mix, settings."""
    dates = pd.to_datetime(df["earnings_date"])
    net_margin = df["net_margin"].dropna() if "net_margin" in df else pd.Series(dtype=float)
    mcap = (10 ** df["log_market_cap"].dropna()) if "log_market_cap" in df else pd.Series(dtype=float)
    growth = df["revenue_growth_yoy"].dropna() if "revenue_growth_yoy" in df else pd.Series(dtype=float)
    per_ticker = df.groupby("symbol").size()
    return {
        "tickers_requested": len(symbols),
        "tickers_used": int(df["symbol"].nunique()),
        "rows": int(len(df)),
        "rows_per_ticker_median": float(per_ticker.median()) if len(per_ticker) else None,
        # Tickers that barely contributed (e.g. few quarters with usable XBRL data).
        "thin_tickers": sorted(per_ticker[per_ticker < THIN_TICKER_ROWS].index.tolist()),
        "first_report": dates.min().strftime("%Y-%m-%d") if len(dates) else None,
        "last_report": dates.max().strftime("%Y-%m-%d") if len(dates) else None,
        "years_of_history": round((dates.max() - dates.min()).days / 365.25, 1) if len(dates) else None,
        "sector_mix": {k: round(float(v), 4) for k, v in df["sector"].value_counts(normalize=True).items()},
        "sectors_covered": int(df["sector"].nunique()),
        "loss_making_share": round(float((net_margin < 0).mean()), 4) if len(net_margin) else None,
        "median_market_cap": float(mcap.median()) if len(mcap) else None,
        "small_cap_share": round(float((mcap < 2e9).mean()), 4) if len(mcap) else None,
        "median_revenue_growth_yoy": float(growth.median()) if len(growth) else None,
        "model_type": run_config.get("model_type"),
        "epochs": run_config.get("epochs"),
        "batch_size": run_config.get("batch_size"),
        "feature_count": max((len(v) for v in features_by_horizon.values()), default=0),
    }


def build_diagnostics(profile: dict, importance: dict, method: str, replayed: dict | None = None) -> dict:
    return {
        "profile": profile,
        "importance": importance,
        "replayed_metrics": replayed or {},
        "method": method,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


# ----------------------------------------------------------------- backfill
def frame_from_training_rows(rows: list[dict]) -> pd.DataFrame:
    """Rebuilds the trainer's dataframe from stored training_data rows."""
    records = []
    for r in rows:
        rec = {"symbol": r["symbol"], "earnings_date": r["earnings_date"], **(r.get("features") or {})}
        for h, v in (r.get("targets") or {}).items():
            rec[f"target_{h}"] = v
        records.append(rec)
    df = pd.DataFrame(records)
    released = pd.to_datetime(df["earnings_date"])

    sector_cols = [c for c in SECTOR_FEATURES if c in df]
    if sector_cols:
        onehot = df[sector_cols].fillna(0)
        df["sector"] = np.where(onehot.max(axis=1) > 0, onehot.idxmax(axis=1).str.removeprefix("sector_"), "other")
    else:
        df["sector"] = "other"

    lag = pd.to_timedelta(df.get("report_lag_days", pd.Series(30.0, index=df.index)).fillna(30.0), unit="D")
    df["period_quarter"] = (released - lag).dt.to_period("Q").astype(str)

    for h, n in config.HORIZONS.items():
        if f"target_{h}" not in df:
            df[f"target_{h}"] = np.nan
        offset = pd.offsets.BDay(n if h == "1d" else n + 1)
        df[f"label_end_{h}"] = (released + offset).dt.strftime("%Y-%m-%d")
        df.loc[df[f"target_{h}"].isna(), f"label_end_{h}"] = None
    return df.sort_values(["earnings_date", "symbol"]).reset_index(drop=True)


def start_backfill(model_id: str) -> str:
    job_id = str(uuid.uuid4())
    _set_status(job_id, done=False, error=False, progress=0.0, message="Starting...", model_id=model_id)
    threading.Thread(target=_backfill, args=(job_id, model_id), daemon=True).start()
    return job_id


def run_backfill_sync(model_id: str) -> dict:
    job_id = str(uuid.uuid4())
    _set_status(job_id, done=False, error=False, progress=0.0, message="Starting...", model_id=model_id)
    _backfill(job_id, model_id)
    return get_status(job_id)


def _backfill(job_id: str, model_id: str):
    from .trainer import _evaluate_horizon  # heavy (TensorFlow); import only when used

    try:
        db = SupabaseLogger()
        model = db.get_model(model_id)
        if not isinstance(model.get("feature_columns"), dict):
            raise ValueError("Only models trained with per-horizon walk-forward evaluation (v2) can be diagnosed.")
        run = db.get_training_run(model["run_id"]) if model.get("run_id") else {}
        cfg = (run or {}).get("config") or {}
        model_type = cfg.get("model_type") or config.DEFAULT_MODEL_TYPE
        epochs = int(cfg.get("epochs") or config.DEFAULT_EPOCHS)
        batch_size = int(cfg.get("batch_size") or config.DEFAULT_BATCH_SIZE)

        def fetched(n):
            _set_status(job_id, progress=0.02, message=f"Loading stored training rows from Supabase ({n:,} so far)...")

        rows = db.fetch_training_data(model["run_id"], progress=fetched)
        if not rows:
            raise ValueError("No stored training rows for this model's run.")
        df = frame_from_training_rows(rows)

        features_by_h = model["feature_columns"]
        importance, replayed = {}, {}
        horizons = [h for h in config.HORIZONS if h in features_by_h]
        for i, h in enumerate(horizons):
            base = 0.08 + 0.88 * i / len(horizons)

            def status(msg, base=base):
                _set_status(job_id, progress=base, message=f"Replaying walk-forward · {msg}")

            hdf = df[df[f"target_{h}"].notna()].reset_index(drop=True)
            cols = [c for c in features_by_h[h] if c in hdf]
            result = _evaluate_horizon(hdf, cols, h, model_type, epochs, batch_size, status)
            importance[h] = result.get("importance", {})
            replayed[h] = {k: result.get(k) for k in ("n", "ic", "directional_accuracy")}

        profile = training_profile(df, model.get("symbols") or [], cfg, features_by_h)
        metrics = dict(model.get("metrics") or {})
        metrics["_diagnostics"] = build_diagnostics(
            profile, importance, "backfilled: stored training rows replayed through walk-forward", replayed,
        )
        db.update_model_metrics(model_id, metrics)
        _set_status(job_id, done=True, progress=1.0, message="Diagnostics saved.")
    except Exception as exc:
        logger.exception("Diagnostics backfill failed for model %s", model_id)
        _set_status(job_id, done=True, error=True, message=str(exc), traceback=traceback.format_exc())
