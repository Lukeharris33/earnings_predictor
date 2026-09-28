"""
Ties data_pipeline -> preprocessing -> model together into one training run,
runs it on a background thread (so the Flask request that kicks it off
returns immediately), and keeps an in-memory status dict the UI can poll.

Each horizon gets its own model, features and training rows (a row only
needs its own horizon's label, so recent reports still train the short
horizons). Every horizon is evaluated walk-forward: each of the last few
calendar years is predicted by a model trained only on earlier earnings
whose labels had fully resolved before that year began. The model that gets
saved is then retrained on all rows.

Every run, regardless of outcome, is logged to Supabase: the training_runs
row, every row of training data, every warning/error, and (if training
succeeds) the model artifact itself.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import traceback
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import config
from .data_pipeline import FEATURE_GROUPS, RANK_FEATURES, VS_TREND_FEATURES, build_training_dataset, horizon_features
from .diagnostics import build_diagnostics, training_profile
from .model import (
    HORIZON_LABELS, evaluate_predictions, network_learning_rate, predict_network, train_gbm, train_network,
)
from .preprocessing import HorizonTarget, RobustPreprocessor
from .price_client import PriceClient
from .sec_client import SECClient
from .supabase_client import SupabaseLogger

logger = logging.getLogger(__name__)

MODEL_VERSION = 2

# In-memory run status, polled by the frontend via /api/status/<run_id>.
# Fine for a single-process `python app.py` dev deployment; the durable
# record of truth is always the Supabase `training_runs` row.
RUNS: dict[str, dict] = {}
_lock = threading.Lock()


def _set_status(run_id: str, **fields):
    with _lock:
        RUNS.setdefault(run_id, {}).update(fields)


def start_training_run(
    symbols: list[str],
    epochs: int,
    batch_size: int,
    name: str | None = None,
    model_type: str = config.DEFAULT_MODEL_TYPE,
) -> str:
    run_id = str(uuid.uuid4())
    _set_status(run_id, phase="starting", message="Warming up...", progress=0.0, done=False)
    thread = threading.Thread(
        target=_run, args=(run_id, symbols, epochs, batch_size, name, model_type), daemon=True
    )
    thread.start()
    return run_id


def run_training_sync(symbols, epochs, batch_size, name=None, model_type=config.DEFAULT_MODEL_TYPE) -> dict:
    """Same as a UI-started run, but blocks and returns the final status."""
    run_id = str(uuid.uuid4())
    _set_status(run_id, phase="starting", message="Warming up...", progress=0.0, done=False)
    _run(run_id, symbols, epochs, batch_size, name, model_type)
    return get_status(run_id)


# ------------------------------------------------------------ fold planning
def _folds(hdf: pd.DataFrame, horizon: str) -> tuple[list[dict], str]:
    """Walk-forward folds, oldest first: [{label, train_idx, test_idx}].
    Training rows must have their label resolved before the test period
    starts (purging), or they'd have "seen" test-period prices."""
    years = pd.to_datetime(hdf["earnings_date"]).dt.year
    label_end = hdf[f"label_end_{horizon}"]
    folds = []
    for year in sorted(years.unique())[-config.WALK_FORWARD_YEARS:]:
        test_idx = np.flatnonzero(years.to_numpy() == year)
        train_idx = np.flatnonzero((label_end < f"{year}-01-01").to_numpy())
        if len(test_idx) >= config.MIN_FOLD_TEST_ROWS and len(train_idx) >= config.MIN_TRAINING_ROWS:
            folds.append({"label": str(year), "train_idx": train_idx, "test_idx": test_idx})
    if folds:
        return folds, "walk_forward"

    # Too little history for yearly folds: one chronological holdout.
    split = min(max(1, int(len(hdf) * (1 - config.VALIDATION_FRACTION))), len(hdf) - 1)
    test_start = hdf["earnings_date"].iloc[split]
    test_idx = np.arange(split, len(hdf))
    train_idx = np.flatnonzero((label_end < test_start).to_numpy())
    if len(train_idx) < config.MIN_TRAINING_ROWS // 2:
        return [], "none"
    return [{"label": f"from {test_start}", "train_idx": train_idx, "test_idx": test_idx}], "single_split"


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], horizon: str,
                 families: set[str], epochs: int, batch_size: int, seed: int = 0,
                 shuffle_groups: dict[str, list[str]] | None = None,
                 score_family: str | None = None) -> tuple[dict, float, dict]:
    """Fits on `train`. Returns ({family: predictions on test in %}, train mean,
    {group: `score_family` predictions on test with that feature group's
    values shuffled across rows}) -- the last is empty without shuffle_groups."""
    pre = RobustPreprocessor(cols, RANK_FEATURES, config.MIN_PEERS_PER_QUARTER)
    X_train = pre.fit_transform(train[cols], train["period_quarter"])
    target = HorizonTarget()
    y_train = target.fit_transform(train[f"target_{horizon}"].to_numpy(dtype="float64"))

    fitted = {}
    if "gbm" in families:
        fitted["gbm"] = train_gbm(X_train, y_train, seed)
    if "nn" in families:
        fitted["nn"] = train_network(X_train, y_train, seed, epochs, batch_size)

    def predict(frame: pd.DataFrame) -> dict:
        X = pre.transform(frame, test["period_quarter"])
        out = {}
        if "gbm" in fitted:
            out["gbm"] = target.inverse(fitted["gbm"].predict(X))
        if "nn" in fitted:
            out["nn"] = target.inverse(predict_network(fitted["nn"], X))
        if "nn" in out and "gbm" in out:
            out["blend"] = (out["nn"] + out["gbm"]) / 2
        return out

    preds = predict(test[cols])
    shuffled = {}
    rng = np.random.default_rng(seed)
    for group, group_cols in (shuffle_groups or {}).items():
        present = [c for c in group_cols if c in cols]
        if not present:
            continue
        # One permutation for the whole group keeps its columns consistent with each other.
        frame = test[cols].copy()
        frame[present] = frame[present].to_numpy()[rng.permutation(len(frame))]
        shuffled[group] = predict(frame)[score_family]
    if "nn" in fitted:
        tf.keras.backend.clear_session()
    return preds, target.mean, shuffled


def _families(model_type: str) -> set[str]:
    return {"blend": {"nn", "gbm"}, "nn": {"nn"}, "gbm": {"gbm"}}[model_type]


def _evaluate_horizon(hdf, cols, horizon, model_type, epochs, batch_size, status, importance: bool = True) -> dict:
    folds, method = _folds(hdf, horizon)
    if not folds:
        return {"evaluation": "none", "reason": "Not enough history to hold anything out."}

    families = _families(model_type)
    pooled: dict[str, list] = {}
    actual, quarters, train_means, fold_rows = [], [], [], []
    ablation = {"with": [], "without": []}
    shuffled_pool: dict[str, list] = {}
    no_trend_cols = [c for c in cols if c not in VS_TREND_FEATURES]
    run_ablation = horizon == "1d" and len(no_trend_cols) < len(cols)

    for i, fold in enumerate(folds, 1):
        status(f"{horizon}: walk-forward fold {i}/{len(folds)} (testing {fold['label']})")
        train, test = hdf.iloc[fold["train_idx"]], hdf.iloc[fold["test_idx"]]
        preds, train_mean, shuffled = _fit_predict(
            train, test, cols, horizon, families | ({"gbm"} if run_ablation else set()), epochs, batch_size,
            seed=i, shuffle_groups=FEATURE_GROUPS if importance else None, score_family=model_type,
        )
        for group, p in shuffled.items():
            shuffled_pool.setdefault(group, []).append(p)
        y = test[f"target_{horizon}"].to_numpy(dtype="float64")
        q = test["period_quarter"].to_numpy()
        for fam, p in preds.items():
            pooled.setdefault(fam, []).append(p)
        actual.append(y)
        quarters.append(q)
        train_means.append(np.full(len(y), train_mean))

        fold_metrics = evaluate_predictions(preds[model_type], y, q, train_mean)
        fold_rows.append({
            "test_period": fold["label"], "n_train": len(train), "n_test": len(test),
            "directional_accuracy": fold_metrics["directional_accuracy"],
            "baseline_majority_accuracy": fold_metrics["baseline_majority_accuracy"],
            "ic": fold_metrics["ic"],
        })

        if run_ablation:
            ablation["with"].append(preds["gbm"])
            reduced, _, _ = _fit_predict(train, test, no_trend_cols, horizon, {"gbm"}, epochs, batch_size, seed=i)
            ablation["without"].append(reduced["gbm"])

    actual = np.concatenate(actual)
    quarters = np.concatenate(quarters)
    train_mean = float(np.mean(np.concatenate(train_means)))
    reported = families | ({"blend"} if model_type == "blend" else set())
    by_family = {
        fam: evaluate_predictions(np.concatenate(p), actual, quarters, train_mean)
        for fam, p in pooled.items() if fam in reported
    }
    out = {
        **by_family[model_type],  # headline numbers are the saved model type's
        "evaluation": method,
        "model_type": model_type,
        "by_family": by_family,
        "folds": fold_rows,
    }
    if shuffled_pool:
        out["importance"] = _importance(shuffled_pool, len(folds), by_family[model_type], actual, quarters, train_mean, cols)
    if run_ablation:
        out["ablation_vs_trend"] = {
            "note": "Gradient-boosted trees on the same folds, with and without the vs-own-trend features.",
            "with": evaluate_predictions(np.concatenate(ablation["with"]), actual, quarters, train_mean),
            "without": evaluate_predictions(np.concatenate(ablation["without"]), actual, quarters, train_mean),
        }
    return out


def _importance(shuffled_pool, n_folds, base, actual, quarters, train_mean, cols) -> dict:
    """Per feature group: how much out-of-sample IC and accuracy fall when the
    group's values are shuffled. Bigger drop = the model leans on it more;
    near zero or negative = it adds nothing the model uses."""
    out = {}
    for group, parts in shuffled_pool.items():
        if len(parts) != n_folds:
            continue
        m = evaluate_predictions(np.concatenate(parts), actual, quarters, train_mean)
        out[group] = {
            "ic_drop": round(base["ic"] - m["ic"], 4) if base["ic"] is not None and m["ic"] is not None else None,
            "accuracy_drop": round(base["directional_accuracy"] - m["directional_accuracy"], 4),
            "n_features": sum(c in cols for c in FEATURE_GROUPS[group]),
        }
    return out


def _fit_final(hdf, cols, horizon, model_type, epochs, batch_size, out_dir, status) -> dict:
    """Retrains on every row for this horizon and writes the artifacts."""
    status(f"{horizon}: training final model on {len(hdf)} rows")
    pre = RobustPreprocessor(cols, RANK_FEATURES, config.MIN_PEERS_PER_QUARTER)
    X = pre.fit_transform(hdf[cols], hdf["period_quarter"])
    target = HorizonTarget()
    y = target.fit_transform(hdf[f"target_{horizon}"].to_numpy(dtype="float64"))

    entry = {
        "features": cols,
        "input_columns": pre.output_columns,
        "target": target.to_dict(),
        "preprocessor_file": f"{horizon}_preprocessor.joblib",
        "nn_files": [],
        "gbm_file": None,
        "n_train": len(hdf),
        "train_start": hdf["earnings_date"].iloc[0],
        "train_end": hdf["earnings_date"].iloc[-1],
    }
    joblib.dump(pre, os.path.join(out_dir, entry["preprocessor_file"]))
    families = _families(model_type)
    if "gbm" in families:
        entry["gbm_file"] = f"{horizon}_gbm.joblib"
        joblib.dump(train_gbm(X, y), os.path.join(out_dir, entry["gbm_file"]))
    if "nn" in families:
        for seed in range(config.ENSEMBLE_SEEDS):
            status(f"{horizon}: training network {seed + 1}/{config.ENSEMBLE_SEEDS} on {len(hdf)} rows")
            net = train_network(X, y, seed, epochs, batch_size)
            fname = f"{horizon}_nn_{seed}.keras"
            net.save(os.path.join(out_dir, fname))
            entry["nn_files"].append(fname)
            tf.keras.backend.clear_session()
    return entry


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:50] or "model"


# --------------------------------------------------------------------- run
def _run(run_id, symbols, epochs, batch_size, name, model_type):
    db = None
    db_run_id = None
    try:
        db = SupabaseLogger()
    except Exception as exc:
        _set_status(run_id, phase="failed", message=f"Supabase config error: {exc}", done=True, error=True)
        return

    now = datetime.now(timezone.utc)
    model_name = (name or "").strip() or f"Model {now.strftime('%Y-%m-%d %H:%M')} UTC"
    local_dir = os.path.join(config.LOCAL_MODEL_DIR, f"{_slug(model_name)}_{now.strftime('%Y%m%d_%H%M%S')}")

    try:
        run_config = {
            "name": model_name, "model_type": model_type, "version": MODEL_VERSION,
            "epochs": epochs, "batch_size": batch_size, "benchmark": config.BENCHMARK_SYMBOL,
            "nn_learning_rate": round(network_learning_rate(batch_size), 6),
        }
        db_run_id = db.create_training_run(symbols, run_config)
        _set_status(run_id, db_run_id=db_run_id, model_name=model_name)

        # ---------------------------------------------------------- 1. fetch + assemble
        def fetch_progress(msg, frac):
            _set_status(run_id, phase="fetching", message=msg, progress=0.02 + 0.33 * frac)

        _set_status(run_id, phase="fetching", message="Fetching filings from SEC EDGAR and prices...", progress=0.02)
        result = build_training_dataset(symbols, SECClient(), PriceClient(), progress_cb=fetch_progress)
        error_count = db.log_events(db_run_id, result.events)

        df = result.dataframe
        if df.empty or len(df) < config.MIN_TRAINING_ROWS:
            msg = (
                f"Only {len(df)} usable earnings events were assembled "
                f"(need at least {config.MIN_TRAINING_ROWS}). Try more tickers, or tickers "
                f"with a longer public trading history."
            )
            db.finish_training_run(db_run_id, "failed", metrics={"reason": msg}, error_count=error_count)
            _set_status(run_id, phase="failed", message=msg, done=True, error=True)
            return

        _set_status(run_id, phase="saving_data", message=f"Saving {len(df)} training rows to Supabase...", progress=0.37)
        db.save_training_data(db_run_id, df, result.feature_columns, HORIZON_LABELS)

        # ------------------------------------------------ 2. per-horizon evaluate + fit
        os.makedirs(local_dir, exist_ok=True)
        df = df.sort_values(["earnings_date", "symbol"]).reset_index(drop=True)
        metrics, horizons = {}, {}
        for hi, horizon in enumerate(HORIZON_LABELS):
            base = 0.4 + 0.5 * hi / len(HORIZON_LABELS)

            def status(msg, base=base):
                _set_status(run_id, phase="training", message=msg, progress=base)

            hdf = df[df[f"target_{horizon}"].notna()].reset_index(drop=True)
            cols = [c for c in horizon_features(horizon) if c in result.feature_columns]
            if len(hdf) < config.MIN_TRAINING_ROWS:
                metrics[horizon] = {"evaluation": "none", "reason": f"Only {len(hdf)} rows have a {horizon} label."}
                continue
            metrics[horizon] = _evaluate_horizon(hdf, cols, horizon, model_type, epochs, batch_size, status)
            horizons[horizon] = _fit_final(hdf, cols, horizon, model_type, epochs, batch_size, local_dir, status)

        if not horizons:
            raise RuntimeError("No horizon had enough labelled rows to train on.")

        importance = {h: m.pop("importance") for h, m in metrics.items() if "importance" in m}
        metrics["_diagnostics"] = build_diagnostics(
            training_profile(df, symbols, run_config, {h: e["features"] for h, e in horizons.items()}),
            importance, "shuffle importance on the walk-forward folds at training time",
        )

        # -------------------------------------------------------------------- 3. save
        _set_status(run_id, phase="saving_model", message="Saving model...", progress=0.92)
        manifest = {
            "version": MODEL_VERSION,
            "name": model_name,
            "model_type": model_type,
            "benchmark": config.BENCHMARK_SYMBOL,
            "hold_band_std": config.HOLD_BAND_STD,
            "horizons": horizons,
        }
        with open(os.path.join(local_dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1)

        model_id = db.save_model(db_run_id, model_name, local_dir, {
            "feature_columns": {h: e["features"] for h, e in horizons.items()},
            "target_horizons": list(horizons),
            "target_scale": {"version": MODEL_VERSION, "horizons": {h: e["target"] for h, e in horizons.items()}},
            "metrics": metrics,
            "symbols": [s.strip().upper() for s in symbols],
        }, storage_name=os.path.basename(local_dir))

        db.finish_training_run(db_run_id, "completed", metrics=metrics, error_count=error_count)
        _set_status(
            run_id, phase="completed", message=f"Training complete: {model_name}", progress=1.0,
            done=True, model_id=model_id, metrics=metrics,
            rows_used=len(df), warnings=error_count,
        )

    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Training run %s failed", run_id)
        if db and db_run_id:
            try:
                db.log_error(db_run_id, "*", "train", str(exc), tb)
                db.finish_training_run(db_run_id, "failed", metrics={"error": str(exc)})
            except Exception:
                logger.exception("Also failed to log the failure to Supabase")
        _set_status(run_id, phase="failed", message=str(exc), done=True, error=True)
    finally:
        shutil.rmtree(local_dir, ignore_errors=True)


def get_status(run_id: str) -> dict | None:
    with _lock:
        return RUNS.get(run_id)
