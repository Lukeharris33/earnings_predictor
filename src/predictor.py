"""
Inference path: load a saved model (cached locally after first download from
Supabase Storage), pull a ticker's most recent quarterly report from SEC EDGAR,
predict the % move at each horizon, and turn that into buy/sell/hold calls.

Two artifact formats:
  v2 (has manifest.json): one model per horizon -- a preprocessor plus a
      network ensemble and/or gradient-boosted trees -- predicting the move
      in excess of the benchmark.
  v1 (model.keras + preprocessor.joblib): the original single four-output
      network predicting raw moves. Still loadable so old models keep working.
"""
from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import config
from .data_pipeline import SECTOR_FEATURES, fetch_latest_report_features
from .model import decide
from .preprocessing import FeaturePreprocessor, HorizonTarget, TargetScaler
from .price_client import PriceClient
from .sec_client import SECClient
from .supabase_client import SupabaseLogger


@dataclass
class _Horizon:
    features: list[str]
    preprocessor: object
    target: HorizonTarget
    networks: list = field(default_factory=list)
    gbm: object = None


@dataclass
class _LoadedModel:
    version: int
    model_type: str = "nn"
    benchmark: str | None = None
    hold_band_std: float = config.HOLD_BAND_STD
    horizons: dict[str, _Horizon] = field(default_factory=dict)
    # v1 only
    keras_model: object = None
    preprocessor: FeaturePreprocessor | None = None
    target_scaler: TargetScaler | None = None


_MODEL_CACHE: dict[str, _LoadedModel] = {}
_cache_lock = threading.Lock()


def _load_model(model_id: str, db: SupabaseLogger) -> _LoadedModel:
    with _cache_lock:
        if model_id in _MODEL_CACHE:
            return _MODEL_CACHE[model_id]

        local_dir = os.path.join(config.LOCAL_MODEL_DIR, "cache", model_id)
        manifest_path = os.path.join(local_dir, "manifest.json")
        if not (os.path.exists(manifest_path) or os.path.exists(os.path.join(local_dir, "model.keras"))):
            db.download_model(model_id, local_dir)

        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
            loaded = _LoadedModel(
                version=manifest["version"],
                model_type=manifest["model_type"],
                benchmark=manifest.get("benchmark"),
                hold_band_std=manifest.get("hold_band_std", config.HOLD_BAND_STD),
            )
            for h, entry in manifest["horizons"].items():
                loaded.horizons[h] = _Horizon(
                    features=entry["features"],
                    preprocessor=joblib.load(os.path.join(local_dir, entry["preprocessor_file"])),
                    target=HorizonTarget.from_dict(entry["target"]),
                    networks=[tf.keras.models.load_model(os.path.join(local_dir, f)) for f in entry["nn_files"]],
                    gbm=joblib.load(os.path.join(local_dir, entry["gbm_file"])) if entry.get("gbm_file") else None,
                )
        else:
            row = db.get_model(model_id)
            loaded = _LoadedModel(
                version=1,
                keras_model=tf.keras.models.load_model(os.path.join(local_dir, "model.keras")),
                preprocessor=FeaturePreprocessor.load(os.path.join(local_dir, "preprocessor.joblib")),
                target_scaler=TargetScaler.from_dict(row["target_scale"]),
            )

        _MODEL_CACHE[model_id] = loaded
        return loaded


def predict_for_symbol(
    model_id: str,
    symbol: str,
    name: str | None = None,
    batch_id: str | None = None,
    db: SupabaseLogger | None = None,
    sec: SECClient | None = None,
    price_client: PriceClient | None = None,
) -> dict:
    db = db or SupabaseLogger()
    model = _load_model(model_id, db)

    earnings_date, features, note, source = fetch_latest_report_features(
        symbol, sec or SECClient(), price_client or PriceClient()
    )
    if not features:
        raise ValueError(f"Could not find a quarterly report with XBRL financials for {symbol}")

    if model.version >= 2:
        predicted_pct, decisions, inputs = _predict_v2(model, features, source["period_quarter"])
    else:
        predicted_pct, decisions, inputs = _predict_v1(model, features)

    for h, window in source["horizon_windows"].items():
        if h in decisions:
            decisions[h].update({
                "window_closed": window["closed"],
                "window_end": window["window_end"],
                "actual_pct_change": window["actual_excess_pct"],
            })

    name = (name or "").strip() or None
    details = {"source": source, "inputs": inputs, "note": note,
               "model_version": model.version, "benchmark": model.benchmark}
    prediction_id = db.save_prediction(
        model_id, symbol.upper(), earnings_date, predicted_pct, decisions,
        name=name, batch_id=batch_id, details=details,
    )

    return {
        "id": prediction_id,
        "name": name,
        "symbol": symbol.upper(),
        "earnings_date": earnings_date,
        "decisions": decisions,
        "note": note,
        "source": source,
        "inputs": inputs,
        "model_version": model.version,
        "benchmark": model.benchmark,
    }


def _predict_v2(model: _LoadedModel, features: dict, quarter: str):
    row = pd.DataFrame([features])
    predicted, stds, scaled_rows = {}, {}, {}
    for h, hz in model.horizons.items():
        X = hz.preprocessor.transform(row, [quarter])
        scaled_rows[h] = X[0]
        parts = []
        if hz.networks:
            X_t = tf.convert_to_tensor(X, dtype=tf.float32)
            parts.append(float(np.mean([net(X_t, training=False).numpy()[0, 0] for net in hz.networks])))
        if hz.gbm is not None:
            parts.append(float(hz.gbm.predict(X)[0]))
        predicted[h] = float(hz.target.inverse(np.mean(parts)))
        stds[h] = hz.target.std

    decisions = decide(predicted, stds, model.hold_band_std)
    return predicted, decisions, _inputs_v2(model, features, quarter, scaled_rows)


def _inputs_v2(model: _LoadedModel, features: dict, quarter: str, scaled_rows: dict) -> list[dict]:
    """One row per input, explained by the horizon that uses the most inputs
    (so every feature appears), tagged with which horizons use it."""
    ref_h = max(model.horizons, key=lambda h: (len(model.horizons[h].features), h == "1m"))
    ref = model.horizons[ref_h]
    rows = ref.preprocessor.explain(features, quarter, scaled_rows[ref_h])
    out = []
    for r in rows:
        if r["feature"] in SECTOR_FEATURES:
            continue  # shown as the company's sector instead
        r["used_by"] = [h for h, hz in model.horizons.items() if r["feature"] in hz.features]
        r["stats_from"] = ref_h
        out.append(r)
    return out


def _predict_v1(model: _LoadedModel, features: dict):
    pre = model.preprocessor
    known = sum(1 for c in pre.feature_columns if c in features)
    if known < len(pre.feature_columns) / 2:
        # Models trained on the old FMP feature set would silently predict
        # from all-imputed (median) inputs.
        raise ValueError(
            "This model was trained on a different feature set (likely the old FMP data). "
            "Train a new model and use that instead."
        )
    X = pre.transform(pd.DataFrame([features]))
    pred_scaled = model.keras_model.predict(X, verbose=0)[0]
    predicted = model.target_scaler.inverse_transform_row(pred_scaled)
    decisions = decide(predicted, model.target_scaler.stds)

    medians = pre.imputer.statistics_
    inputs = []
    for i, col in enumerate(pre.feature_columns):
        raw = features.get(col)
        missing = raw is None or not math.isfinite(raw)
        inputs.append({
            "feature": col,
            "value": None if missing else float(raw),
            "imputed": missing,
            "clipped": False,
            "training_median": float(medians[i]),
            "z_score": float(X[0][i]),
            "peer_percentile": None,
            "used_by": list(predicted),
        })
    return predicted, decisions, inputs
