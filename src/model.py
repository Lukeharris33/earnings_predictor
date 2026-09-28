"""
Per-horizon regressors that predict the *signed magnitude* of the forward %
move in excess of the benchmark, not a binary up/down label. Buy/sell/hold
decisions are derived from that magnitude afterwards, and the confidence
"weight" attached to each decision scales with how large the predicted move
is -- so a predicted +0.3% move and a predicted +9% move are not treated the
same.

Two model families, usable alone or averaged ("blend"):
  - a small, L2-regularized network, trained with several random seeds and
    averaged (small tabular datasets overfit big networks easily);
  - gradient-boosted trees (scikit-learn's HistGradientBoostingRegressor,
    the same algorithm as LightGBM), which usually do well on this kind of
    data and are robust to outliers.

Evaluation reports each against naive baselines so a number like "56%
directional accuracy" can be judged: "always up" accuracy, the zero-forecast
MAE, rank correlation (IC) and the top-vs-bottom-quintile return spread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingRegressor
from tensorflow.keras import callbacks, layers, models, regularizers

from .config import config

HORIZON_LABELS = list(config.HORIZONS.keys())


# ------------------------------------------------------------------ network
BASE_LEARNING_RATE = 1e-3  # at the default batch size


def network_learning_rate(batch_size: int) -> float:
    """Adam's learning rate scaled with the square root of the batch size
    (bigger batches give less noisy gradients, so can take bigger steps),
    anchored so the default batch size keeps BASE_LEARNING_RATE."""
    return BASE_LEARNING_RATE * (batch_size / config.DEFAULT_BATCH_SIZE) ** 0.5


def build_network(input_dim: int, seed: int = 0, learning_rate: float = BASE_LEARNING_RATE) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    l2 = regularizers.l2(1e-4)
    inputs = layers.Input(shape=(input_dim,), name="earnings_features")
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="linear", name="excess_pct_change")(x)

    model = models.Model(inputs, outputs, name="earnings_move_predictor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        # Huber is robust to the fat-tailed outliers common in earnings-day
        # price moves (a handful of +/-30% gaps shouldn't dominate the loss).
        loss=tf.keras.losses.Huber(delta=1.0),
    )
    return model


def train_network(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    epochs: int = config.DEFAULT_EPOCHS,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
) -> tf.keras.Model:
    """Early-stops on the chronologically last slice of the training rows
    (rows arrive sorted by date), so stopping never peeks at the future."""
    split = int(len(X) * (1 - config.VALIDATION_FRACTION))
    split = min(max(split, 1), len(X) - 1)
    model = build_network(X.shape[1], seed, network_learning_rate(batch_size))
    model.fit(
        X[:split], y[:split],
        validation_data=(X[split:], y[split:]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
        ],
        verbose=0,
    )
    return model


def predict_network(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    return model.predict(X, verbose=0, batch_size=1024)[:, 0]


# ----------------------------------------------------------------- boosting
def train_gbm(X: np.ndarray, y: np.ndarray, seed: int = 0) -> HistGradientBoostingRegressor:
    gbm = HistGradientBoostingRegressor(
        loss="squared_error",  # targets are already winsorized
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    gbm.fit(X, y)
    return gbm


# --------------------------------------------------------------- evaluation
def evaluate_predictions(pred: np.ndarray, actual: np.ndarray, quarters: np.ndarray, train_mean: float) -> dict:
    """Metrics in real % points for one horizon on out-of-sample rows."""
    pred, actual = np.asarray(pred, dtype="float64"), np.asarray(actual, dtype="float64")
    n = len(actual)
    frac_up = float(np.mean(actual > 0))
    df = pd.DataFrame({"pred": pred, "actual": actual, "q": quarters})

    ic_by_quarter = [
        g["pred"].corr(g["actual"], method="spearman")
        for _, g in df.groupby("q") if len(g) >= 5 and g["pred"].nunique() > 1
    ]
    ic_by_quarter = [v for v in ic_by_quarter if np.isfinite(v)]

    spread = None
    if n >= 20 and df["pred"].nunique() > 5:
        quintile = pd.qcut(df["pred"].rank(method="first"), 5, labels=False)
        spread = float(df.loc[quintile == 4, "actual"].mean() - df.loc[quintile == 0, "actual"].mean())

    pooled_ic = df["pred"].corr(df["actual"], method="spearman") if df["pred"].nunique() > 1 else np.nan
    return {
        "n": n,
        "mae_pct_points": round(float(np.mean(np.abs(pred - actual))), 3),
        "directional_accuracy": round(float(np.mean(np.sign(pred) == np.sign(actual))), 4),
        "ic": round(float(pooled_ic), 4) if np.isfinite(pooled_ic) else None,
        "ic_by_quarter_mean": round(float(np.mean(ic_by_quarter)), 4) if ic_by_quarter else None,
        "top_minus_bottom_quintile": round(spread, 3) if spread is not None else None,
        # Naive baselines on the same rows.
        "baseline_always_up_accuracy": round(frac_up, 4),
        "baseline_majority_accuracy": round(max(frac_up, 1 - frac_up), 4),
        "baseline_zero_mae": round(float(np.mean(np.abs(actual))), 3),
        "baseline_train_mean_mae": round(float(np.mean(np.abs(actual - train_mean))), 3),
    }


# ---------------------------------------------------------------- decisions
def decide(
    predicted_pct_by_horizon: dict[str, float],
    std_by_horizon: dict[str, float],
    hold_band_std: float = config.HOLD_BAND_STD,
) -> dict:
    """
    Turns predicted % moves into decisions with a 0-1 confidence weight.
    Moves within `hold_band_std` standard deviations of zero are HOLD --
    too small to act on. The weight is the predicted magnitude relative to
    that horizon's typical move (2 standard deviations = full confidence),
    so a 1-day call and a 1-year call are judged on the same relative scale.
    """
    decisions = {}
    for label, pct in predicted_pct_by_horizon.items():
        typical_move = max(float(std_by_horizon.get(label, 1.0)), 1e-6)
        band = hold_band_std * typical_move
        if pct > band:
            action = "BUY"
        elif pct < -band:
            action = "SELL"
        else:
            action = "HOLD"
        weight = float(np.clip(abs(pct) / (2 * typical_move), 0.0, 1.0))
        decisions[label] = {
            "predicted_pct_change": round(float(pct), 3),
            "action": action,
            "confidence_weight": round(weight, 3),
            "hold_band_pct": round(band, 3),
        }
    return decisions
