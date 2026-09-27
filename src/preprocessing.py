"""
Handles the "normalise null values" requirement:
  1. Median-impute missing numeric fields (earnings reports very often omit
     line items like R&D expense for companies that don't report one).
  2. Standard-scale everything so no single large-magnitude field like
     "revenue" dominates the loss versus a ratio field like "grossProfitRatio".

The fitted imputer/scaler are saved alongside every model so inference uses
the exact same transformation the model was trained on.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class FeaturePreprocessor:
    def __init__(self, feature_columns: list[str]):
        self.feature_columns = feature_columns
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.feature_columns).astype("float64")
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        self._fitted = True
        return X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit before transform().")
        X = df.reindex(columns=self.feature_columns).astype("float64")
        X_imputed = self.imputer.transform(X)
        return self.scaler.transform(X_imputed)

    def save(self, path: str):
        joblib.dump(
            {
                "feature_columns": self.feature_columns,
                "imputer": self.imputer,
                "scaler": self.scaler,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "FeaturePreprocessor":
        payload = joblib.load(path)
        obj = cls(payload["feature_columns"])
        obj.imputer = payload["imputer"]
        obj.scaler = payload["scaler"]
        obj._fitted = True
        return obj


class TargetScaler:
    """Standardizes each horizon's % change target independently, and keeps
    the raw standard deviation around so the app can turn a predicted
    magnitude into a 0-1 confidence weight at inference time."""

    def __init__(self, horizon_labels: list[str]):
        self.horizon_labels = horizon_labels
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}

    def fit_transform(self, targets_df: pd.DataFrame) -> np.ndarray:
        cols = []
        for label in self.horizon_labels:
            col = targets_df[f"target_{label}"].astype("float64")
            mean, std = float(col.mean()), float(col.std() or 1.0)
            std = std if std > 1e-6 else 1.0
            self.means[label] = mean
            self.stds[label] = std
            cols.append((col - mean) / std)
        return np.column_stack(cols)

    def transform(self, targets_df: pd.DataFrame) -> np.ndarray:
        cols = []
        for label in self.horizon_labels:
            col = targets_df[f"target_{label}"].astype("float64")
            cols.append((col - self.means[label]) / self.stds[label])
        return np.column_stack(cols)

    def inverse_transform_row(self, scaled_row) -> dict[str, float]:
        return {
            label: float(scaled_row[i] * self.stds[label] + self.means[label])
            for i, label in enumerate(self.horizon_labels)
        }

    def to_dict(self) -> dict:
        return {"horizon_labels": self.horizon_labels, "means": self.means, "stds": self.stds}

    @classmethod
    def from_dict(cls, payload: dict) -> "TargetScaler":
        obj = cls(payload["horizon_labels"])
        obj.means = payload["means"]
        obj.stds = payload["stds"]
        return obj


# --------------------------------------------------------------------- v2
def _peer_percentile(ref: np.ndarray, value: float) -> float:
    """Where `value` falls among a quarter's reported values, 0-1 (ties split)."""
    if not np.isfinite(value) or len(ref) == 0:
        return np.nan
    lo = np.searchsorted(ref, value, side="left")
    hi = np.searchsorted(ref, value, side="right")
    return float((lo + hi) / 2 / len(ref))


class RobustPreprocessor:
    """Per-horizon feature transform (used by v2 models):

    1. Clip each feature to its 1st-99th percentile in the training data, so
       an extreme filing (e.g. revenue near zero -> -500% margins) can't
       dominate the inputs.
    2. Median-impute missing values, and add a 0/1 "was missing" column for
       every feature that was ever missing in training -- not reporting
       something is information too.
    3. For RANK_FEATURES, add the value's percentile among all companies that
       reported the same calendar quarter (economy-wide swings cancel out).
       Inference ranks against the stored quarter, or the latest earlier
       quarter the training data has.
    4. Standard-scale everything.
    """

    def __init__(self, feature_columns: list[str], rank_columns: list[str], min_peers: int = 8):
        self.feature_columns = list(feature_columns)
        self.rank_columns = [c for c in rank_columns if c in self.feature_columns]
        self.min_peers = min_peers
        self.lower: np.ndarray | None = None
        self.upper: np.ndarray | None = None
        self.medians: np.ndarray | None = None
        self.missing_columns: list[str] = []
        self.peer_refs: dict[str, dict[str, np.ndarray]] = {}  # quarter -> feature -> sorted values
        self.scaler = StandardScaler()

    @property
    def output_columns(self) -> list[str]:
        return (
            self.feature_columns
            + [f"{c}__missing" for c in self.missing_columns]
            + [f"{c}__peer_pct" for c in self.rank_columns]
        )

    def fit_transform(self, df: pd.DataFrame, quarters: pd.Series) -> np.ndarray:
        X = df.reindex(columns=self.feature_columns).to_numpy(dtype="float64")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            self.lower = np.nan_to_num(np.nanpercentile(X, 1, axis=0), nan=0.0)
            self.upper = np.nan_to_num(np.nanpercentile(X, 99, axis=0), nan=0.0)
            clipped = np.clip(X, self.lower, self.upper)
            self.medians = np.nan_to_num(np.nanmedian(clipped, axis=0), nan=0.0)
        self.missing_columns = [c for i, c in enumerate(self.feature_columns) if np.isnan(X[:, i]).any()]

        self.peer_refs = {}
        quarters = pd.Series(quarters).astype(str).to_numpy()
        for q in np.unique(quarters):
            mask = quarters == q
            if mask.sum() < self.min_peers:
                continue
            refs = {}
            for c in self.rank_columns:
                vals = X[mask, self.feature_columns.index(c)]
                refs[c] = np.sort(vals[np.isfinite(vals)]).astype("float32")
            self.peer_refs[q] = refs
        return self.scaler.fit_transform(self._assemble(X, quarters))

    def transform(self, df: pd.DataFrame, quarters) -> np.ndarray:
        X = df.reindex(columns=self.feature_columns).to_numpy(dtype="float64")
        quarters = pd.Series(quarters).astype(str).to_numpy()
        return self.scaler.transform(self._assemble(X, quarters))

    def _reference_quarter(self, quarter: str) -> str | None:
        if quarter in self.peer_refs:
            return quarter
        earlier = [q for q in self.peer_refs if q <= quarter]
        if earlier:
            return max(earlier)
        return min(self.peer_refs) if self.peer_refs else None

    def peer_percentiles(self, X: np.ndarray, quarters: np.ndarray) -> np.ndarray:
        out = np.full((len(X), len(self.rank_columns)), np.nan)
        for r, q in enumerate(quarters):
            ref_q = self._reference_quarter(q)
            if ref_q is None:
                continue
            for j, c in enumerate(self.rank_columns):
                out[r, j] = _peer_percentile(self.peer_refs[ref_q][c], X[r, self.feature_columns.index(c)])
        return out

    def _assemble(self, X: np.ndarray, quarters: np.ndarray) -> np.ndarray:
        clipped = np.clip(X, self.lower, self.upper)
        missing_idx = [self.feature_columns.index(c) for c in self.missing_columns]
        indicators = np.isnan(X[:, missing_idx]).astype("float64")
        imputed = np.where(np.isnan(clipped), self.medians, clipped)
        peers = np.nan_to_num(self.peer_percentiles(X, quarters), nan=0.5)
        return np.hstack([imputed, indicators, peers])

    def explain(self, features: dict, quarter: str, scaled_row: np.ndarray) -> list[dict]:
        """Per input feature: raw value, whether it was missing (imputed) or
        clipped, the training median, its z-score, and its peer percentile."""
        X = np.array([[_as_float(features.get(c)) for c in self.feature_columns]])
        peers = self.peer_percentiles(X, np.array([quarter]))[0]
        out = []
        for i, c in enumerate(self.feature_columns):
            raw = X[0, i]
            missing = not np.isfinite(raw)
            out.append({
                "feature": c,
                "value": None if missing else float(raw),
                "imputed": missing,
                "clipped": (not missing) and not (self.lower[i] <= raw <= self.upper[i]),
                "training_median": float(self.medians[i]),
                "z_score": float(scaled_row[i]),
                "peer_percentile": (
                    None if c not in self.rank_columns or np.isnan(peers[self.rank_columns.index(c)])
                    else float(peers[self.rank_columns.index(c)])
                ),
            })
        return out


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


class HorizonTarget:
    """One horizon's target transform: winsorize to the training 1st-99th
    percentile (so a few +/-80% moves don't dominate), then standardize.
    `std` is kept to size confidence weights and the HOLD band."""

    def __init__(self):
        self.lower = self.upper = self.mean = 0.0
        self.std = 1.0

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        self.lower, self.upper = (float(v) for v in np.percentile(y, [1, 99]))
        clipped = np.clip(y, self.lower, self.upper)
        self.mean = float(clipped.mean())
        std = float(clipped.std())
        self.std = std if std > 1e-6 else 1.0
        return (clipped - self.mean) / self.std

    def inverse(self, scaled):
        return np.asarray(scaled) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"lower": self.lower, "upper": self.upper, "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "HorizonTarget":
        obj = cls()
        obj.lower, obj.upper, obj.mean, obj.std = d["lower"], d["upper"], d["mean"], d["std"]
        return obj
