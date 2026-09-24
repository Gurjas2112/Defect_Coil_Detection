"""Shared preprocessing and feature engineering for training and inference."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

SEED = 42
FEATURE_COLS = [f"X{i}" for i in range(1, 50)]


def engineer_features(df, fit_iso=None):
    df = df.copy()
    raw_cols = [c for c in df.columns if c.startswith('X')]
    vals = df[raw_cols].values

    # ── Aggregate statistics ──
    df['agg_mean']    = vals.mean(axis=1)
    df['agg_std']     = vals.std(axis=1)
    df['agg_max']     = vals.max(axis=1)
    df['agg_min']     = vals.min(axis=1)
    df['agg_range']   = df['agg_max'] - df['agg_min']
    df['agg_iqr']     = np.percentile(vals, 75, axis=1) - np.percentile(vals, 25, axis=1)
    df['agg_median']  = np.median(vals, axis=1)
    df['agg_skew']    = pd.DataFrame(vals).skew(axis=1).values
    df['agg_kurt']    = pd.DataFrame(vals).kurtosis(axis=1).values

    # ── Stage-wise means (process has multiple stages) ──
    # X1-X10: Stage 1 | X11-X20: Stage 2 | X21-X30: Stage 3
    # X31-X40: Stage 4 | X41-X49: Stage 5
    stages = [(1,10), (11,20), (21,30), (31,40), (41,49)]
    for i, (s, e) in enumerate(stages, 1):
        scols = [f'X{j}' for j in range(s, e+1) if f'X{j}' in df.columns]
        df[f'stage{i}_mean'] = df[scols].mean(axis=1)
        df[f'stage{i}_std']  = df[scols].std(axis=1)
        df[f'stage{i}_max']  = df[scols].max(axis=1)

    # ── Cross-stage interaction features ──
    df['s1_s2_diff']  = df['stage1_mean'] - df['stage2_mean']
    df['s2_s3_diff']  = df['stage2_mean'] - df['stage3_mean']
    df['s3_s4_diff']  = df['stage3_mean'] - df['stage4_mean']
    df['s4_s5_diff']  = df['stage4_mean'] - df['stage5_mean']
    df['s1_s5_ratio'] = df['stage1_mean'] / (df['stage5_mean'] + 1e-8)

    # ── Spike / crash detection ──
    df['spike_count'] = (vals > vals.mean() + 2*vals.std()).sum(axis=1)
    df['crash_count'] = (vals < vals.mean() - 2*vals.std()).sum(axis=1)
    df['spike_ratio'] = df['spike_count'] / (df['crash_count'] + 1)

    return df


def fit_preprocessor(X, y):
    """Fit imputer, clip bounds and Isolation Forest on the training data."""
    X = X[FEATURE_COLS]
    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_COLS)

    bounds = {c: (X_imp[c].quantile(0.01), X_imp[c].quantile(0.99)) for c in FEATURE_COLS}
    for c, (lo, hi) in bounds.items():
        X_imp[c] = X_imp[c].clip(lo, hi)

    X_fe = engineer_features(X_imp)
    iso = IsolationForest(n_estimators=400, contamination=0.05, max_samples='auto', random_state=SEED)
    iso.fit(X_fe[np.asarray(y) == 0].values)

    return {'imputer': imputer, 'clip_bounds': bounds, 'iso': iso}


def transform(X, prep):
    """Apply the fitted preprocessing and return the full model feature matrix."""
    X = X[FEATURE_COLS].astype(float)
    X_imp = pd.DataFrame(prep['imputer'].transform(X), columns=FEATURE_COLS)
    for c, (lo, hi) in prep['clip_bounds'].items():
        X_imp[c] = X_imp[c].clip(lo, hi)

    X_fe = engineer_features(X_imp)
    X_fe['iso_anomaly'] = -prep['iso'].score_samples(X_fe.values)
    return X_fe


def validate_input(df):
    """Return a list of human-readable problems with an uploaded dataframe."""
    problems = []
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        shown = ', '.join(missing[:10]) + (' ...' if len(missing) > 10 else '')
        problems.append(f"Missing {len(missing)} required column(s): {shown}")

    present = [c for c in FEATURE_COLS if c in df.columns]
    bad = [c for c in present
           if pd.to_numeric(df[c], errors='coerce').isna().sum() > df[c].isna().sum()]
    if bad:
        problems.append(f"Non-numeric values found in: {', '.join(bad[:10])}")

    if len(df) == 0:
        problems.append("The file contains no rows.")
    return problems
