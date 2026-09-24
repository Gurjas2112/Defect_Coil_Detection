"""Train the defect-detection ensemble and save it for the Streamlit app.

Usage:
    python train.py   # writes models/defect_model.joblib
"""

import os
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from pipeline import FEATURE_COLS, SEED, fit_preprocessor, transform

warnings.filterwarnings('ignore')

N_FOLDS   = 5
CLASS_WT  = 40
LR        = 0.015
N_EST     = 700
MAX_DEPTH = 5
MODEL_PATH = os.path.join('models', 'defect_model.joblib')


def make_models(seed):
    lgb = LGBMClassifier(
        n_estimators=N_EST, class_weight={0: 1, 1: CLASS_WT},
        learning_rate=LR, max_depth=MAX_DEPTH,
        min_child_samples=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=seed, verbose=-1,
    )
    xgb = XGBClassifier(
        n_estimators=N_EST, scale_pos_weight=CLASS_WT,
        learning_rate=LR, max_depth=MAX_DEPTH,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, eval_metric='logloss', verbosity=0,
    )
    cat = CatBoostClassifier(
        iterations=N_EST, class_weights={0: 1, 1: CLASS_WT},
        learning_rate=LR, depth=MAX_DEPTH, l2_leaf_reg=3.0,
        random_seed=seed, verbose=0, allow_writing_files=False,
    )
    return [lgb, xgb, cat]


def ensemble_proba(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def pick_threshold(y, probs):
    """Highest-precision threshold that keeps recall at 1.0 on out-of-fold data."""
    best_t, best_prec = None, -1.0
    for t in np.arange(0.001, 0.60, 0.0005):
        preds = (probs >= t).astype(int)
        if recall_score(y, preds, zero_division=0) == 1.0:
            prec = precision_score(y, preds, zero_division=0)
            if prec > best_prec:
                best_t, best_prec = t, prec
    if best_t is None:
        best_t = float(probs[y == 1].min())
    return float(best_t)


def main():
    train = pd.read_csv('dataset/train.csv')
    X_raw = train[FEATURE_COLS]
    y = train['Y'].values

    print(f"Loaded train.csv: {train.shape[0]} coils, {int(y.sum())} defects ({y.mean()*100:.2f}%)")

    prep = fit_preprocessor(X_raw, y)
    X_fe = transform(X_raw, prep)
    X = X_fe.values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        models = make_models(SEED + fold)
        for m in models:
            m.fit(X[tr], y[tr])
        oof[va] = ensemble_proba(models, X[va])
        print(f"  Fold {fold+1}/{N_FOLDS} ROC-AUC: {roc_auc_score(y[va], oof[va]):.4f}")

    threshold = pick_threshold(y, oof)
    preds = (oof >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    metrics = {
        'threshold': threshold,
        'recall': float(recall_score(y, preds)),
        'precision': float(precision_score(y, preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, oof)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }
    print(f"\nOOF @ threshold {threshold:.4f}: recall={metrics['recall']:.4f} "
          f"precision={metrics['precision']:.4f} ROC-AUC={metrics['roc_auc']:.4f}")
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")

    print("\nRefitting ensemble on the full training set...")
    final_models = make_models(SEED)
    for m in final_models:
        m.fit(X, y)

    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'prep': prep,
        'models': final_models,
        'threshold': threshold,
        'oof_metrics': metrics,
        'oof_probs': oof,
        'oof_labels': y,
        'feature_names': X_fe.columns.tolist(),
        'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }, MODEL_PATH, compress=3)
    print(f"Saved {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1e6:.2f} MB)")


if __name__ == '__main__':
    main()
