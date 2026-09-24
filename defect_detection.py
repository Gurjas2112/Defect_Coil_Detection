

# Library Imports & Configuration

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (recall_score, precision_score, f1_score,
                              roc_auc_score, classification_report,
                              confusion_matrix, precision_recall_curve)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               IsolationForest, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek

# ── Global Config ──
SEED       = 42
N_FOLDS    = 5
CLASS_WT   = 40          # positive class weight multiplier
LR         = 0.015       # learning rate
N_EST      = 700         # estimators per model
MAX_DEPTH  = 5

np.random.seed(SEED)
pd.set_option('display.max_columns', 60)
print("✅ All libraries loaded successfully")
print(f"   Config → SEED={SEED} | Folds={N_FOLDS} | ClassWeight=1:{CLASS_WT} | LR={LR}")


#Data Loading & Structural Audit

# %%
train = pd.read_csv('dataset/train.csv')
test  = pd.read_csv('dataset/test.csv')
sample = pd.read_csv('dataset/sample_submission.csv')

X_train_raw = train.drop(['CoilID', 'Y'], axis=1)
y_train     = train['Y']
test_ids    = test['CoilID']
X_test_raw  = test.drop(['CoilID'], axis=1)

print("=" * 55)
print(f"  TRAIN  →  rows: {train.shape[0]:>5}  |  cols: {train.shape[1]}")
print(f"  TEST   →  rows: {test.shape[0]:>5}  |  cols: {test.shape[1]}")
print(f"  SAMPLE →  rows: {sample.shape[0]:>5}  |  cols: {sample.shape[1]}")
print("=" * 55)
print(f"\n  Class Distribution (Train):")
vc = y_train.value_counts()
print(f"    Y=0 (Normal) : {vc[0]:>4}  ({vc[0]/len(y_train)*100:.2f}%)")
print(f"    Y=1 (Defect) : {vc[1]:>4}  ({vc[1]/len(y_train)*100:.2f}%)")
print(f"    Imbalance Ratio: 1:{vc[0]//vc[1]}")
print("=" * 55)

print(f"\n  Missing Values (Train): {X_train_raw.isnull().sum().sum()} total")
print(f"  Missing Values (Test) : {X_test_raw.isnull().sum().sum()} total")
miss_cols = X_train_raw.isnull().sum()
print(f"  Columns with NaN: {(miss_cols > 0).sum()} features affected")

#Exploratory Data Analysis

# ── 2.1 Statistical Summary by Class ──
print("Statistical Summary — Defect Coils vs Normal Coils")
print("-" * 55)
defect_mask = y_train == 1
feat_cols = X_train_raw.columns.tolist()

stats = pd.DataFrame({
    'Normal_Mean'  : X_train_raw[~defect_mask].mean(),
    'Defect_Mean'  : X_train_raw[defect_mask].mean(),
    'Normal_Std'   : X_train_raw[~defect_mask].std(),
    'Defect_Std'   : X_train_raw[defect_mask].std(),
})
stats['Mean_Diff'] = (stats['Defect_Mean'] - stats['Normal_Mean']).abs()
stats = stats.sort_values('Mean_Diff', ascending=False)
print(stats.head(15).round(4))

# ── 2.2 Missing value breakdown ──
print("\n  Top Features with Missing Values:")
miss = X_train_raw.isnull().sum().sort_values(ascending=False)
miss = miss[miss > 0]
if len(miss) > 0:
    print(miss.head(10))
else:
    print("  No features have >0 missing values in top 10")

# ── 2.3 Skewness ──
skew = X_train_raw.skew().sort_values(ascending=False)
print(f"\n  High-skew features (|skew| > 2): {(skew.abs() > 2).sum()}")
print(f"  Most skewed: {skew.head(5).to_dict()}")



#Preprocessing Pipeline

# ── 3.1 Median Imputation (robust to outliers) ──
imputer = SimpleImputer(strategy='median')
X_tr_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
X_te_imp = pd.DataFrame(imputer.transform(X_test_raw),      columns=X_test_raw.columns)
print(f"✅ Imputation done | Remaining NaN — Train: {X_tr_imp.isnull().sum().sum()} | Test: {X_te_imp.isnull().sum().sum()}")

#Outlier clipping at 1st-99th percentile ──
for col in X_tr_imp.columns:
    lo, hi = X_tr_imp[col].quantile(0.01), X_tr_imp[col].quantile(0.99)
    X_tr_imp[col] = X_tr_imp[col].clip(lo, hi)
    X_te_imp[col] = X_te_imp[col].clip(lo, hi)
print("✅ Outlier clipping done (1st–99th percentile per feature)")

# ── 3.3 Mutual Information feature importance ──
mi_scores = mutual_info_classif(X_tr_imp, y_train, random_state=SEED)
mi_df = pd.DataFrame({'Feature': X_train_raw.columns, 'MI_Score': mi_scores})
mi_df = mi_df.sort_values('MI_Score', ascending=False)
print(f"\n  Top 10 features by Mutual Information:")
print(mi_df.head(10).to_string(index=False))

# Advanced Feature Engineering

# %%
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

X_tr_fe = engineer_features(X_tr_imp)
X_te_fe = engineer_features(X_te_imp)

# ── Isolation Forest anomaly score ──
iso = IsolationForest(n_estimators=400, contamination=0.05, max_samples='auto', random_state=SEED)
iso.fit(X_tr_fe[y_train == 0].values)
X_tr_fe['iso_anomaly'] = -iso.score_samples(X_tr_fe.values)
X_te_fe['iso_anomaly'] = -iso.score_samples(X_te_fe.values)

print(f"✅ Feature engineering complete")
print(f"   Original features   : {X_train_raw.shape[1]}")
print(f"   Engineered features : {X_tr_fe.shape[1]}")
print(f"   Added               : {X_tr_fe.shape[1] - X_train_raw.shape[1]} new features")
print(f"\n   Isolation Forest anomaly signal:")
print(f"   Defect coils  → mean anomaly: {X_tr_fe.loc[y_train==1,'iso_anomaly'].mean():.4f}")
print(f"   Normal coils  → mean anomaly: {X_tr_fe.loc[y_train==0,'iso_anomaly'].mean():.4f}")


# Stratified K-Fold Ensemble Training

# %%
X_tr_arr = X_tr_fe.values
X_te_arr = X_te_fe.values

skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs  = np.zeros(len(y_train))
test_probs = np.zeros(len(X_te_arr))
fold_metrics = []

print(f"Training {N_FOLDS}-fold ensemble: LightGBM + XGBoost + CatBoost")
print("=" * 55)

for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_tr_arr, y_train)):
    Xf, yf = X_tr_arr[tr_idx], y_train.iloc[tr_idx]
    Xv, yv = X_tr_arr[val_idx], y_train.iloc[val_idx]

    fold_oof_preds  = []
    fold_test_preds = []

    # ── LightGBM ──
    lgb = LGBMClassifier(
        n_estimators=N_EST, class_weight={0:1, 1:CLASS_WT},
        learning_rate=LR, max_depth=MAX_DEPTH,
        min_child_samples=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED + fold_idx, verbose=-1
    )
    lgb.fit(Xf, yf, eval_set=[(Xv, yv)], callbacks=[])
    fold_oof_preds.append(lgb.predict_proba(Xv)[:, 1])
    fold_test_preds.append(lgb.predict_proba(X_te_arr)[:, 1])

    # ── XGBoost ──
    xgb = XGBClassifier(
        n_estimators=N_EST, scale_pos_weight=CLASS_WT,
        learning_rate=LR, max_depth=MAX_DEPTH,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED + fold_idx, eval_metric='logloss', verbosity=0
    )
    xgb.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
    fold_oof_preds.append(xgb.predict_proba(Xv)[:, 1])
    fold_test_preds.append(xgb.predict_proba(X_te_arr)[:, 1])

    # ── CatBoost ──
    cat = CatBoostClassifier(
        iterations=N_EST, class_weights={0:1, 1:CLASS_WT},
        learning_rate=LR, depth=MAX_DEPTH,
        l2_leaf_reg=3.0, random_seed=SEED + fold_idx, verbose=0
    )
    cat.fit(Xf, yf, eval_set=(Xv, yv))
    fold_oof_preds.append(cat.predict_proba(Xv)[:, 1])
    fold_test_preds.append(cat.predict_proba(X_te_arr)[:, 1])

    # ── Average ensemble ──
    oof_fold      = np.mean(fold_oof_preds, axis=0)
    test_fold     = np.mean(fold_test_preds, axis=0)
    oof_probs[val_idx] = oof_fold
    test_probs        += test_fold / N_FOLDS

    # Quick metric at 0.5
    fold_pred_hard = (oof_fold >= 0.5).astype(int)
    rec  = recall_score(yv, fold_pred_hard, zero_division=0)
    prec = precision_score(yv, fold_pred_hard, zero_division=0)
    roc  = roc_auc_score(yv, oof_fold)
    fold_metrics.append({'fold': fold_idx+1, 'recall': rec, 'precision': prec, 'roc_auc': roc})
    print(f"  Fold {fold_idx+1}/{N_FOLDS} → Recall@0.5: {rec:.3f} | Precision@0.5: {prec:.3f} | ROC-AUC: {roc:.4f}")

print("=" * 55)
fm = pd.DataFrame(fold_metrics)
print(f"  Mean ROC-AUC : {fm['roc_auc'].mean():.4f} ± {fm['roc_auc'].std():.4f}")
print("✅ Ensemble training complete")


# OOF Threshold Optimization

# ── Sweep threshold to find Recall=1.0 with max Precision ──
thresholds = np.arange(0.001, 0.60, 0.0005)
results = []

for t in thresholds:
    preds = (oof_probs >= t).astype(int)
    rec   = recall_score(y_train, preds, zero_division=0)
    prec  = precision_score(y_train, preds, zero_division=0)
    f1    = f1_score(y_train, preds, zero_division=0)
    n_pos = preds.sum()
    results.append((t, rec, prec, f1, n_pos))

res_df = pd.DataFrame(results, columns=['threshold','recall','precision','f1','n_positive'])

# Filter Recall = 1.0 → pick highest precision
recall1 = res_df[res_df['recall'] == 1.0]
best_row = recall1.loc[recall1['precision'].idxmax()]
best_thresh = best_row['threshold']
best_prec   = best_row['precision']

print("Threshold Sweep Summary (Recall = 1.0 rows):")
print(recall1[['threshold','recall','precision','f1','n_positive']].tail(10).to_string(index=False))
print()
print(f"  ★ Selected Threshold : {best_thresh:.4f}")
print(f"  ★ OOF Precision      : {best_prec:.4f}")
print(f"  ★ OOF Recall         : 1.0000  (0 False Negatives)")

# ── Final OOF classification report ──
final_oof_preds = (oof_probs >= best_thresh).astype(int)
print("\n  OOF Classification Report:")
print(classification_report(y_train, final_oof_preds, target_names=['Normal','Defect'], digits=4))

# ── Confusion matrix on OOF ──
cm = confusion_matrix(y_train, final_oof_preds)
print(f"  Confusion Matrix (OOF):")
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")



#Test Prediction & Submission Generation

# ── Apply optimized threshold to test probabilities ──
predictions = (test_probs >= best_thresh).astype(int)

print(f"Test Set Predictions Summary:")
print(f"  Total test coils   : {len(predictions)}")
print(f"  Predicted DEFECT   : {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)")
print(f"  Predicted NORMAL   : {(predictions==0).sum()} ({(predictions==0).sum()/len(predictions)*100:.2f}%)")
print(f"  Threshold used     : {best_thresh:.4f}")

# ── Build submission DataFrame ──
submission = pd.DataFrame({'CoilID': test_ids, 'Y': predictions})

# Verify format matches sample_submission
assert list(submission.columns) == ['CoilID', 'Y'], "Column mismatch!"
assert len(submission) == 339, f"Row count mismatch: {len(submission)}"
assert submission['Y'].isin([0,1]).all(), "Invalid Y values!"

print(f"\n✅ Submission validation passed")
print(f"   Shape      : {submission.shape}")
print(f"   Columns    : {list(submission.columns)}")
print(f"   Y unique   : {sorted(submission.Y.unique())}")
print()
print("First 10 predictions:")
print(submission.head(10).to_string(index=False))

submission.to_csv('expected_submission.csv', index=False)
print("\n✅ Saved: expected_submission.csv")


#Feature Importance & Model Insights


# ── Train final model on full data for feature importance ──
lgb_final = LGBMClassifier(
    n_estimators=N_EST, class_weight={0:1, 1:CLASS_WT},
    learning_rate=LR, max_depth=MAX_DEPTH,
    min_child_samples=3, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1
)
lgb_final.fit(X_tr_arr, y_train)

feat_names = X_tr_fe.columns.tolist()
importances = lgb_final.feature_importances_
imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
imp_df = imp_df.sort_values('Importance', ascending=False)

print("Top 20 Features by LightGBM Importance:")
print(imp_df.head(20).to_string(index=False))

print(f"\nKey Insight: iso_anomaly rank = {imp_df[imp_df.Feature=='iso_anomaly'].index[0]+1}")
print(f"Engineered features in Top-20: {(imp_df.head(20).Feature.str.startswith(('agg','stage','s1','s2','s3','s4','spike','crash','iso'))).sum()}")


