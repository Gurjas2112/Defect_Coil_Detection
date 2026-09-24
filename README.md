# Hot-Rolling Steel Defect Detection — Approach

**Author:** Gurjas Singh Gandhi
**Implementation:** [defect_detection_gsg.py](defect_detection_gsg.py)

## Problem

Flag defective hot-rolled steel coils from 49 anonymised process parameters
(`X1..X49`). The classes are heavily imbalanced (66 defects out of 1352 training
coils, ~4.9%), so the hard part is recalling the rare defect class without
drowning in false positives.

## Pipeline overview

1. **Imputation + missingness signal.** Median-impute `X1..X49`, but first keep
   a `*_isna` flag per column and a `missing_total` count — a dropped sensor
   reading can itself indicate an abnormal coil.

2. **Feature engineering** (`build_features`):
   - Per-row descriptive stats: mean, std, median, min, max, range, IQR.
   - **Gradient features** over the ordered trace: mean/max absolute step
     between consecutive parameters and the number of sign changes — how
     abruptly the process varies.
   - **Rolling-window volatility** (windows 3/5/7) across `X1..X49`, treating
     each coil as a sequential process trace and summarising local roughness.
   - Extreme-value counts from per-row z-scores.

3. **Anomaly signals learned from normal coils only** (`assemble_matrix`):
   - **PCA reconstruction residual** — PCA is fit on `Y=0` coils, so defects,
     which live off the normal manifold, reconstruct poorly and score high.
   - **k-NN distance to normal coils** — mean distance to the 5 nearest normal
     coils in standardised space.

4. **Model.** A single LightGBM classifier evaluated with
   `RepeatedStratifiedKFold` (5 folds × 2 repeats). Imbalance is handled with
   `scale_pos_weight` set to twice the negative/positive ratio. Test
   probabilities are averaged across all fold models.

5. **Threshold selection** (`pick_threshold`). Computed only on out-of-fold
   probabilities to avoid overfitting the cutoff. It first tries the
   highest-precision threshold that still reaches the target recall; if that
   target is unreachable, it falls back to the threshold maximising F-beta with
   `beta=3` (recall weighted 3× precision).

## Results (out-of-fold)

| Metric | Value |
|---|---|
| OOF recall | ~0.85 |
| OOF precision | ~0.36 |
| Predicted defects on test | ~76 / 339 |

A few defects remain statistically indistinguishable from normal coils under
these features (their out-of-fold probability is near zero), so guaranteed
100% recall is not achieved with this feature set — the design instead leans
toward recall via the F-beta(3) fallback.

## Possible next steps

- A second, complementary anomaly model (e.g. one-class SVM or local outlier
  factor) blended with the gradient booster.
- Probability calibration (isotonic) for cleaner thresholding.
- Interaction features between the highest-importance parameters.

## Reproduce

```bash
pip install -r requirements.txt
python defect_detection.py   # writes submission_gsg.csv
```
