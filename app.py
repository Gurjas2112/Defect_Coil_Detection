"""Streamlit app for testing the Alpha-defect detection model.

Run locally:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import FEATURE_COLS, transform, validate_input

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / 'models' / 'defect_model.joblib'
SAMPLE_PATH = ROOT / 'dataset' / 'test.csv'

st.set_page_config(page_title="Coil Alpha-Defect Detection", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_sample_csv():
    return pd.read_csv(SAMPLE_PATH).head(20).to_csv(index=False).encode('utf-8')


def predict_proba(bundle, df):
    X = transform(df, bundle['prep']).values
    return np.mean([m.predict_proba(X)[:, 1] for m in bundle['models']], axis=0)


def oof_metrics_at(bundle, threshold):
    probs, y = bundle['oof_probs'], bundle['oof_labels']
    pred = probs >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return recall, precision, fn, fp


if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train.py` to create `models/defect_model.joblib`.")
    st.stop()

bundle = load_bundle()
default_t = float(bundle['threshold'])

# ── Sidebar ──
with st.sidebar:
    st.header("About")
    st.write(
        "Predicts **Alpha defects** in hot-rolled steel coils from 49 process "
        "parameters (`X1`–`X49`) recorded across the rolling stages. "
        "The model is an ensemble of LightGBM, XGBoost and CatBoost."
    )

    st.header("Decision threshold")
    if 'threshold' not in st.session_state:
        st.session_state.threshold = default_t
    threshold = st.slider(
        "Flag a coil as defective when its probability is at least:",
        min_value=0.0005, max_value=0.95, step=0.0005, format="%.4f", key='threshold',
    )
    st.button("Reset to trained threshold",
              on_click=lambda: st.session_state.update(threshold=default_t))
    st.caption(f"Trained threshold: {default_t:.4f} (chosen for 100% recall on validation data).")

    recall, precision, fn, fp = oof_metrics_at(bundle, threshold)
    st.header("Validation performance at this threshold")
    c1, c2 = st.columns(2)
    c1.metric("Recall", f"{recall:.1%}")
    c2.metric("Precision", f"{precision:.1%}")
    c1.metric("Missed defects", fn)
    c2.metric("False alarms", fp)
    st.caption("Out-of-fold results on the 1352 training coils. A lower threshold catches more "
               "defects but raises more false alarms.")
    st.caption(f"Model trained: {bundle.get('trained_at', 'unknown')}")

# ── Main ──
st.title("Coil Alpha-Defect Detection")
st.write(
    "Upload a CSV with one row per coil. It must contain columns **`X1` … `X49`** "
    "(numeric; blanks are allowed and are filled with training medians). "
    "A **`CoilID`** column is optional and is carried through to the results."
)

st.download_button(
    "Download a sample CSV (20 test coils)", load_sample_csv(),
    file_name="sample_coils.csv", mime="text/csv",
)

uploaded = st.file_uploader("Upload coil data (CSV)", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to get predictions, or download the sample above to try it out.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the file as CSV: {exc}")
    st.stop()

problems = validate_input(df)
if problems:
    st.error("The uploaded file can't be scored:\n\n" + "\n".join(f"- {p}" for p in problems))
    st.stop()

inputs = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
with st.spinner(f"Scoring {len(df)} coil(s)..."):
    probs = predict_proba(bundle, inputs)

ids = df['CoilID'] if 'CoilID' in df.columns else pd.Series(range(1, len(df) + 1), name='CoilID')
results = pd.DataFrame({
    'CoilID': ids.values,
    'defect_probability': probs.round(4),
    'prediction': np.where(probs >= threshold, 'Defect', 'Normal'),
    'Y': (probs >= threshold).astype(int),
})

n_def = int(results['Y'].sum())
m1, m2, m3 = st.columns(3)
m1.metric("Coils scored", len(results))
m2.metric("Flagged as defect", n_def)
m3.metric("Flagged share", f"{n_def / len(results):.1%}")

st.subheader("Predictions")
st.dataframe(
    results.sort_values('defect_probability', ascending=False),
    use_container_width=True, hide_index=True,
    column_config={
        'defect_probability': st.column_config.ProgressColumn(
            "Defect probability", min_value=0.0, max_value=1.0, format="%.4f"),
    },
)

st.subheader("Probability distribution")
hist, edges = np.histogram(probs, bins=20, range=(0, 1))
st.bar_chart(pd.DataFrame({'coils': hist}, index=[f"{e:.2f}" for e in edges[:-1]]))

st.download_button(
    "Download predictions (CSV)", results.to_csv(index=False).encode('utf-8'),
    file_name="predictions.csv", mime="text/csv",
)
