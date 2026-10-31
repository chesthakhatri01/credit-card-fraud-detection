import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled"]

# Row 0 of the dataset (a genuine transaction), same sample as predict_new_transaction.ipynb
SAMPLE = {
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155, "V5": -0.338321,
    "V6": 0.462388, "V7": 0.239599, "V8": 0.098698, "V9": 0.363787, "V10": 0.090794,
    "V11": -0.551600, "V12": -0.617801, "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
    "V16": -0.470401, "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
    "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928, "V25": 0.128539,
    "V26": -0.189115, "V27": 0.133558, "V28": -0.021053, "Amount": 149.62,
}

# Fallback if amount_scaler.pkl is not in the repo: mean / std of Amount over the full dataset
AMOUNT_MEAN, AMOUNT_STD = 88.349619, 250.119670


@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")


@st.cache_resource
def load_scaler():
    if os.path.exists("amount_scaler.pkl"):
        return joblib.load("amount_scaler.pkl")
    return None


def scale_amount(amounts):
    sc = load_scaler()
    if sc is not None:
        return sc.transform(amounts.to_frame("Amount"))[:, 0]
    return (amounts - AMOUNT_MEAN) / AMOUNT_STD


def prepare(df):
    out = df.copy()
    out["Amount_scaled"] = scale_amount(out["Amount"].astype(float))
    return out[FEATURES]


model = load_model()

st.title("💳 Credit Card Fraud Detector")
st.caption("Random Forest trained on 284,807 transactions (0.17% fraud). "
           "Test set: fraud precision 0.92, recall 0.82, F1 0.86.")

tab1, tab2, tab3 = st.tabs(["Single transaction", "Batch (CSV)", "About"])

with tab1:
    st.write("Enter the 28 anonymised PCA features (V1–V28) and the amount.")
    if "vals" not in st.session_state:
        st.session_state.vals = dict(SAMPLE)
    if st.button("Load sample transaction"):
        st.session_state.vals = dict(SAMPLE)
        st.rerun()

    amount = st.number_input("Amount", min_value=0.0, value=float(st.session_state.vals["Amount"]), step=1.0)
    cols = st.columns(4)
    row = {}
    for i in range(1, 29):
        k = f"V{i}"
        row[k] = cols[(i - 1) % 4].number_input(k, value=float(st.session_state.vals[k]),
                                                format="%.6f", key=f"in_{k}_{st.session_state.vals[k]}")
    row["Amount"] = amount

    if st.button("Check transaction", type="primary"):
        X = prepare(pd.DataFrame([row]))
        pred = int(model.predict(X)[0])
        p_fraud = float(model.predict_proba(X)[0][1])
        if pred == 1:
            st.error(f"🚨 FRAUD detected — fraud probability {p_fraud*100:.2f}%")
        else:
            st.success(f"✅ Genuine transaction — fraud probability {p_fraud*100:.2f}%")

with tab2:
    st.write("Upload a CSV with columns `V1`…`V28` and `Amount` (extra columns such as `Time` or `Class` are ignored).")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        need = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            X = prepare(df)
            res = df.copy()
            res["fraud_probability"] = model.predict_proba(X)[:, 1]
            res["prediction"] = model.predict(X)
            n = int(res["prediction"].sum())
            st.metric("Flagged as fraud", f"{n} of {len(res):,}")
            st.dataframe(res.sort_values("fraud_probability", ascending=False).head(200))
            st.download_button("Download results", res.to_csv(index=False), "predictions.csv", "text/csv")

with tab3:
    st.markdown("""
**How it works:** Amount is standardised, then the 29 features go to a Random Forest (100 trees, balanced class weights).

| Model | Precision (fraud) | Recall (fraud) | F1 |
|---|---|---|---|
| Logistic Regression | 0.06 | 0.92 | 0.11 |
| Random Forest (used here) | 0.92 | 0.82 | 0.86 |

**Limitations:** features V1–V28 are anonymised, so real card data cannot be entered directly; this is an academic demo, not a production fraud system.
""")
