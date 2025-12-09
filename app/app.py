import sys
import os

# Add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import streamlit as st
import pandas as pd

from src.inference import FraudModelService
from src.config import FEATURE_COLS

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

@st.cache_resource
def load_service():
    return FraudModelService()

service = load_service()

st.title("💳 Credit Card Fraud Detection")
st.write(
    """
    This app uses a machine learning model trained on the ULB credit card dataset
    to predict whether a credit card transaction is likely fraudulent.
    """
)

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "About PCA"])

with tab1:
    st.subheader("Single Transaction Inference")
    st.write(
        "Enter transaction details below. For PCA features (V1–V28), you can keep them at 0.0 "
        "for demo purposes or paste real values from the dataset."
    )

    # Basic inputs for Time and Amount
    time_val = st.number_input("Time (seconds since first transaction)", value=0.0)
    amount_val = st.number_input("Amount", value=1.0, min_value=0.0)

    # Build input fields for all PCA components
    pca_values = {}
    for col in [c for c in FEATURE_COLS if c not in ["Time", "Amount"]]:
        pca_values[col] = st.number_input(col, value=0.0)

    if st.button("Predict Fraud", type="primary"):
        features = {
            "Time": time_val,
            "Amount": amount_val,
            **pca_values,
        }

        result = service.predict_single(features)

        st.markdown("### Prediction Result")
        st.metric(
            label="Fraud Probability",
            value=f"{result['fraud_probability']:.4f}"
        )
        st.caption(f"Decision threshold: {result['threshold']:.4f}")

        if result["is_fraud"]:
            st.error("⚠️ This transaction is likely FRAUDULENT.")
        else:
            st.success("✅ This transaction is likely LEGITIMATE.")

with tab2:
    st.subheader("Batch CSV Prediction")
    st.write(
        """
        Upload a CSV file for batch prediction.  
        **Required columns:** `Time`, `Amount`, `V1` ... `V28`  
        (You can take a subset of the original Kaggle dataset, drop the `Class` column and upload it.)
        """
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
        else:
            missing = set(FEATURE_COLS) - set(df.columns)
            if missing:
                st.error(f"Uploaded file is missing required columns: {missing}")
            else:
                proba, labels = service.predict_batch(df)
                df_result = df.copy()
                df_result["fraud_probability"] = proba
                df_result["is_fraud"] = labels

                st.write("Preview of predictions:")
                st.dataframe(df_result.head(20))

                csv_data = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Download predictions as CSV",
                    data=csv_data,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )
with tab3:
    st.header("ℹ️ Understanding PCA Components (V1–V28)")
    st.markdown("""
### 🔍 What Are the PCA Components (V1–V28)?

The features **V1, V2, ..., V28** in this dataset are **not original transaction fields**.  
They are **anonymized PCA components**, created by applying *Principal Component Analysis (PCA)* on confidential credit card data.

---

### 🔐 Why PCA?
Banks cannot share sensitive transaction details like merchant category, customer profile, location, etc.  
So researchers replaced those features with **mathematical transformations** that:
- Protect privacy  
- Reduce dimensionality  
- Capture important fraud patterns  

---

### 🧠 What does a PCA component represent?

A PCA component is a **compressed mixture** of many original transaction attributes.

It is **not interpretable** as:
- Merchant  
- Country  
- User income  
- Card type  
- Risk level  

Instead, it represents a direction in the data that explains variation.

Think of it like compressing a photo:
- You lose the original pixels  
- But important structure is kept  

---

### 📉 Why do PCA features help fraud detection?

Fraud often appears as:
- Unusual spending behavior  
- Irregular transaction timing  
- Deviations from past patterns  

PCA transforms the data so these anomalies **stand out clearly**.

This is why components like **V14, V17, V4** often become important indicators of fraud.

---

### 🧭 Summary

| Question | Answer |
|---------|--------|
| What are V1–V28? | PCA-transformed features |
| Why used? | Privacy protection + signal extraction |
| Can we interpret them? | No — they are abstract math components |
| Are they useful? | Yes — they capture fraud-driven patterns |

PCA allows sharing the dataset **without exposing sensitive bank data** while keeping the underlying fraud signals intact.
    """)
