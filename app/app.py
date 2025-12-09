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

tab_single, tab_batch = st.tabs(["Single Transaction", "Batch Prediction"])

with tab_single:
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

with tab_batch:
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
