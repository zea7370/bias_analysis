import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from modules.preprocessing import load_and_preprocess_data
from modules.model import train_model, evaluate_model
from modules.fairness_metrics import compute_fairness
from modules.mitigation import mitigate_bias

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bias Detection & Fairness Analysis",
    layout="wide"
)

st.title("Bias Detection and Fairness Analysis in ML Decision Systems")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Adult Census Dataset",
    type=["csv"]
)

# ---------------- MAIN LOGIC ----------------
if uploaded_file is not None:
    with st.spinner("Processing data... Please wait."):
        # Load & preprocess data
        X_train, X_test, y_train, y_test, sens_train, sens_test = load_and_preprocess_data(uploaded_file)

        # Train model
        model = train_model(X_train, y_train)
        performance, y_pred = evaluate_model(model, X_test, y_test)

        # ---------------- PERFORMANCE ----------------
        st.subheader("Model Performance")
        st.json(performance)

        st.subheader("Model Performance Visualization")

        perf_df = pd.DataFrame.from_dict(
            performance,
            orient="index",
            columns=["Score"]
        )

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(perf_df.index, perf_df["Score"])
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Score")
        ax1.set_title("Model Performance Metrics")

        st.pyplot(fig1, clear_figure=True)
        plt.close(fig1)

        # ---------------- FAIRNESS BEFORE ----------------
        st.subheader("Fairness Metrics (Before Mitigation)")

        fairness = compute_fairness(
            y_test,
            y_pred,
            sens_test["sex"]
        )

        st.json(fairness)

        # ---------------- BIAS MITIGATION ----------------
        st.subheader("Applying Bias Mitigation...")

        mitigated_model = mitigate_bias(
            X_train,
            y_train,
            sens_train["sex"]
        )

        y_pred_mitigated = mitigated_model.predict(X_test)

        # ---------------- FAIRNESS AFTER ----------------
        st.subheader("Fairness Metrics (After Mitigation)")

        fairness_after = compute_fairness(
            y_test,
            y_pred_mitigated,
            sens_test["sex"]
        )

        st.json(fairness_after)

        # ---------------- FAIRNESS COMPARISON CHART ----------------
        st.subheader("Fairness Metrics Comparison (Before vs After)")

        fair_df = pd.DataFrame({
            "Before Mitigation": fairness,
            "After Mitigation": fairness_after
        })

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fair_df.plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Metric Value")
        ax2.set_title("Fairness Metrics Before vs After Bias Mitigation")
        ax2.axhline(0, color="black", linewidth=0.8)

        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)

else:
    st.info("👆 Please upload the dataset to begin analysis.")
