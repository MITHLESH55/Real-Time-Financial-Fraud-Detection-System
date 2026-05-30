import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mcdm_ai import FraudDetectionSystem, calculate_weights_ahp, calculate_classification_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


st.set_page_config(layout="wide")

# ------------------------------------------
# Setup Criteria, Weights, and Preferences
# ------------------------------------------
criteria = ["Transaction Amount", "Transaction Frequency", "Time of Day",
            "Location Match", "IP Address Risk", "Device Recognition"]

benefit_criteria = np.array([False, False, False, True, False, True])

criteria_comparisons = np.array([
    [1, 3, 5, 7, 3, 5],
    [1/3, 1, 3, 5, 1, 3],
    [1/5, 1/3, 1, 3, 1/3, 1],
    [1/7, 1/5, 1/3, 1, 1/5, 1/3],
    [1/3, 1, 3, 5, 1, 3],
    [1/5, 1/3, 1, 3, 1/3, 1]
])

weights, cr = calculate_weights_ahp(criteria_comparisons)

input_ranges = {
    "Transaction Amount": (0, 1000000),
    "Transaction Frequency": (0, 100),
    "Time of Day": (0, 24),
    "Location Match": (0.0, 1.0),
    "IP Address Risk": (0.0, 1.0),
    "Device Recognition": (0.0, 1.0)
}

# ------------------------------------------
# UI: Transaction Input
# ------------------------------------------
st.title("🔍 Real-time Fraud Detection & Prevention")

n_txns = st.number_input("Enter number of transactions", min_value=1, max_value=50, value=3)

st.subheader("📥 Enter Transaction Data")
transactions = []
for i in range(n_txns):
    with st.expander(f"Transaction {i + 1}"):
        txn = []
        for crit in criteria:
            min_val, max_val = input_ranges[crit]
            val = st.number_input(f"{crit} (T{i+1})", min_value=min_val, max_value=max_val, key=f"{crit}_{i}")
            txn.append(val)
        transactions.append(txn)

# ------------------------------------------
# Run Detection
# ------------------------------------------
if st.button("🚨 Run Detection"):
    fds = FraudDetectionSystem(criteria, criteria_comparisons, weights, benefit_criteria)
    transaction_names = [f"T{i+1}" for i in range(n_txns)]
    fds.load_transaction_data(transactions, transaction_names)
    st.session_state["fds"] = fds

    detection_results = fds.detect_fraud(method="ensemble")
    st.session_state["detection_results"] = detection_results

    st.subheader("📊 Fraud Detection Results")
    fig1 = fds.visualize_detection_results()
    st.pyplot(fig1)

    # Real strategy names from mcdm_ai
    strategy_names = [
        "Enhanced Authentication",
        "Transaction Limits",
        "Time-based Rules",
        "Location Verification",
        "Device Fingerprinting",
        "Behavioral Biometrics",
        "Transaction Monitoring"
    ][:5]  # You only generate 5 dummy strategies — adapt if needed

    prevention_strategies = np.random.rand(5, len(criteria))
    fds.load_prevention_strategies(prevention_strategies, strategy_names)

    if len(detection_results["majority_suspicious"]) > 0:
        suspicious_index = detection_results["majority_suspicious"][0]
        fds.recommend_strategies(method="ensemble", suspicious_transaction=suspicious_index)

        st.subheader("🛡️ Recommended Prevention Strategies")
        fig2 = fds.visualize_prevention_results()
        st.pyplot(fig2)

        st.subheader("🧪 Risk Factor Analysis")
        analysis = fds.analyze_transaction(suspicious_index, method="ensemble")
        fig3 = fds.visualize_risk_analysis(analysis)
        st.pyplot(fig3)

    st.success("✅ Detection and recommendation complete!")

# ------------------------------------------
# Show Summary Metrics & Plots
# ------------------------------------------
if st.button("📊 Show Ensemble Summary Plots"):
    if "fds" in st.session_state and "detection_results" in st.session_state:
        fds = st.session_state["fds"]
        detection_results = st.session_state["detection_results"]

        
        
        # Comparison plots
        comparison = fds.compare_methods()
        st.session_state["comparison"] = comparison
        figs = fds.visualize_method_comparison(comparison)

        st.subheader("🧠 Detection Agreement (Jaccard Index)")
        st.pyplot(figs["detection_agreement"])

        st.subheader("🎯 Prevention Strategy Correlation")
        st.pyplot(figs["prevention_correlation"])

        st.subheader("🔢 Suspicious Counts by Method")
        st.pyplot(figs["suspicious_counts"])
        
    else:
        st.warning("Please run detection first.")
