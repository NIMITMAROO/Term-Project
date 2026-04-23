import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ==============================
# Load Models
# ==============================
model = joblib.load("best_model_random_forest.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")

# ==============================
# Label Mapping (IMPORTANT)
# ==============================
label_map = {
    0: "High",
    1: "Low",
    2: "Medium"
}

class_names = ["High", "Low", "Medium"]

# ==============================
# App Config
# ==============================
st.set_page_config(page_title="Solubility Predictor", layout="wide")

st.title("💧 Aqueous Solubility Predictor")

st.markdown("""
This application predicts the aqueous solubility class of a compound based on molecular descriptors.

**Classes:**
- High Solubility
- Medium Solubility
- Low Solubility
""")

# ==============================
# Sidebar Input
# ==============================
st.sidebar.header("Input Molecular Descriptors")

input_data = {}

# Feature ranges (chemically reasonable)
feature_ranges = {
    "MolWt": (0.0, 1000.0),
    "MolLogP": (-5.0, 10.0),
    "TPSA": (0.0, 300.0),
    "NumHAcceptors": (0, 15),
    "NumHDonors": (0, 10),
    "NumRotatableBonds": (0, 20),
    "NumValenceElectrons": (0, 200),
    "RingCount": (0, 10)
}

for feature in feature_columns:
    min_val, max_val = feature_ranges.get(feature, (0.0, 100.0))

    input_data[feature] = st.sidebar.slider(
        feature,
        float(min_val),
        float(max_val),
        float((min_val + max_val) / 2)
    )

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# ==============================
# Display Input
# ==============================
st.subheader("🔍 Input Features")
st.write(input_df)

# ==============================
# Prediction
# ==============================
if st.button("Predict Solubility"):

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    probs = model.predict_proba(input_scaled)

    # Map prediction to label
    predicted_class = label_map[prediction[0]]

    # ==============================
    # Output Section
    # ==============================
    st.subheader("📊 Prediction Result")

    # Color-coded result
    if predicted_class == "High":
        st.success(f"Predicted Class: **{predicted_class} Solubility**")
    elif predicted_class == "Medium":
        st.warning(f"Predicted Class: **{predicted_class} Solubility**")
    else:
        st.error(f"Predicted Class: **{predicted_class} Solubility**")

    # ==============================
    # Probability Visualization
    # ==============================
    st.subheader("📈 Prediction Probabilities")

    prob_df = pd.DataFrame(probs, columns=class_names)

    st.bar_chart(prob_df.T)

    # Show exact probabilities
    st.write("### Detailed Probabilities")
    st.write(prob_df)