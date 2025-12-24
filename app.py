import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained objects
# -----------------------------
model = joblib.load("fruits_model/fruit_model.pkl")
encoders = joblib.load("fruits_model/encoders.pkl")
scaler = joblib.load("fruits_model/scaler.pkl")
target_encoder = joblib.load("fruits_model/target_encoder.pkl")

# Feature order used during training
feature_order = model.feature_names_in_

st.set_page_config(page_title="Fruit Classification App")
st.title("🍎 Fruit Type Prediction")

st.write("Select fruit characteristics to predict fruit type")

# -----------------------------
# USER INPUT
# -----------------------------
user_data = {}

st.sidebar.header("Numeric Features")

for feature in feature_order:
    if feature in encoders:
        # Categorical features (main page)
        user_data[feature] = st.selectbox(
            f"Select {feature}",
            encoders[feature].classes_
        )
    else:
        # Numeric features (sidebar sliders)
        user_data[feature] = st.sidebar.slider(
            f"{feature}",
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Fruit Type"):
    input_df = pd.DataFrame([user_data])

    # Encode categorical features
    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    # Scale numeric features
    numeric_cols = scaler.feature_names_in_
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Ensure correct feature order
    input_df = input_df[feature_order]

    # Predict
    prediction = model.predict(input_df)
    fruit = target_encoder.inverse_transform(prediction)[0]

    st.success(f"🍓 Predicted Fruit: **{fruit}**")
