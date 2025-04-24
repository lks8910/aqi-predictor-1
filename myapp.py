import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")  

# Define feature order (make sure it matches your training data)
FEATURES = ["CO", "NO", "NOX", "NO2", "O3", "PM10", "PM25", "RH", "SO2", "TMP", "WDR", "WSP", "traffic"]

# Page config and title
st.set_page_config(page_title="AQI Predictor", layout="centered")
st.title("🌫️ Air Quality Index (AQI) Predictor")
st.markdown("Predict AQI using environmental and traffic data.")

# Input mode selection
mode = st.radio("Choose Input Method:", ("Upload CSV", "Use Sliders"))

# --- CSV Upload Mode ---
if mode == "Upload CSV":
    uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Uploaded Data")
        st.dataframe(df.head())

        try:
            # Check if required columns are present
            if all(feature in df.columns for feature in FEATURES):
                predictions = model.predict(df[FEATURES])
                df["Predicted_AQI"] = predictions

                st.subheader("📈 Predicted AQI Values")
                st.dataframe(df[["Predicted_AQI"]].head())

                st.line_chart(df["Predicted_AQI"])

                # Download option
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results", data=csv, file_name="aqi_predictions.csv", mime="text/csv")
            else:
                missing = list(set(FEATURES) - set(df.columns))
                st.error(f"Missing columns in CSV: {', '.join(missing)}")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    else:
        st.info(f"Upload a CSV file with these columns:\n{', '.join(FEATURES)}")

# --- Slider Input Mode ---
elif mode == "Use Sliders":
    st.subheader("🎛️ Enter Feature Values")

    input_values = {}
    for feature in FEATURES:
        input_values[feature] = st.slider(f"{feature}", 0.0, 1.0, 0.5, 0.01)

    input_df = pd.DataFrame([input_values])

    if st.button("🚀 Predict AQI"):
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"🌍 Predicted AQI: **{prediction:.2f}**")
        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")
