import streamlit as st
import pandas as pd
import joblib
from reportlab.pdfgen import canvas
from datetime import datetime
import base64
import os

# Load model + encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="FloraPredictAI", page_icon="🌱", layout="centered")

st.title("🌱 FloraPredictAI — Plant Survival Prediction")
st.write("Predict the most suitable plant species based on environmental parameters.")

# ---------- INPUT UI ----------
soil_type = st.selectbox("Soil Type", ["Sandy", "Clay", "Loamy", "Silty"])
light = st.selectbox("Light Availability", ["High", "Medium", "Low"])
moisture = st.selectbox("Moisture Level", ["High", "Medium", "Low"])
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)
disturbance = st.selectbox("Disturbance Level", ["Low", "Medium", "High"])
human = st.selectbox("Human Interference", ["Low", "Medium", "High"])
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)

if st.button("🔮 Predict Species"):
    
    # Build dataframe
    input_df = pd.DataFrame({
        "soil_type": [soil_type],
        "light": [light],
        "moisture": [moisture],
        "temperature": [temperature],
        "disturbance": [disturbance],
        "human_interference": [human],
        "ph": [ph]
    })
    
    # Encode categorical columns
    for col in ["soil_type", "light", "moisture", "disturbance", "human_interference"]:
        input_df[col] = encoders[col].transform(input_df[col])

    # Predict
    pred = model.predict(input_df)[0]
    species = encoders["species"].inverse_transform([pred])[0]

    # Confidence
    confidence = round(max(model.predict_proba(input_df)[0]) * 100, 2)

    # ---------- DISPLAY RESULT ----------
    st.success(f"🌿 **Recommended Species:** {species}")
    st.info(f"📊 **Confidence Level:** {confidence}%")

    # ---------- LOGGING ----------
    log_file = "prediction_logs.csv"
    exists = os.path.isfile(log_file)

    log_data = pd.DataFrame({
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "soil_type": [soil_type],
        "light": [light],
        "moisture": [moisture],
        "temperature": [temperature],
        "disturbance": [disturbance],
        "human_interference": [human],
        "ph": [ph],
        "species": [species],
        "confidence": [confidence]
    })

    if exists:
        log_data.to_csv(log_file, mode="a", header=False, index=False)
    else:
        log_data.to_csv(log_file, index=False)

    st.write("📁 Prediction logged successfully.")

    # ---------- PDF GENERATION ----------
    pdf_name = f"FloraReport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    c = canvas.Canvas(pdf_name)

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, 800, "FloraPredictAI — Prediction Report")

    c.setFont("Helvetica", 12)
    y = 760
    fields = {
        "Soil Type": soil_type,
        "Light Availability": light,
        "Moisture Level": moisture,
        "Temperature (°C)": temperature,
        "Disturbance Level": disturbance,
        "Human Interference": human,
        "pH Level": ph,
        "Predicted Species": species,
        "Confidence (%)": f"{confidence}%"
    }

    for key, value in fields.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    c.save()

    # Convert PDF to downloadable link
    with open(pdf_name, "rb") as f:
        pdf_bytes = f.read()
        b64 = base64.b64encode(pdf_bytes).decode()

    href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_name}">📄 Download PDF Report</a>'
    st.markdown(href, unsafe_allow_html=True)
