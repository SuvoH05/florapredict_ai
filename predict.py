import joblib
import pandas as pd

# Load saved model + encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

print("\n🌱 FloraPredictAI — Plant Survival Prediction\n")

# EXACT INPUT FIELDS YOU SPECIFIED
soil_type = input("Soil Type (Sandy / Clay / Loamy / Silty): ")
light = input("Light (High / Medium / Low): ")
moisture = input("Moisture (High / Medium / Low): ")
temperature = float(input("Temperature (°C): "))
disturbance = input("Disturbance (Low / Medium / High): ")
human = input("Human Interference (Low / Medium / High): ")
ph = float(input("pH Level (e.g., 6.5): "))

# Build DataFrame exactly matching training schema
input_df = pd.DataFrame({
    "soil_type": [soil_type],
    "light": [light],
    "moisture": [moisture],
    "temperature": [temperature],
    "disturbance": [disturbance],
    "human_interference": [human],
    "ph": [ph]
})

# Encode categorical columns using their own encoders
for col in ["soil_type", "light", "moisture", "disturbance", "human_interference"]:
    input_df[col] = encoders[col].transform(input_df[col])

# Predict
prediction = model.predict(input_df)[0]

# Decode species label
species = encoders["species"].inverse_transform([prediction])[0]

print("\n🌿 Recommended Plant Species:", species)
print("\n✅ Prediction complete!\n")
