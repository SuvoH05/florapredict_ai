import joblib
import pandas as pd
from colorama import Fore, Style, init
from reportlab.pdfgen import canvas
from datetime import datetime
import os

# Enable color for Windows
init(autoreset=True)

# Load model + encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

print(Fore.GREEN + "\n🌱  FLORAPREDICT-AI — Plant Survival Prediction\n" + Style.RESET_ALL)

# Helper for validated choices
def choice(prompt, valid):
    while True:
        val = input(prompt).strip().title()
        if val in valid:
            return val
        print(Fore.RED + f"Invalid input! Choose from: {', '.join(valid)}\n")

# Valid categories
soil_options = ["Sandy", "Clay", "Loamy", "Silty"]
level_options = ["High", "Medium", "Low"]
disturbance_options = ["Low", "Medium", "High"]
human_options = ["Low", "Medium", "High"]

# USER INPUTS
soil_type = choice("Soil Type (Sandy / Clay / Loamy / Silty): ", soil_options)
light = choice("Light (High / Medium / Low): ", level_options)
moisture = choice("Moisture (High / Medium / Low): ", level_options)

while True:
    try:
        temperature = float(input("Temperature (°C): "))
        break
    except:
        print(Fore.RED + "Enter a valid number!\n")

disturbance = choice("Disturbance (Low / Medium / High): ", disturbance_options)
human = choice("Human Interference (Low / Medium / High): ", human_options)

while True:
    try:
        ph = float(input("pH Level (e.g., 6.5): "))
        break
    except:
        print(Fore.RED + "Enter a valid pH number!\n")

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

# Encode categoricals
for col in ["soil_type", "light", "moisture", "disturbance", "human_interference"]:
    input_df[col] = encoders[col].transform(input_df[col])

# Predict species
prediction = model.predict(input_df)[0]
species = encoders["species"].inverse_transform([prediction])[0]

# 🔥 Confidence Score
probs = model.predict_proba(input_df)[0]
confidence = max(probs) * 100

# Print Results Beautifully
print("\n" + Fore.CYAN + "="*50)
print(Fore.YELLOW + "🌿 Recommended Plant Species:")
print(Fore.GREEN + f"➡️  {species}")
print(Fore.YELLOW + f"\n📊 Confidence Level: {confidence:.2f}%")
print(Fore.CYAN + "="*50 + Style.RESET_ALL)

print(Fore.GREEN + "\n✨ Prediction complete!\n")

# 📁 Create predictions_log.csv if not exists
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
    "confidence": [round(confidence, 2)]
})

if exists:
    log_data.to_csv(log_file, mode="a", header=False, index=False)
else:
    log_data.to_csv(log_file, index=False)

print(Fore.BLUE + "📁 Logged prediction to prediction_logs.csv\n")

# 📝 PDF REPORT GENERATION
pdf_name = f"FloraReport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
c = canvas.Canvas(pdf_name)

c.setFont("Helvetica-Bold", 20)
c.drawString(50, 800, "FloraPredictAI: Prediction Report")

c.setFont("Helvetica", 12)
y = 760

fields = {
    "Soil Type": soil_type,
    "Light": light,
    "Moisture": moisture,
    "Temperature (°C)": temperature,
    "Disturbance": disturbance,
    "Human Interference": human,
    "pH Level": ph,
    "Predicted Species": species,
    "Confidence (%)": f"{confidence:.2f}"
}

for key, value in fields.items():
    c.drawString(50, y, f"{key}: {value}")
    y -= 20

c.save()

print(Fore.MAGENTA + f"📄 PDF Report Generated: {pdf_name}\n")
