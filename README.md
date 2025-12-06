# 🌱 FloraPredictAI

FloraPredictAI is a machine learning-based plant species survival prediction system.  
It predicts the most suitable plant species for a given environment using environmental and human impact factors.

This project is built as a scalable prototype that currently runs via Command Line Interface (CLI) and can be easily extended into a web or mobile application.

---

## 🚀 Features
- ✅ Predicts plant species based on environmental conditions
- ✅ Uses a Decision Tree Classifier
- ✅ Command Line Interface (CLI) based
- ✅ Accuracy evaluation
- ✅ Scalable for web/app integration
- ✅ Lightweight and fast

---

## 🧠 Input Parameters
- Soil Type  
- Light Availability  
- Moisture  
- Temperature  
- Disturbance Level  
- Human Interference  
- pH Level

---

## 🎯 Output
- Most suitable plant species for survival under given conditions

---

## 🛠 Tech Stack
- Python
- Pandas
- Scikit-learn
- Joblib

---

## 📁 Project Structure
```
FloraPredictAI/
│ 
├── dataset.csv 
├── train_model.py 
├── predict.py 
├── model.pkl 
├── encoder.pkl 
└── requirements.txt
```

---

## ⚙️ Installation
```bash
pip install pandas scikit-learn joblib
```

▶️ How to Run
Train the Model:-
```bash
python train_model.py
```
Predict a Species
```bash
python predict.py
```
#🔮 Future Scope

Web App using Flask / FastAPI

Android & iOS App Integration

Real-time sensor-based predictions

Large-scale agricultural datasets

Cloud-based ML deployment

#👨‍💻 Developed By-
Suvojoti Howlader
