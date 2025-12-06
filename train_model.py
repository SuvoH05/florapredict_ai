import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")

# Categorical columns based on your exact input fields
categorical_cols = [
    "soil_type",
    "light",
    "moisture",
    "disturbance",
    "human_interference",
    "species"
]

# Create separate encoder for each categorical field
encoders = {}

for col in categorical_cols:
    enc = LabelEncoder()
    data[col] = enc.fit_transform(data[col])
    encoders[col] = enc

# Features & label
X = data.drop("species", axis=1)
y = data["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("🎉 Model trained successfully!")
print("📊 Accuracy:", round(accuracy * 100, 2), "%")

# Save model + encoders
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("\n✅ model.pkl and encoders.pkl saved.")
