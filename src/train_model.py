import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Column names
columns = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Proteins",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
    "Target"
]

# Load dataset
df = pd.read_csv(
    "data/raw/indian_liver_patient.csv",
    header=None,
    names=columns
)

# Preprocessing
df["Albumin_and_Globulin_Ratio"].fillna(
    df["Albumin_and_Globulin_Ratio"].median(),
    inplace=True
)
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Target"] = df["Target"].map({1: 1, 2: 0})

X = df.drop("Target", axis=1)
y = df["Target"]

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train final model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")
