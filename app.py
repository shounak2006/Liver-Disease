import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Load trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Liver Disease Analyzer")

st.title("Liver Disease Risk Analyzer")
st.caption("Virtual decision-support system based on liver function test values")

st.subheader("Patient Information")

age = st.number_input("Age", min_value=1, max_value=100)
gender = st.selectbox("Gender", ["Male", "Female"])

st.subheader("Liver Function Test Parameters")

total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", help="Normal: ~0.1 – 1.2")
direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", help="Normal: < 0.3")
alk_phosphate = st.number_input("Alkaline Phosphatase (IU/L)", help="Normal: ~44 – 147")
alt = st.number_input("ALT / SGPT (IU/L)", help="Normal: ~7 – 56")
ast = st.number_input("AST / SGOT (IU/L)", help="Normal: ~10 – 40")
total_proteins = st.number_input("Total Proteins (g/dL)", help="Normal: ~6.0 – 8.3")
albumin = st.number_input("Albumin (g/dL)", help="Normal: ~3.5 – 5.0")
agr = st.number_input("Albumin–Globulin Ratio", help="Normal: ~1.0 – 2.5")


if st.button("Analyze Liver Health"):
    gender_val = 1 if gender == "Male" else 0

    input_df = pd.DataFrame([[
        age,
        gender_val,
        total_bilirubin,
        direct_bilirubin,
        alk_phosphate,
        alt,
        ast,
        total_proteins,
        albumin,
        agr
    ]], columns=[
        "Age",
        "Gender",
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alkaline_Phosphotase",
        "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Total_Proteins",
        "Albumin",
        "Albumin_and_Globulin_Ratio"
    ])

    prediction = model.predict(input_df)[0]
    st.subheader("Liver Test Overview")

    chart_data = pd.DataFrame({
    "Parameter": [
        "Total Bilirubin",
        "Direct Bilirubin",
        "Alkaline Phosphatase",
        "ALT",
        "AST",
        "Total Proteins",
        "Albumin",
        "A/G Ratio"
    ],
    "Value": [
        total_bilirubin,
        direct_bilirubin,
        alk_phosphate,
        alt,
        ast,
        total_proteins,
        albumin,
        agr
     ]
    })

    st.bar_chart(chart_data.set_index("Parameter"))

    probability = model.predict_proba(input_df)[0][1]


    st.subheader("Result")

    if prediction == 1:
        st.error("High Risk of Liver Disease Detected")
        st.write(f"Risk Probability: {probability:.2f}")
    else:
        st.success("Low Risk of Liver Disease Detected")
        st.write(f"Risk Probability: {probability:.2f}")

        st.subheader("Model Performance Overview")

    # Confusion matrix from final Logistic Regression evaluation
    cm = [
        [7, 27],
        [4, 79]
    ]

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix – Logistic Regression")

    st.pyplot(fig)


st.warning("⚠ This system is for screening assistance only and not a diagnostic tool.")
