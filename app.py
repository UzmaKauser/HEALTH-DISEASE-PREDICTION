import streamlit as st
import joblib
import numpy as np
import requests
import json


# Mapping of numerical labels to disease names (update this based on your dataset)
disease_mapping = {
    0: "Fungal Infection",
    1: "Allergy",
    2: "Gerd",
    3: "Chronic Cholestasis",
    4: "Drug Reaction",
    5: "Peptic Ulcer Disease",
    6: "AIDS",
    7: "Diabetes",
    8: "Gastroenteritis",
    9: "Bronchial Asthma",
    10: "Hypertension",
    11: "Migraine",
    12: "Cervical Spondylosis",
    13: "Paralysis (Brain Hemorrhage)",
    14: "Jaundice",
    15: "Malaria",
    16: "Chicken Pox",
    17: "Dengue",
    18: "Typhoid",
    19: "Hepatitis A",
    20: "Hepatitis B",
    21: "Hepatitis C",
    22: "Hepatitis D",
    23: "Hepatitis E",
    24: "Alcoholic Hepatitis",
    25: "Tuberculosis",
    26: "Common Cold",
    27: "Pneumonia",
    28: "Dimorphic Hemorrhoids (Piles)",
    29: "Heart Attack",
    30: "Varicose Veins",
    31: "Hypothyroidism",
    32: "Hyperthyroidism",
    33: "Hypoglycemia",
    34: "Osteoarthritis",
    35: "Arthritis",
    36: "Vertigo Paroymsal Positional Vertigo",
    37: "Acne",
    38: "Urinary Tract Infection",
    39: "Psoriasis",
    40: "Impetigo"
}


# Streamlit UI
st.set_page_config(page_title="Disease Prediction App", layout="centered")
st.title("🩺 Disease Prediction App")
st.write("Enter your details and symptoms to predict the disease.")

# User inputs
name = st.text_input("👤 Name")
age = st.number_input("🎂 Age", min_value=1, max_value=120, step=1)

st.write("### ✅ Select Symptoms")
symptoms = {
    "Fatigue": st.checkbox("Fatigue"),
    "High Fever": st.checkbox("High Fever"),
    "Headache": st.checkbox("Headache"),
    "Nausea": st.checkbox("Nausea"),
    "Loss of Appetite": st.checkbox("Loss of Appetite"),
}

# Convert selected symptoms into a numerical list (1 for checked, 0 for unchecked)
input_features = np.array([int(value) for value in symptoms.values()]).reshape(1, -1)

# Predict button
# Predict button
if st.button("🩺 Predict Disease"):
    if not name or age <= 0:
        st.error("⚠️ Please enter a valid Name and Age.")
    else:
        data = {
            "name": name,
            "age": age,
            "symptoms": input_features.flatten().tolist()  # Convert array to list for JSON
        }

        # Send data to Flask backend
        response = requests.post(" http://172.20.10.4:8503", json=data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"🩺 **Prediction for {name} (Age {age}):** {result['prediction']}")
        else:
            st.error("⚠️ Failed to get prediction. Please check the backend.")


st.write("---")
st.write("🔬 This app uses a **RandomForestClassifier** trained on symptom data.")
