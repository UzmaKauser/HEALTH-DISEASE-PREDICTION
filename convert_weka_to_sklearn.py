import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# Define feature names based on the tree
feature_names = ["itching", "skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "headache", "mood_swings", "abdominal_pain", "obesity"]

# Define training data (binary symptoms based on extracted decision rules from the tree)
X_train = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No symptoms
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Itching & Skin rash → Fungal Infection
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Chills → Dengue
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Joint Pain → Malaria
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # Vomiting & Fatigue → Hepatitis D
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # Fatigue & Headache → Hypertension
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # Mood Swings & Abdominal Pain & Obesity → Diabetes
])

# Corresponding disease labels
y_train = np.array([
    "No Disease", "Fungal Infection", "Dengue", "Malaria", "Hepatitis D", "Hypertension", "Diabetes"
])

# Train DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)

# Print the rules of the trained tree
print(export_text(clf, feature_names=feature_names))

# Example prediction
#sample_patient = np.array([[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]])  # Itching & Joint Pain
#prediction = clf.predict(sample_patient)
#print("Predicted Disease:", prediction[0])
import joblib

# Save the trained model
joblib.dump(clf, "prognosis_model.joblib")
print("Model saved as prognosis_model.joblib")
