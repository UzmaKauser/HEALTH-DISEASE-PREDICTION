from flask import Flask, request, jsonify
import sqlite3
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Scikit-Learn model

# Load the trained model using joblib
model = joblib.load("prognosis_model.joblib")  # ✅ Change file extension to `.joblib`


# Database setup
def init_db():
    """Initialize the SQLite database and create table if not exists"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        age INTEGER NOT NULL,
                        symptoms TEXT NOT NULL,
                        prediction TEXT NOT NULL)''')
    conn.commit()
    conn.close()

@app.route("/")
def home():
    return "Disease Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from request
        data = request.get_json()
        name = data["name"]
        age = data["age"]
        symptoms = data["symptoms"]

        # Convert symptoms into a NumPy array
        symptoms_array = np.array(symptoms).reshape(1, -1)

        # Get probability predictions for all possible diseases
        prediction_proba = model.predict_proba(symptoms_array)[0]
        threshold = 0.2  # Adjust this threshold if needed

        # Select diseases with probabilities above threshold
        predicted_diseases = [
            disease for disease, prob in zip(model.classes_, prediction_proba) if prob > threshold
        ]

        # If no disease is above threshold, return the top 2 predictions
        if not predicted_diseases:
            top_indices = prediction_proba.argsort()[-2:][::-1]  # Get indices of top 2
            predicted_diseases = [model.classes_[i] for i in top_indices]

        # Store in database
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO patients (name, age, symptoms, prediction) VALUES (?, ?, ?, ?)",
            (name, age, ", ".join(map(str, symptoms)), ", ".join(predicted_diseases))
        )
        conn.commit()
        conn.close()

        return jsonify({"name": name, "age": int(age), "predictions": [str(disease) for disease in predicted_diseases]}), 200


    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/patients", methods=["GET"])
def get_patients():
    """Retrieve all stored patient records"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients")
    patients = [{"id": row[0], "name": row[1], "age": row[2], "symptoms": row[3], "prediction": row[4]}
                for row in cursor.fetchall()]
    conn.close()
    return jsonify(patients), 200

if __name__ == "__main__":
    init_db()  # Ensure database is initialized
    app.run(debug=True, host="0.0.0.0", port=5000)


import joblib
model = joblib.load("model.pkl")  # Ensure model.pkl exists
from flask_cors import CORS
app = Flask(__name__)
CORS(app)



    


                       
        
   
