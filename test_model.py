import joblib

# Load the model
try:
    model = joblib.load("model.pkl")
    print("✅ Model Loaded Successfully!")
    print("🔹 Model Type:", type(model))

    # Check number of expected input features
    if hasattr(model, "n_features_in_"):
        print("🔹 Expected Number of Features:", model.n_features_in_)
    else:
        print("⚠️ Error: Model does not have 'n_features_in_' attribute.")
except Exception as e:
    print("❌ Error loading model:", e)
