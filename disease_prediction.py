 #disease_prediction.py

def predict_disease(fatigue, headache, high_fever, loss_of_appetite, nausea):
    if fatigue <= 0:
        if headache <= 0:
            if high_fever > 0:
                return "AIDS"
            else:
                return "Unknown"
        elif headache > 0:
            if loss_of_appetite <= 0:
                return "Gerd"
            else:
                return "Chronic Cholestasis"
    elif fatigue > 0:
        if high_fever <= 0:
            if nausea <= 0:
                return "Diabetes"
            else:
                return "Hepatitis C"
        elif high_fever > 0:
            if headache <= 0:
                if loss_of_appetite <= 0:
                    return "Bronchial Asthma"
                else:
                    return "Hepatitis E"
            elif headache > 0:
                if nausea <= 0:
                    return "Chicken Pox"
                else:
                    return "Dengue"

import joblib

# Save the trained model
joblib.dump(model, "prognosis_model.joblib")





   
