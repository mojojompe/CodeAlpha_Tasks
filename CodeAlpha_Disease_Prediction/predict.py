import pandas as pd
import joblib
import numpy as np

def load_model_and_scaler():
    """Loads the trained model and scaler."""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        print("Error: Model or Scaler file not found. Please run train_models.py first.")
        return None, None

def get_user_input():
    """Gets patient data from user input."""
    print("\n--- Disease Prediction System ---")
    print("Please enter patient details:")
    
    try:
        age = float(input("Age: "))
        blood_pressure = float(input("Blood Pressure (Systolic): "))
        cholesterol = float(input("Cholesterol: "))
        glucose_level = float(input("Glucose Level: "))
        fever = int(input("Fever (1 for Yes, 0 for No): "))
        cough = int(input("Cough (1 for Yes, 0 for No): "))
        fatigue = int(input("Fatigue (1 for Yes, 0 for No): "))
        
        return pd.DataFrame({
            'Age': [age],
            'Blood_Pressure': [blood_pressure],
            'Cholesterol': [cholesterol],
            'Glucose_Level': [glucose_level],
            'Fever': [fever],
            'Cough': [cough],
            'Fatigue': [fatigue]
        })
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return None

def predict_disease(model, scaler, input_data):
    """Predicts presence of disease."""
    # Scale the input features
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, 'predict_proba') else None
    
    return prediction[0], probability

if __name__ == "__main__":
    model, scaler = load_model_and_scaler()
    
    if model and scaler:
        input_data = get_user_input()
        
        if input_data is not None:
            prediction, probability = predict_disease(model, scaler, input_data)
            
            print("\n--- Prediction Result ---")
            if prediction == 1:
                print("Result: Positive (Disease Detected)")
            else:
                print("Result: Negative (Healthy)")
                
            if probability is not None:
                print(f"Probability: {probability:.2%}")
