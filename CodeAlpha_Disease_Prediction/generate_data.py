import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples=1000):
    """Generates a synthetic medical dataset for disease prediction."""
    np.random.seed(42)

    # Features
    age = np.random.randint(18, 90, num_samples)
    blood_pressure = np.random.randint(90, 180, num_samples) # Systolic
    cholesterol = np.random.randint(150, 300, num_samples)
    glucose_level = np.random.randint(70, 200, num_samples)
    
    # Random symptoms (0: No, 1: Yes)
    symptom_fever = np.random.randint(0, 2, num_samples)
    symptom_cough = np.random.randint(0, 2, num_samples)
    symptom_fatigue = np.random.randint(0, 2, num_samples)
    
    # Target variable generation (simplified logic for correlation)
    # Higher age, BP, cholesterol, glucose increase risk
    risk_score = (
        (age / 90) * 0.3 + 
        (blood_pressure / 180) * 0.2 + 
        (cholesterol / 300) * 0.2 + 
        (glucose_level / 200) * 0.2 +
        (symptom_fever * 0.05) +
        (symptom_fatigue * 0.05)
    )
    
    # Add some noise
    risk_score += np.random.normal(0, 0.05, num_samples)
    
    # Threshold for disease presence
    disease_present = (risk_score > 0.65).astype(int)

    data = pd.DataFrame({
        'Age': age,
        'Blood_Pressure': blood_pressure,
        'Cholesterol': cholesterol,
        'Glucose_Level': glucose_level,
        'Fever': symptom_fever,
        'Cough': symptom_cough,
        'Fatigue': symptom_fatigue,
        'Disease_Present': disease_present
    })

    return data

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    output_file = 'disease_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    print(df.head())
    print("\nClass distribution:")
    print(df['Disease_Present'].value_counts())
