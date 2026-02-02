# Disease Prediction System

## Overview
This project applies classification techniques (Logistic Regression, SVM, Random Forest, XGBoost) to predict the presence of a disease based on patient health data.

## Features
- **Synthetic Data Generation**: Creates a realistic medical dataset (`generate_data.py`).
- **Model Training**: Trains and evaluates multiple algorithms (`train_models.py`).
- **Prediction Interface**: CLI tool to predict disease for new patients (`predict.py`).
- **Visualization**: Generates performance comparison plots (`model_comparison.png`).

## Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data**:
   ```bash
   python generate_data.py
   ```

3. **Train Models**:
   ```bash
   python train_models.py
   ```
   This will save the best model to `best_model.pkl` and a scaler to `scaler.pkl`.

## Usage
To make a prediction for a new patient, run:
```bash
python predict.py
```
Follow the prompts to enter patient details (Age, BP, Cholesterol, etc.).

## Models Implemented
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

## Results
The performance of each model is printed to the console during training, and a comparison chart `model_comparison.png` is generated.


Copyright © 2026 Emmanuel Jompe
All rights reserved.
