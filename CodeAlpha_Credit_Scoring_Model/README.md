# Credit Scoring Model

## Overview
This project predicts individual creditworthiness using financial history data. It implements a machine learning pipeline that cleans the data, performs feature engineering, and trains classification models to categorize credit scores as **Good**, **Standard**, or **Poor**.

## Features
- **Data Preprocessing**: Handles missing values, cleanses currency/age strings, and robustly parses special characters.
- **Feature Engineering**: Derives useful features like `Credit_History_Age` in months.
- **Modeling**: Implements and compares:
  - Logistic Regression (Baseline)
  - Decision Tree Classifier
  - Random Forest Classifier
- **Evaluation**: Uses Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.

## Dataset
The project uses `train.csv` for training and evaluation.
- **Target Variable**: `Credit_Score` (Multiclass: Good, Standard, Poor)
- **Features**: Mix of numerical (e.g., Annual Income, Outstanding Debt) and categorical (e.g., Payment Behaviour, Occupation).

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn

Install dependencies via pip:
```bash
pip install pandas numpy scikit-learn
```

## Usage
To run the model training and evaluation script:

```bash
python credit_scoring_model.py
```

The script will:
1. Load `train.csv`.
2. Clean and preprocess the data.
3. Train the models.
4. Output the classification reports and accuracy scores to the console.

## Model Performance
The **Random Forest Classifier** achieved the best performance on the test set:

| Model | Accuracy | ROC AUC |
|-------|----------|---------|
| **Random Forest** | **~79%** | **0.91** |
| Decision Tree | ~68% | 0.74 |
| Logistic Regression | ~58% | N/A |

### Detailed Report (Random Forest)
```
              precision    recall  f1-score   support

        Good       0.74      0.71      0.72      3527
        Poor       0.79      0.80      0.79      5874
    Standard       0.80      0.81      0.81     10599

    accuracy                           0.79     20000
```


Copyright © 2026 Emmanuel Jompe
All rights reserved.