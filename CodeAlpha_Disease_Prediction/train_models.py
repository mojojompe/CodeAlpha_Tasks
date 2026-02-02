import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

def load_data(filepath):
    """Loads the dataset."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocesses the data: scales features and splits into train/test."""
    X = df.drop('Disease_Present', axis=1)
    y = df['Disease_Present']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, y_train):
    """Trains multiple models and returns them."""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluates models and returns a performance dataframe."""
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
        
        print(f"\n--- {name} Results ---")
        print(classification_report(y_test, y_pred))
        
    return pd.DataFrame(results)

def plot_results(results_df):
    """Plots model performance comparison."""
    plt.figure(figsize=(10, 6))
    melted_results = pd.melt(results_df, id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(x="Model", y="Score", hue="Metric", data=melted_results)
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.savefig('model_comparison.png')
    print("Performance plot saved as 'model_comparison.png'")

if __name__ == "__main__":
    # Load and Preprocess
    print("Loading data...")
    df = load_data('disease_data.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train
    models = train_models(X_train, y_train)
    
    # Evaluate
    results_df = evaluate_models(models, X_test, y_test)
    print("\nSummary of Results:")
    print(results_df)
    
    # Plot
    plot_results(results_df)
    
    # Save best model (based on F1 Score)
    best_model_name = results_df.sort_values(by='F1 Score', ascending=False).iloc[0]['Model']
    best_model = models[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Best model and scaler saved.")
