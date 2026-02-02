import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def clean_currency(x):
    if isinstance(x, str):
        # Remove underscores and general cleanup
        x = x.replace('_', '').strip()
        # Remove any non-numeric characters except dot and minus
        x = re.sub(r'[^\d.-]', '', x)
        if not x:
            return np.nan
    return float(x)

def clean_age(x):
    if isinstance(x, str):
        x = x.replace('_', '').strip()
        x = re.sub(r'[^\d]', '', x)
        if not x:
            return np.nan
    return float(x)

def parse_credit_history_age(x):
    if isinstance(x, str):
        if 'Years' in x and 'Months' in x:
            years = int(re.search(r'(\d+) Years', x).group(1))
            months = int(re.search(r'(\d+) Months', x).group(1))
            return years * 12 + months
        elif 'NA' in x:
            return np.nan
    return np.nan

def load_and_clean_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Drop IDs and Identifiers
    cols_to_drop = ['ID', 'Customer_ID', 'Name', 'SSN']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Numeric Columns Cleaning
    numeric_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 
                    'Num_of_Loan', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 
                    'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance']
    
    print("Cleaning numeric columns...")
    for col in numeric_cols:
        if col in df.columns:
            if col == 'Age':
                 df[col] = df[col].apply(clean_age)
            else:
                 df[col] = df[col].apply(clean_currency)
    
    # Feature Engineering
    print("Feature Engineering...")
    if 'Credit_History_Age' in df.columns:
        df['Credit_History_Age'] = df['Credit_History_Age'].apply(parse_credit_history_age)
        
    # Categorical Columns Cleaning
    # Occupation, Payment_Behaviour, Credit_Mix might have underscores or empty values
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].replace('_', np.nan)
        df[col] = df[col].replace('_______', np.nan) # Seen commonly in this dataset
        
    # Handle Target
    # Standardize target labels if needed (though usually they are consistant)
    
    return df

def preprocess_data(df, is_train=True):
    print("Preprocessing data...")
    # Separate numeric and categorical
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Exclude Credit_Score from features if it exists
    if 'Credit_Score' in numeric_cols:
        numeric_cols = numeric_cols.drop('Credit_Score')
    if 'Credit_Score' in categorical_cols:
        categorical_cols = categorical_cols.drop('Credit_Score')

    # Impute Missing Values
    # Numeric -> Median
    num_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    
    # Categorical -> Most Frequent
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    # Encoding Categorical Features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    return df, numeric_cols, categorical_cols

def main():
    train_path = 'train.csv'
    
    # 1. Load and Clean
    df = load_and_clean_data(train_path)
    
    # 2. Encode Target
    target_col = 'Credit_Score'
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col])
    print(f"Target Classes: {le_target.classes_}")
    
    # 3. Preprocess Features
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_processed, num_cols, cat_cols = preprocess_data(X)
    
    # 4. Split Data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    # 5. Scale Data (Important for Logistic Regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 6. Model Training & Evaluation
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    print("\n" + "="*40)
    print("Model Evaluation Results")
    print("="*40)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le_target.classes_)
        
        # Calculate ROC AUC (Handling multiclass)
        y_prob = model.predict_proba(X_test)
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except:
            roc_auc = "N/A"
            
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC AUC: {roc_auc}")
        print("Classification Report:")
        print(report)
        
        results[name] = acc

if __name__ == "__main__":
    main()
