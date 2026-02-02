import pandas as pd
import os

files = ['test.csv', 'train.csv']

for f in files:
    print(f"--- Inspecting {f} ---")
    try:
        df = pd.read_csv(f)
        print("Shape:", df.shape)
        print("\nInfo:")
        print(df.info())
        print("\nDescribe:")
        print(df.describe(include='all'))
        print("\nMissing Values:")
        print(df.isnull().sum())
        if 'Credit_Score' in df.columns:
             print("\nTarget Distribution:")
             print(df['Credit_Score'].value_counts())
    except Exception as e:
        print(f"Error reading {f}: {e}")
    print("\n" + "="*30 + "\n")
