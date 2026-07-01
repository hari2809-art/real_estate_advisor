import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):

    # Remove duplicates
    df = df.drop_duplicates()

    # Fix warning
    df = df.ffill()

    # Feature Engineering
    df['Price_per_SqFt'] = df['Price_in_Lakhs'] * 100000 / df['Size_in_SqFt']
    df['Age_of_Property'] = 2025 - df['Year_Built']

    # 🔥 Encode ALL categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns

    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    return df, encoders