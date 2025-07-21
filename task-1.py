# Data Pipeline for ETL using Pandas and Scikit-learn

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Extract (Load data)
def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Data loaded successfully!")
    print(df.head())
    return df

# Step 2: Transform (Preprocessing and cleaning)
def preprocess_data(df):
    # Drop missing values
    df = df.dropna()
    # Encode categorical columns (example)
    for col in df.select_dtypes(include=['object']):
        df[col] = pd.factorize(df[col])[0]
    print("Data preprocessed!")
    return df

# Step 3: Feature Scaling
def scale_features(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Features scaled!")
    return X_scaled, y

# Step 4: Train-test split (optional step if preparing for ML)
def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print("Data split into train and test sets!")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Filepath of your dataset
    filepath = "data.csv"  # Replace with your dataset path
    target_column = "target"  # Replace with your target column name

    df = load_data(filepath)
    df = preprocess_data(df)
    X_scaled, y = scale_features(df, target_column)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # Save preprocessed data (optional)
    pd.DataFrame(X_scaled).to_csv("processed_features.csv", index=False)
    pd.DataFrame(y).to_csv("processed_target.csv", index=False)

    print("Data pipeline completed successfully!")