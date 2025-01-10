import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def train_model(data, target_column, save_path):
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data. Available columns: {data.columns.tolist()}")

    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, save_path)
    return model, X_test, y_test

if __name__ == "__main__":
    data = pd.read_csv('data/exoplanet_processed.csv')
    print(data.columns)  # Debug column names
    print(data.head())   # Debug data snippet
    model, X_test, y_test = train_model(data, 'habitable', '../src/model.pkl')
