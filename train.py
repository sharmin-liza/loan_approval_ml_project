import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Best Model-RF Ensemble
data = pd.read_csv("loan_approval_dataset.csv")
print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")

data.columns = data.columns.str.strip()

X_train = data.drop(['loan_status', 'loan_id'], axis=1)
y_train = data['loan_status']

le_education = LabelEncoder()
le_self_employed = LabelEncoder()
le_status = LabelEncoder()

X_train['education'] = le_education.fit_transform(X_train['education'].str.strip())
X_train['self_employed'] = le_self_employed.fit_transform(X_train['self_employed'].str.strip())
y_train = le_status.fit_transform(y_train.str.strip())

print(f"Features shape: {X_train.shape}")
print(f"Training model...")

model = RandomForestClassifier(max_depth=8, min_samples_split=5, n_estimators=300, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "loan_approval_model.pkl")
print("✓ Model successfully trained and saved to loan_approval_model.pkl")
print(f"Training accuracy: {model.score(X_train, y_train):.4f}")
