import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from src.preprocess import preprocess_data

# load and preprocess
df = preprocess_data("../data/Walmart.csv")

# features and target
X = df.drop(columns=['Weekly_Sales'])
y = df['Weekly_Sales']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestRegressor(n_estimators=100)

model.fit(X_train, y_train)

# save model
joblib.dump(model, "model.pkl")

print("Model trained successfully")