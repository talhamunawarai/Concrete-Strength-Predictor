import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "Concrete_Data.xls"
MODEL_FILE = "model.json"

df = pd.read_excel(DATA_FILE)

# The first 8 columns are inputs; the last column is compressive strength.
X = df.iloc[:, :8]
y = df.iloc[:, 8]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

model_data = {
    "model_type": "LinearRegression",
    "target": "Concrete compressive strength (MPa)",
    "features": [
        "Cement",
        "Blast Furnace Slag",
        "Fly Ash",
        "Water",
        "Superplasticizer",
        "Coarse Aggregate",
        "Fine Aggregate",
        "Age"
    ],
    "coefficients": model.coef_.tolist(),
    "intercept": float(model.intercept_),
    "test_metrics": {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions)))
    },
    "training": {
        "test_size": 0.20,
        "random_state": 42,
        "rows": int(len(df))
    }
}

with open(MODEL_FILE, "w", encoding="utf-8") as f:
    json.dump(model_data, f, indent=4)

print("Model trained successfully.")
print(f"R²:   {model_data['test_metrics']['r2']:.4f}")
print(f"MAE:  {model_data['test_metrics']['mae']:.4f}")
print(f"RMSE: {model_data['test_metrics']['rmse']:.4f}")
print(f"Saved: {MODEL_FILE}")
