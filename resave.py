import joblib
from ml_models import RandomForest, DecisionTree
from sklearn.preprocessing import StandardScaler

# Define paths explicitly
BASE_DIR = "D:/Trinh/Machine-Learning/nhom11"
rf = joblib.load(f"{BASE_DIR}/randomforest_best.pkl")
scaler = joblib.load(f"{BASE_DIR}/scaler.pkl")

# Re-save with clean imports
joblib.dump(rf, f"{BASE_DIR}/rf_clean.pkl")
joblib.dump(scaler, f"{BASE_DIR}/scaler_clean.pkl")

print("✅ Models re-saved successfully as rf_clean.pkl and scaler_clean.pkl")
