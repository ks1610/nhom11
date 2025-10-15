# import joblib
# from ml_models import RandomForest, DecisionTree
# from sklearn.preprocessing import StandardScaler

# # Load your old model
# rf = joblib.load("randomforest_best.pkl")
# scaler = joblib.load("scaler.pkl")

# # Re-save with clean imports
# joblib.dump(rf, "rf_clean.pkl")
# joblib.dump(scaler, "scaler_clean.pkl")

import joblib
rf = joblib.load("rf_clean.pkl")
print(type(rf))
print("Has predict:", hasattr(rf, "predict"))
print("Has predict_proba:", hasattr(rf, "predict_proba"))
