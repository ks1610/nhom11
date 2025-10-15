import joblib
scaler = joblib.load("scaler.pkl")
print(type(scaler))
print(scaler.__module__)
