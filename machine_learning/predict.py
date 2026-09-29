import pandas as pd
from pathlib import Path
import joblib
from machine_learning.preprocessing import preprocess_prediction_data

BASE_DIR = (Path(__file__).resolve().parent.parent)
MODEL_DIR = (BASE_DIR / "models")

def load_model():

    model = joblib.load(MODEL_DIR / "best_model.pkl")

    features = joblib.load(MODEL_DIR / "features.pkl")

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    le_dict = joblib.load(MODEL_DIR / "le_dict.pkl")

    return (
        model,
        features,
        scaler,
        le_dict
    )

# Dự đoán khả năng khách hàng rời bỏ
def predict_churn(df):
    (model,features,scaler,le_dict) = load_model()

    df_process, customer_ids = (
        preprocess_prediction_data(
            df,
            scaler,
            le_dict,
            features
        )
    )
    prediction = model.predict(df_process)

    if hasattr( model,"predict_proba"):
        probabilities = (model.predict_proba(df_process)[:, 1])
    else:
        probabilities = prediction.astype(float)

    result_df = pd.DataFrame({
        "CustomerID":customer_ids,
        "Churn Probability":(probabilities * 100).round(2),
        "Prediction":prediction})

#Phân loại mức rủi ro
    def classify_risk(probability):
        if probability >= 70:
            return "Cao"
        elif probability >= 50:
            return "Trung bình"
        else:
            return "Thấp"
    result_df["Risk Level"] = result_df["Churn Probability"].apply(classify_risk)
    return result_df