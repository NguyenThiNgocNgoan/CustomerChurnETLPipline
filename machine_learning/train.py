
# Chạy toàn bộ quá trình Machine Learning
#
# 1. Đọc dữ liệu
# 2. Tiền xử lý
# 3. Chia Train/Test
# 4. Huấn luyện 4 mô hình
# 5. Đánh giá
# 6. So sánh
# 7. Chọn mô hình tốt nhất
# 8. Lưu model

import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from machine_learning.preprocessing import preprocess_training_data
from machine_learning.model_comparison import compare_models,save_best_model


BASE_DIR = (Path(__file__).resolve().parent.parent)
DATA_PATH = (BASE_DIR/ "data"/ "Telco-Customer-Churn.csv")
MODEL_DIR = (BASE_DIR/ "models")

def train_model():
    print("=== TRAINING - MACHINE LEARNING ===")

    df = pd.read_csv(DATA_PATH)
    print(f"Số dòng dữ liệu: {len(df)}")
    print(f"Số cột dữ liệu: {len(df.columns)}")

    print("\nTiền xử lý dữ liệu.")

    (
        X,
        y,
        scaler,
        le_dict,
        target_encoder,
        features
    ) = preprocess_training_data(df)

    print(f"Số lượng features: {len(features)}")

    print("\nChia dữ liệu Train/Test...")
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    print(f"Training: {len(X_train)} dòng")
    print(f"Testing : {len(X_test)} dòng")

    #So sánh mô hình
    (
        results_df,
        trained_models,
        best_model_name,
        best_model
    ) = compare_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

#Lưu các thông tin
    MODEL_DIR.mkdir(exist_ok=True) #tạo thư mục models

    save_best_model( best_model,best_model_name)

    joblib.dump(features,MODEL_DIR / "features.pkl")
    joblib.dump(scaler,MODEL_DIR / "scaler.pkl")
    joblib.dump(le_dict,MODEL_DIR / "le_dict.pkl")
    joblib.dump(target_encoder,MODEL_DIR / "target_encoder.pkl")
    results_df.to_csv(MODEL_DIR / "model_comparison.csv",index=False)

    print(f"\nMô hình được lựa chọn: {best_model_name}")
    print("\nCác file đã lưu:")
    print("- best_model.pkl")
    print("- best_model_name.pkl")
    print("- features.pkl")
    print("- scaler.pkl" )
    print("- le_dict.pkl")
    print("- target_encoder.pkl")
    print("- model_comparison.csv")

if __name__ == "__main__":

    train_model()