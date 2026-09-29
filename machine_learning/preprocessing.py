import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#Cột nhị phân
BINARY_COLUMNS = [
    "Gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling"
]

#Cột nhiều giá trị
MULTI_COLUMNS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]

#Cột số
NUMERIC_COLUMNS = [
    "Tenure",
    "MonthlyCharges",
    "TotalCharges"
]

def clean_ml_data(df):
    df = df.copy()
    rename_map = {
        "customerID": "CustomerID",
        "gender": "Gender",
        "tenure": "Tenure"
    }
    df = df.rename(columns=rename_map)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
    df["TotalCharges"] = (df["TotalCharges"].fillna(0))

    return df

def preprocess_training_data(df):
    print(f"\nTIỀN XỬ LÝ DỮ LIỆU TRAINING.")

    df = clean_ml_data(df)

    df = df.drop(columns=["CustomerID"])

    # Cột nhị phân
    le_dict = {}
    for col in BINARY_COLUMNS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le

    #Encoding biến mục tiêu ("Churn")
    target_encoder = LabelEncoder()
    df["Churn"] = target_encoder.fit_transform(df["Churn"].astype(str))

    #Cột nhiều giá trị
    existing_multi = [col for col in MULTI_COLUMNS if col in df.columns]
    df = pd.get_dummies(df, columns=existing_multi, drop_first=True)

    # Chuyển True = 1 ; False = 0
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # CHuẩn hoá  (Scaling)
    scaler = MinMaxScaler()
    existing_numeric = [col for col in NUMERIC_COLUMNS if col in df.columns]
    df[existing_numeric] = scaler.fit_transform(df[existing_numeric])

    #Tách X, y VÀ Lưu các features
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    features = list(X.columns)

    print(f"Số lượng features tạo ra: {len(features)}")
    print(f"Số lượng mẫu huấn luyện: {len(X)}")

    return X, y, scaler, le_dict, target_encoder, features


def preprocess_prediction_data(df, scaler, le_dict, features):
    df_process = clean_ml_data(df)

    # Lưu và xóa CustomerID
    customer_ids = None
    if "CustomerID" in df_process.columns:
        customer_ids = df_process["CustomerID"].copy()
        df_process = df_process.drop(columns=["CustomerID"])

    # Bỏ cột Churn nếu nó nằm trong dữ liệu dự đoán
    if "Churn" in df_process.columns:
        df_process = df_process.drop(columns=["Churn"])

    for col in BINARY_COLUMNS:
        if col in df_process.columns and col in le_dict:
            encoder = le_dict[col]
            values = df_process[col].astype(str)

            # Nếu gặp giá trị chưa train, gán bằng giá trị đầu tiên trong classes_
            known_values = set(encoder.classes_)
            values = values.apply(lambda x: x if x in known_values else encoder.classes_[0])

            df_process[col] = encoder.transform(values)

#One - hot encoding
    existing_multi = [col for col in MULTI_COLUMNS if col in df_process.columns]
    df_process = pd.get_dummies(df_process, columns=existing_multi, drop_first=True)

    for col in df_process.columns:
        if df_process[col].dtype == bool:
            df_process[col] = df_process[col].astype(int)

# Đảm bảo đúng các features(số lượng, thứ tự)
# Cột thiếu điền 0. Cột nào dư (do label lạ tạo ra) sẽ tự động bị loại bỏ.
    df_process = df_process.reindex(columns=features, fill_value=0)

#Scaling
    existing_numeric = [col for col in NUMERIC_COLUMNS if col in df_process.columns]
    if existing_numeric:
        df_process[existing_numeric] = scaler.transform(df_process[existing_numeric])

    return df_process, customer_ids