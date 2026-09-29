# # Huấn luyện và so sánh 4 mô hình:
# # 1. Logistic Regression
# # 2. Decision Tree
# # 3. Random Forest
# # 4. SVM
# #Trả về: results_df, trained_models, best_model_name, best_model
#
# import pandas as pd
# import joblib
# from pathlib import Path
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.calibration import CalibratedClassifierCV
# from machine_learning.evaluate import evaluate_model
#
# #Lưu model
# BASE_DIR = Path(__file__).resolve().parent.parent
# MODEL_DIR = BASE_DIR / "models"
#
# #tạo mô hình
# def create_models():
#
#     models = {
#         "Logistic Regression":
#             LogisticRegression(
#                 max_iter=1000,
#                 random_state=42
#             ),
#
#         "Decision Tree":
#             DecisionTreeClassifier(
#                 max_depth=10,
#                 random_state=42
#             ),
#
#         "Random Forest":
#             RandomForestClassifier(
#                 n_estimators=100,
#                 max_depth=10,
#                 random_state=42
#             ),
#
#         "SVM":
#             CalibratedClassifierCV(
#                 estimator=SVC(kernel="rbf",random_state=42),
#                 method="sigmoid",
#                 cv=5
#             )
#     }
#
#     return models
#
# #Huấn luyện và so sánh
# def compare_models(X_train, X_test, y_train, y_test):
#     print(f"\nSo sánh 4 mô hình Machine Learning.")
#
# #Tạo mô hình
#     models = create_models()
#
#     results = []
#
#     trained_models = {}
#
#     # Huấn luyện từng mô hình
#
#     for model_name, model in models.items():
#         print(
#             f"Đang huấn luyện: {model_name}"
#         )
#
#         model.fit(X_train,y_train)
#
#         # Lưu model đã train vào dictionary
#         trained_models[model_name] = model
#
#         # Đánh giá
#         result = evaluate_model(
#             model,
#             X_test,
#             y_test,
#             model_name
#         )
#
#         results.append(result)
#
#     # Chuyển kết quả thành DataFrame
#
#     results_df = pd.DataFrame(results)
#
#     # F1-score giảm dần, được dùng làm tiêu chí chính để lựa chọn mô hình.
#     results_df = results_df.sort_values(by="F1-score",ascending=False).reset_index(drop=True)
#
#     # Mô hình
#     best_model_name = (results_df.iloc[0]["Model"])
#
#     best_model = trained_models[best_model_name]
#
#     # Bảng so sánh
#     print("Kết quả so sánh.")
#     print(results_df.to_string(index=False))
#     print(f"MÔ HÌNH TỐT NHẤT: {best_model_name}")
#     print(f"F1-score: {results_df.iloc[0]['F1-score']:.4f}")
#
#     return (
#         results_df,
#         trained_models,
#         best_model_name,
#         best_model
#     )
#
# # Lưu mô hình tốt nhất
# def save_best_model(best_model,best_model_name):
#     # Tạo thư mục models nếu chưa tồn tại
#     MODEL_DIR.mkdir(exist_ok=True)
#
#     model_path = (MODEL_DIR / "best_model.pkl")
#     joblib.dump(best_model,model_path)
#
#     model_name_path = (MODEL_DIR / "best_model_name.pkl")
#     joblib.dump(best_model_name,model_name_path)
#
#     print("Lưu mô hình.")
#     print(f"Mô hình: {best_model_name}")
#     print(f"File: {model_path}")
#
#     return model_path

# Huấn luyện và so sánh 4 mô hình với đầy đủ tham số tùy chỉnh:
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# 4. SVM
# Trả về: results_df, trained_models, best_model_name, best_model

import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from machine_learning.evaluate import evaluate_model

# Lưu model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Tạo mô hình với đầy đủ tham số truyền vào
def create_models(
    lr_C=1.0,
    lr_max_iter=1000,
    n_estimators=100,
    max_depth_rf=8,
    svm_C=1.0,
    svm_kernel="rbf",
    dt_max_depth=10,
    dt_criterion="gini"
):
    models = {
        "Logistic Regression":
            LogisticRegression(
                C=lr_C,
                max_iter=lr_max_iter,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=dt_max_depth,
                criterion=dt_criterion,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth_rf,
                random_state=42
            ),

        "SVM":
            CalibratedClassifierCV(
                estimator=SVC(
                    C=svm_C,
                    kernel=svm_kernel,
                    random_state=42
                ),
                method="sigmoid",
                cv=5
            )
    }

    return models

# Huấn luyện và so sánh nhận tham số tùy chỉnh
def compare_models(
    X_train, X_test, y_train, y_test,
    lr_C=1.0,
    lr_max_iter=1000,
    n_estimators=100,
    max_depth_rf=8,
    svm_C=1.0,
    svm_kernel="rbf",
    dt_max_depth=10,
    dt_criterion="gini"
):
    print(f"\nSo sánh 4 mô hình Machine Learning với tham số tùy chỉnh.")

    # Tạo mô hình với các tham số đã chọn
    models = create_models(
        lr_C=lr_C,
        lr_max_iter=lr_max_iter,
        n_estimators=n_estimators,
        max_depth_rf=max_depth_rf,
        svm_C=svm_C,
        svm_kernel=svm_kernel,
        dt_max_depth=dt_max_depth,
        dt_criterion=dt_criterion
    )

    results = []
    trained_models = {}

    # Huấn luyện từng mô hình
    for model_name, model in models.items():
        print(f"Đang huấn luyện: {model_name}")

        model.fit(X_train, y_train)

        # Lưu model đã train vào dictionary
        trained_models[model_name] = model

        # Đánh giá
        result = evaluate_model(
            model,
            X_test,
            y_test,
            model_name
        )

        results.append(result)

    # Chuyển kết quả thành DataFrame
    results_df = pd.DataFrame(results)

    # F1-score giảm dần, được dùng làm tiêu chí chính để lựa chọn mô hình.
    results_df = results_df.sort_values(by="F1-score", ascending=False).reset_index(drop=True)

    # Mô hình tốt nhất
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    # Bảng so sánh
    print("Kết quả so sánh.")
    print(results_df.to_string(index=False))
    print(f"MÔ HÌNH TỐT NHẤT: {best_model_name}")
    print(f"F1-score: {results_df.iloc[0]['F1-score']:.4f}")

    return (
        results_df,
        trained_models,
        best_model_name,
        best_model
    )

# Lưu mô hình tốt nhất
def save_best_model(best_model, best_model_name):
    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "best_model.pkl"
    joblib.dump(best_model, model_path)

    model_name_path = MODEL_DIR / "best_model_name.pkl"
    joblib.dump(best_model_name, model_name_path)

    print("Lưu mô hình.")
    print(f"Mô hình: {best_model_name}")
    print(f"File: {model_path}")

    return model_path