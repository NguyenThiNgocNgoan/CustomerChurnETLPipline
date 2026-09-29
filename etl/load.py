import pandas as pd
from sqlalchemy import text
from database.connection import get_engine


def split_tables(df):
    """
    Chia dữ liệu khách hàng thành 3 bảng:
    - KHACHHANG
    - HOPDONGTAICHINH
    - CHITIETDICHVU

    GIỮ NGUYÊN TOÀN BỘ dữ liệu đã được làm sạch và xử lý thủ công từ giao diện,
    không tự động drop_duplicates để tuân thủ thao tác của người dùng.
    """

    if df is None or df.empty:
        raise ValueError("Dữ liệu đầu vào rỗng, không thể nạp vào MySQL.")

    df_clean = df.copy()

    # Chuẩn hóa mã khách hàng
    if "CustomerID" not in df_clean.columns:
        raise ValueError("Không tìm thấy cột CustomerID trong dữ liệu.")

    df_clean["CustomerID"] = (
        df_clean["CustomerID"]
        .astype(str)
        .str.strip()
    )

    # Chia dữ liệu thành 3 bảng đúng theo cấu trúc cột
    # Bảng KHACHHANG
    customer_columns = [
        "CustomerID",
        "Gender",
        "SeniorCitizen",
        "Partner",
        "Dependents"
    ]

    # Bảng HOPDONGTAICHINH (PaymentMethod nằm ở đây)
    contract_columns = [
        "CustomerID",
        "Tenure",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "PaymentMethod",  # <-- Đặt PaymentMethod ở bảng hợp đồng tài chính
        "Churn"
    ]

    # Bảng CHITIETDICHVU (Không có PaymentMethod)
    service_columns = [
        "CustomerID",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling"
    ]

    # Chỉ lấy các cột thực sự tồn tại trong dữ liệu
    customer_columns = [
        col for col in customer_columns
        if col in df_clean.columns
    ]

    contract_columns = [
        col for col in contract_columns
        if col in df_clean.columns
    ]

    service_columns = [
        col for col in service_columns
        if col in df_clean.columns
    ]

    df_customer = df_clean[customer_columns].copy()
    df_contract = df_clean[contract_columns].copy()
    df_service = df_clean[service_columns].copy()

    return df_customer, df_contract, df_service


def load_data(df, engine=None):
    if engine is None:
        engine = get_engine()

    df_customer, df_contract, df_service = split_tables(df)

    try:
        with engine.begin() as conn:
            # Xóa sạch dữ liệu cũ trong MySQL trước khi nạp dữ liệu mới đã làm sạch
            conn.execute(text("DELETE FROM CHITIETDICHVU"))
            conn.execute(text("DELETE FROM HOPDONGTAICHINH"))
            conn.execute(text("DELETE FROM KHACHHANG"))

            # Nạp dữ liệu mới
            if not df_customer.empty:
                df_customer.to_sql(name="KHACHHANG", con=conn, if_exists="append", index=False, chunksize=1000)
            if not df_contract.empty:
                df_contract.to_sql(name="HOPDONGTAICHINH", con=conn, if_exists="append", index=False, chunksize=1000)
            if not df_service.empty:
                df_service.to_sql(name="CHITIETDICHVU", con=conn, if_exists="append", index=False, chunksize=1000)

        return {
            "KHACHHANG": len(df_customer),
            "HOPDONGTAICHINH": len(df_contract),
            "CHITIETDICHVU": len(df_service)
        }
    except Exception as e:
        raise RuntimeError(f"Lỗi khi nạp dữ liệu vào MySQL: {e}")