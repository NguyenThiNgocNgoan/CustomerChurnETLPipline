import pandas as pd
from pathlib import Path

def extract_data(file_path=None):
    if file_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        file_path = base_dir / "data" / "Telco-Customer-Churn.csv"

    if hasattr(file_path, "read"):
        print("EXTRACT: Đang đọc file từ giao diện...")
        # Kiểm tra tên file để chọn hàm đọc cho đúng (CSV hay Excel)
        file_name = getattr(file_path, "name", "")
        if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    else:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {file_path}")

        print(f"Đang đọc file: {file_path}")
        if str(file_path).endswith(".xlsx") or str(file_path).endswith(".xls"):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

    print(f"Số dòng: {len(df)}")
    print(f"Số cột: {len(df.columns)}")

    return df