# THIẾT KẾ VÀ XÂY DỰNG HỆ THỐNG ETL PIPELINE TỰ ĐỘNG PHỤC VỤ PHÂN TÍCH DỮ LIỆU KHÁCH HÀNG TRÊN NỀN TẢNG MYSQL 

> **Đồ Án Ngành Công Nghệ Thông Tin - Trường Đại học Mở TP.HCM**
> 
> **Sinh viên thực hiện:** Nguyễn Thị Ngọc Ngoan 
> 
> **MSSV:** 2351050114
> 
> **Giảng viên hướng dẫn:** TS. Nguyễn Tiến Đạt

---

## Giới thiệu

Hệ thống được xây dựng nhằm giải quyết bài toán phân mảnh dữ liệu tại các doanh nghiệp vừa và nhỏ (SMEs). Ứng dụng cung cấp một quy trình khép kín bao gồm 3 phân hệ chính:
1. **ETL Pipeline (Trích xuất - Biến đổi - Nạp):** Tự động hóa quá trình làm sạch dữ liệu thô (từ CSV/Excel) và nạp vào cơ sở dữ liệu MySQL đã được chuẩn hóa 3NF.
2. **Dashboard Phân tích:** Trực quan hóa dữ liệu đa chiều, giúp theo dõi trực tiếp các chỉ số kinh doanh và tỷ lệ rời bỏ (Churn) từ CSDL.
3. **Thí nghiệm Máy học (Machine Learning):** Tiền xử lý dữ liệu và huấn luyện các mô hình (Logistic Regression, Random Forest, Decision Tree, SVM) nhằm dự báo rủi ro khách hàng rời bỏ dịch vụ, hỗ trợ bộ phận CSKH ra quyết định.
---

## Công nghệ sử dụng

- **Ngôn ngữ cốt lõi:** Python 3.10+
- **Giao diện (Frontend):** Streamlit
- **Xử lý dữ liệu:** Pandas, NumPy
- **Trực quan hóa:** Plotly
- **Machine Learning:** Scikit-Learn
- **Cơ sở dữ liệu:** MySQL Server 8.0, SQLAlchemy (ORM)

---

## Cấu trúc thư mục

```text
DoAnNganh/
│
├── .venv/                      # Môi trường ảo Python (Virtual Environment)
│
├── analysis/                   # Thư mục chứa mã nguồn phân tích dữ liệu 
│   └── eda.py                  # Mã nguồn khám phá dữ liệu (EDA)
│
├── data/                       # Thư mục chứa dữ liệu thô
│   └── Telco-Customer-Churn.csv 
│
├── database/                   # Thư mục xử lý cơ sở dữ liệu
│   └── connection.py           # Thiết lập kết nối đến MySQL (SQLAlchemy)
│
├── etl/                        # Thư mục xử lý quy trình luân chuyển dữ liệu
│   ├── etl_pipeline.py         # File chạy toàn bộ luồng ETL (Extract -> Transform -> Load)
│   ├── extract.py              # Đọc tệp và trích xuất dữ liệu đầu vào
│   ├── load.py                 # Tách bảng dữ liệu 3NF và nạp vào MySQL
│   └── transform.py            # Làm sạch, ép kiểu, xử lý ô rỗng, xóa dữ liệu trùng lặp
│
├── machine_learning/           # Thư mục xử lý thuật toán Machine Learning.
│   ├── evaluate.py             # Đánh giá chỉ số mô hình (Accuracy, F1-Score, Precision, Recall)
│   ├── model_comparison.py     # Huấn luyện và so sánh các thuật toán phân lớp
│   ├── predict.py              # Xử lý dự báo trên tập dữ liệu mới
│   ├── preprocessing.py        # Tiền xử lý (One-Hot Encoding, Label Encoding, Scaling)
│   └── train.py                # Mã nguồn huấn luyện mô hình độc lập
├── models/                     # Thư mục lưu trữ tài nguyên Machine Learning đã huấn luyện
│   ├── best_model.pkl          # Trọng số của mô hình tốt nhất
│   ├── best_model_name.pkl     # Tên mô hình tốt nhất
│   ├── features.pkl            # Danh sách các đặc trưng (features) đã huấn luyện
│   ├── le_dict.pkl             # ưu trữ Label Encoders
│   ├── model_comparison.csv    # Bảng kết quả so sánh các mô hình
│   ├── scaler.pkl              # Bộ chuẩn hóa dữ liệu (MinMaxScaler)
│   └── target_encoder.pkl      # Bộ mã hóa biến mục tiêu (Churn)
│
├── app.py                      # Giao diện người dùng
├── Readme.md                   # Tài liệu hướng dẫn dự án
└── requirements.txt            # Danh sách các thư viện cần cài đặt
```
--- 

## Hướng dẫn cài đặt và chạy ứng dụng

**Bước 1:** Yêu cầu môi trường 
- Đảm bảo đã cài đặt Python (>=3.10) và MySQL Server (>=8.0)

**Bước 2:** Tạo cơ sở dữ liệu MySQL
- Mở MySQL Workbench, chạy lệnh: 
>CREATE DATABASE CustomerChurnDB;

**Bước 3:** Cấu hình kết nối MySQL

- Mở file connection.py, cập nhật các thông tin user và passwork
>DB_USER = "root"
>
>DB_PASSWORD = " " # Nhập mật khẩu MySQL của bạn
>
>DB_HOST = "127.0.0.1"
>
>DB_PORT = 3306
>
>DB_NAME = "CustomerChurnDB"

**Bước 4:** Cài đặt thư viện Python 

- Mở Terminal (góc dưới bên trái của sổ Pycharm) và chạy lệnh cài đặt môi trường: 
>pip install -r requirements.txt

**Bước 5:**  Khởi chạy giao diện 

- Mở Terminal (góc dưới bên trái của sổ Pycharm) và chạy lệnh cài đặt môi trường: 
>streamlit run app.py

**LƯU Ý:**
Đảm bảo cửa sổ Terminal đã chọn đúng môi trường ảo (.venv)