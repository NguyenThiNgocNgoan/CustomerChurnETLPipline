import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path
import io

#IMPORT CÁC MODULE
BASE_DIR = Path(__file__).resolve().parent
try:
    from database.connection import get_engine
except ImportError:
    get_engine = None

from etl.extract import extract_data
from etl.transform import (
    rename_columns,
    remove_duplicates,
    transform_data,
    clean_total_charges,
    clean_senior_citizen,
    clean_numeric_columns,
    clean_categorical_columns
)
from etl.transform import rename_columns, remove_duplicates, transform_data
try:
    from etl_pipeline import run_etl_pipeline
except ImportError:
    try:
        from etl.etl_pipeline import run_etl_pipeline
    except ImportError:
        run_etl_pipeline = None

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
try:
    from machine_learning.preprocessing import preprocess_training_data
except ImportError:
    preprocess_training_data = None

#CẤU HÌNH GIAO DIỆN & CSS
st.set_page_config(page_title="Data Analytics Platform", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .main-title { font-size: 28px; font-weight: 700; color: #111827; margin-bottom: 8px; }
        .sub-title { font-size: 14px; color: #6b7280; margin-bottom: 25px; }
        [data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
        .sidebar-title { font-size: 18px; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 20px; }
        .sidebar-section { font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-top: 25px; margin-bottom: 10px; }
        .metric-container { display: flex; justify-content: space-between; gap: 15px; margin-bottom: 25px; }
        .metric-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; flex: 1; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
        .metric-title { font-size: 14px; color: #6b7280; font-weight: 500; margin-bottom: 10px; }
        .metric-value { font-size: 32px; font-weight: 700; color: #111827; }
        .custom-table-container { border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; background: #fff; margin-bottom: 20px; }
        .custom-table-header { background-color: #f9fafb; padding: 12px 20px; border-bottom: 1px solid #e5e7eb; font-size: 14px; font-weight: 600; color: #374151; display: flex; justify-content: space-between; }
        table.custom-table { width: 100%; border-collapse: collapse; font-size: 14px; }
        table.custom-table th, table.custom-table td { padding: 14px 20px; text-align: left; border-bottom: 1px solid #f3f4f6; color: #4b5563; }
        table.custom-table th { font-weight: 600; background-color: #f9fafb; color: #374151; }
        .badge-green { background-color: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; }
        .badge-red { background-color: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; }
        .badge-blue { background-color: #dbeafe; color: #1e40af; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 700; margin-left: 8px; }
        .insight-box { background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 16px 20px; border-radius: 4px; margin-top: 20px; margin-bottom: 20px; }
        .insight-title { color: #dc2626; font-weight: 600; font-size: 14px; margin-bottom: 6px; }
        .legend-card, .info-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; height: 100%; }
        .info-row { display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 14px; color: #4b5563; }
        .info-row-value { font-weight: 600; color: #111827; }
        button[kind="primary"] { background-color: #3b82f6 !important; color: white !important; border-radius: 8px !important; font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True
)


def page_header(title, description=""):
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="sub-title">{description}</div>', unsafe_allow_html=True)

#LẤY DỮ LIỆU TỪ MYSQL
@st.cache_data(ttl=60)
def fetch_data_from_mysql():
    if get_engine is None: return pd.DataFrame()
    try:
        engine = get_engine()
        query = """
            SELECT k.CustomerID, k.Gender, k.SeniorCitizen, k.Partner, k.Dependents,
                   h.Tenure, h.Contract, h.MonthlyCharges, h.TotalCharges, h.Churn,
                   c.PhoneService, c.MultipleLines, c.InternetService, c.OnlineSecurity,
                   c.OnlineBackup, c.DeviceProtection, c.TechSupport, c.StreamingTV, c.StreamingMovies, c.PaperlessBilling
            FROM KHACHHANG k
            JOIN HOPDONGTAICHINH h ON k.CustomerID = h.CustomerID
            JOIN CHITIETDICHVU c ON k.CustomerID = c.CustomerID
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Lỗi truy vấn MySQL: {e}")
        return pd.DataFrame()


# 4. SIDEBAR ĐIỀU HƯỚNG BÊN TRÁI MÀN HÌNH
menu = st.sidebar.radio("", ["DỮ LIỆU KHÁCH HÀNG", "PHÂN TÍCH DỮ LIỆU", "THÍ NGHIỆM MÁY HỌC"],
                        label_visibility="collapsed")
st.sidebar.divider()

# MÀN HÌNH 1: DỮ LIỆU KHÁCH HÀNG
if menu == "DỮ LIỆU KHÁCH HÀNG":
    with st.sidebar:
        st.markdown('<div class="sidebar-title">DATA ANALYTICS</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">DỮ LIỆU ĐẦU VÀO</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Kéo thả file CSV/XLSX", type=["csv", "xlsx"], label_visibility="collapsed")

        if uploaded_file is not None:
            st.session_state["saved_file_bytes"] = uploaded_file.getvalue()
            st.session_state["saved_file_name"] = uploaded_file.name
            # Reset trạng thái khi nạp file mới
            st.session_state.pop("remove_duplicates_confirmed", None)
            st.session_state.pop("custom_dropped_indices", None)
            st.session_state.pop("fill_na_method", None)
            st.session_state.pop("na_column_strategies", None)

        has_file = "saved_file_bytes" in st.session_state

        if has_file:
            st.markdown(
                f"<span style='color:green;font-weight:600;'>Đang nạp: {st.session_state['saved_file_name']}</span>",
                unsafe_allow_html=True)
        else:
            st.markdown("<small style='color:gray'>Vui lòng nạp file để bắt đầu</small>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">LÀM SẠCH DỮ LIỆU</div>', unsafe_allow_html=True)
        btn_remove_dup_click = st.button("Kiểm tra trùng lặp", use_container_width=True)
        btn_fill_na_click = st.button("Kiểm tra ô rỗng", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        btn_etl = st.button("CHẠY TOÀN BỘ ETL PIPELINE", type="primary", use_container_width=True)

        st.info("Trạng thái hệ thống\n\nQuy trình sẽ tự động trích xuất, làm sạch và nạp dữ liệu vào CSDL MySQL.")

    page_header("Tổng quan & Làm sạch Dữ liệu", "Xem trước cấu trúc dữ liệu khách hàng gốc và thực thi đường ống ETL.")

    if has_file:
        file_buffer = io.BytesIO(st.session_state["saved_file_bytes"])
        file_buffer.name = st.session_state["saved_file_name"]

        df_raw = extract_data(file_buffer)
        if df_raw is None or df_raw.empty:
            st.error("Tệp dữ liệu rỗng! Vui lòng tải lại.")
            del st.session_state["saved_file_bytes"]
            del st.session_state["saved_file_name"]
            st.rerun()

        df_display = df_raw.copy()

        if rename_columns:
            df_display = rename_columns(df_display)
        else:
            df_display = df_display.rename(columns={"customerID": "CustomerID", "gender": "Gender", "tenure": "Tenure"})

        # Ghi nhận trạng thái nút bấm ở sidebar
        if btn_remove_dup_click:
            st.session_state["show_dup_section"] = True
            st.session_state["show_na_section"] = False

        if btn_fill_na_click:
            st.session_state["show_na_section"] = True
            st.session_state["show_dup_section"] = False

        # 1. KIỂM TRA & XỬ LÝ DÒNG TRÙNG
        if st.session_state.get("show_dup_section", False):
            st.markdown("---")
            st.subheader("Kiểm tra Trùng lặp Dữ liệu")

            check_mode = st.radio(
                "Chọn tiêu chí kiểm tra:",
                ["Trùng hoàn toàn tất cả các cột", "Trùng theo Mã khách hàng (CustomerID)"],
                horizontal=True,
                key="dup_check_mode_radio"
            )

            if check_mode == "Trùng hoàn toàn tất cả các cột":
                duplicates_df = df_display[df_display.duplicated(keep=False)].copy()
                msg_warning = f"Phát hiện **{len(duplicates_df)}** dòng dữ liệu bị trùng lặp hoàn toàn!"
                msg_empty = "Thông báo: Không có dòng dữ liệu bị trùng hoàn toàn."
            else:
                if "CustomerID" in df_display.columns:
                    duplicates_df = df_display[df_display.duplicated(subset=["CustomerID"], keep=False)].copy()
                    duplicates_df = duplicates_df.sort_values(by="CustomerID")
                    msg_warning = f"Phát hiện **{len(duplicates_df)}** dòng có mã khách hàng bị trùng nhau!"
                    msg_empty = "Thông báo: Không có mã khách hàng nào bị trùng lặp."
                else:
                    duplicates_df = pd.DataFrame()
                    msg_warning = "Không tìm thấy cột CustomerID trong dữ liệu."

            if not duplicates_df.empty:
                st.warning(msg_warning)

                if "Delete" not in duplicates_df.columns:
                    duplicates_df.insert(0, "Delete", False)

                edited_duplicates = st.data_editor(
                    duplicates_df,
                    use_container_width=True,
                    key="dup_editor",
                    hide_index=True
                )

                col_y, col_n = st.columns(2)
                with col_y:
                    if st.button("Xác nhận xóa các dòng đã chọn", type="primary", use_container_width=True):
                        # Lưu trực tiếp danh sách các chỉ số index dòng mà người dùng tích chọn
                        rows_to_drop_indices = edited_duplicates[edited_duplicates["Delete"] == True].index.tolist()
                        if len(rows_to_drop_indices) > 0:
                            st.session_state["custom_dropped_indices"] = rows_to_drop_indices
                            st.session_state["remove_duplicates_confirmed"] = True
                            st.success(f"Đã chọn xóa {len(rows_to_drop_indices)} dòng được tích!")
                        else:
                            st.warning("Bạn chưa chọn dòng nào để xóa.")
                with col_n:
                    if st.button("Đóng", use_container_width=True):
                        st.session_state["custom_dropped_indices"] = []
                        st.session_state["remove_duplicates_confirmed"] = False
                        st.info("Đã đóng và giữ nguyên dữ liệu.")
            else:
                st.success(msg_empty)
            st.markdown("---")

        # 2. KIỂM TRA & XỬ LÝ Ô RỖNG (HIỂN THỊ KÈM BẢNG CHỨA CÁC DÒNG DỮ LIỆU RỖNG)
        if st.session_state.get("show_na_section", False):
            st.markdown("---")
            st.subheader("Kiểm tra & Xử lý Giá trị Rỗng Chi Tiết theo Từng Cột")

            na_counts = df_display.isna().sum()
            cols_with_na = na_counts[na_counts > 0]

            if not cols_with_na.empty:
                st.warning(f"Phát hiện **{df_display.isna().any(axis=1).sum()}** dòng chứa giá trị rỗng trên các cột sau:")

                # 1. Hiển thị bảng tổng hợp các cột bị rỗng
                df_na_summary = pd.DataFrame({
                    "Tên cột": cols_with_na.index,
                    "Số lượng ô rỗng": cols_with_na.values,
                    "Tỷ lệ rỗng (%)": (cols_with_na.values / len(df_display) * 100).round(2)
                })
                st.dataframe(df_na_summary, use_container_width=True, hide_index=True)

                # 2. HIỂN THỊ THÊM BẢNG CHỨA CÁC DÒNG DỮ LIỆU THỰC TẾ BỊ RỖNG
                st.markdown("#### Dòng dữ liệu chứa ô rỗng:")
                na_rows_df = df_display[df_display.isna().any(axis=1)]
                st.dataframe(na_rows_df, use_container_width=True)

                st.markdown("#### Thiết lập phương pháp điền:")
                column_strategies = {}

                for col in cols_with_na.index:
                    is_numeric = pd.api.types.is_numeric_dtype(df_display[col])
                    default_idx = 0 if is_numeric else 1

                    strategy = st.selectbox(
                        f"Cột: **{col}** ({cols_with_na[col]} ô trống)",
                        ["Median (Trung vị)", "Mode (Phổ biến nhất)", "Giá trị tùy chỉnh (0 / Không xác định)"],
                        index=default_idx,
                        key=f"na_strat_{col}"
                    )

                    if "Median" in strategy:
                        column_strategies[col] = "median"
                    elif "Mode" in strategy:
                        column_strategies[col] = "mode"
                    else:
                        column_strategies[col] = "custom"

                if st.button("Xác nhận phương pháp điền ô rỗng", type="primary"):
                    st.session_state["na_column_strategies"] = column_strategies
                    st.success("Đã lưu phương pháp điền ô rỗng thành công!")
            else:
                st.success("Thông báo: Dữ liệu không có ô rỗng (Missing Values).")
            st.markdown("---")

        # Lấy các thiết lập làm sạch từ session state
        remove_dup_flag = st.session_state.get("remove_duplicates_confirmed", False)
        dropped_indices = st.session_state.get("custom_dropped_indices", [])
        na_strategies = st.session_state.get("na_column_strategies", {})

        # Xử lý xem trước xóa dòng trên giao diện (bằng index)
        if remove_dup_flag and dropped_indices:
            df_display = df_display.drop(index=dropped_indices, errors='ignore')

        # Xử lý điền rỗng trên giao diện xem trước
        if na_strategies:
            for col, strat in na_strategies.items():
                if col in df_display.columns:
                    if strat == "median":
                        if pd.api.types.is_numeric_dtype(df_display[col]):
                            fill_val = df_display[col].median()
                        else:
                            fill_val = df_display[col].mode().iloc[0] if not df_display[col].mode().empty else 0
                        df_display[col] = df_display[col].fillna(fill_val)
                    elif strat == "mode":
                        fill_val = df_display[col].mode().iloc[0] if not df_display[col].mode().empty else (
                            0 if pd.api.types.is_numeric_dtype(df_display[col]) else "")
                        df_display[col] = df_display[col].fillna(fill_val)
                    else:
                        fill_val = 0 if pd.api.types.is_numeric_dtype(df_display[col]) else "Không xác định"
                        df_display[col] = df_display[col].fillna(fill_val)

        # 3. --- CHẠY ETL PIPELINE VÀ LƯU VÀO MYSQL (Sử dụng trực tiếp df_display hiện tại) ---
        if btn_etl:
            with st.spinner("Đang đồng bộ dữ liệu đã làm sạch từ giao diện và nạp vào MySQL..."):
                try:
                    # Sử dụng trực tiếp df_display (đã qua các bước xóa dòng thủ công và điền rỗng trực quan trên màn hình)
                    df_etl_run = df_display.copy()

                    # Chuẩn hóa các cột đặc thù trước khi đẩy xuống CSDL
                    df_etl_run = clean_total_charges(df_etl_run)
                    df_etl_run = clean_senior_citizen(df_etl_run)
                    df_etl_run = clean_numeric_columns(df_etl_run, fill_method="median")
                    df_etl_run = clean_categorical_columns(df_etl_run, fill_method="mode")

                    # Kết nối và gọi hàm nạp dữ liệu vào MySQL
                    engine = get_engine()
                    from etl.load import load_data

                    result_counts = load_data(df_etl_run, engine)

                    st.success("Đã hoàn thành làm sạch và nạp dữ liệu thành công vào MySQL!")
                    if isinstance(result_counts, tuple):
                        result_counts = result_counts[0]
                    for table, count in result_counts.items():
                        st.write(f"Bảng **{table}**: {count} dòng")
                except Exception as e:
                    st.error(f"Lỗi khi chạy ETL: {e}")

        total_rows = len(df_display)
        html_metrics = f"""
        <div class="metric-container">
            <div class="metric-card"><div class="metric-title">Tổng dòng</div><div class="metric-value">{total_rows:,}</div></div>
            <div class="metric-card"><div class="metric-title">Tổng cột</div><div class="metric-value">{len(df_display.columns):,}</div></div>
        </div>
        """
        st.markdown(html_metrics, unsafe_allow_html=True)

        st.markdown('<div class="custom-table-header"><span>Toàn bộ dữ liệu</span></div>', unsafe_allow_html=True)
        st.dataframe(df_display, use_container_width=True, height=500)
    else:
        st.info("Vui lòng tải tệp CSV ở thanh menu bên trái để bắt đầu trích xuất dữ liệu.")

# MÀN HÌNH 2
elif menu == "PHÂN TÍCH DỮ LIỆU":
    df_db = fetch_data_from_mysql()

    if df_db.empty:
        st.warning("Cơ sở dữ liệu MySQL đang trống. Vui lòng quay lại Màn hình 1 và chạy tiến trình ETL.")
    else:
        cols = df_db.columns.tolist()

        with st.sidebar:
            st.markdown('<div class="sidebar-title" style="color:#3b82f6;">DATA ANALYTICS</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="sidebar-section">THIẾT LẬP BIỂU ĐỒ</div>', unsafe_allow_html=True)

            # Dropdown lựa chọn các loại biểu đồ
            chart_type = st.selectbox("Loại biểu đồ", [
                "Biểu đồ Phân tán (Scatter)",
                "Biểu đồ Cột (Bar)",
                "Biểu đồ tròn (Pie Chart)",
                "Ma trận tương quan (Correlation Matrix)"
            ])

            st.markdown('<div class="sidebar-section">CẤU HÌNH TRỤC TỌA ĐỘ</div>', unsafe_allow_html=True)
            x_col = st.selectbox("Chọn trục X / Cột phân tích tỷ lệ", cols,
                                 index=cols.index("Tenure") if "Tenure" in cols else 0)
            y_col = st.selectbox("Chọn trục Y (Dùng cho Scatter/Bar)", cols,
                                 index=cols.index("MonthlyCharges") if "MonthlyCharges" in cols else 0)

            st.markdown('<div class="sidebar-section">ĐỊNH DẠNG HIỂN THỊ</div>', unsafe_allow_html=True)
            color_col = st.selectbox("Phân loại màu", cols, index=cols.index("Churn") if "Churn" in cols else 0)
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.button("LÀM MỚI BIỂU ĐỒ", type="primary", use_container_width=True)

        col_title, col_pill = st.columns([4, 1])
        with col_title:
            page_header("Phân tích Trực quan hóa Đa chiều", "Dữ liệu được lấy từ MySQL đã chuẩn hóa đồng bộ.")
        with col_pill:
            st.markdown(
                f'<div style="background:#f1f5f9; padding:6px 12px; border-radius:20px; font-size:12px; font-weight:600; text-align:center; color:#475569; float:right; margin-top:10px;">Tổng MySQL: {len(df_db):,}</div>',
                unsafe_allow_html=True)

        with st.expander("Bấm vào đây để xem toàn bộ dữ liệu đang phân tích", expanded=False):
            st.dataframe(df_db, use_container_width=True)

        # 1. Xử lý hiển thị Biểu đồ Phân tán hoặc Biểu đồ Cột
        if chart_type in ["Biểu đồ Phân tán (Scatter)", "Biểu đồ Cột (Bar)"]:
            col_chart, col_legend = st.columns([3, 1])
            with col_chart:
                df_plot = df_db.dropna(subset=[x_col, y_col, color_col]).sample(min(1000, len(df_db)), random_state=42)

                if str(color_col).lower() == "churn":
                    df_plot["Phân loại"] = df_plot[color_col].map(
                        {"Yes": "Có rời bỏ (Churn)", "No": "Không rời bỏ", 1: "Có rời bỏ (Churn)",
                         0: "Không rời bỏ"}).fillna(df_plot[color_col])
                    color_target = "Phân loại"
                    color_map = {"Có rời bỏ (Churn)": '#ef4444', "Không rời bỏ": '#3b82f6'}
                else:
                    color_target, color_map = color_col, None

                if chart_type == "Biểu đồ Phân tán (Scatter)":
                    fig = px.scatter(
                        df_plot,
                        x=x_col,
                        y=y_col,
                        color=color_target,
                        color_discrete_map=color_map,
                        opacity=0.8
                    )
                    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
                else:
                    fig = px.histogram(df_plot, x=x_col, y=y_col, color=color_target, color_discrete_map=color_map,
                                       barmode="group")

                fig.update_layout(plot_bgcolor='white', margin=dict(l=40, r=40, t=20, b=40),
                                  xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
                                  yaxis=dict(showgrid=True, gridcolor='#f1f5f9'), showlegend=False, height=450)
                st.plotly_chart(fig, use_container_width=True)

            with col_legend:
                st.markdown("""
                <div class="legend-card">
                    <div style="font-weight:700; font-size:14px; color:#334155; margin-bottom:15px;">CHÚ THÍCH</div>
                    <div style="display:flex; align-items:center; margin-bottom:10px; font-size:14px; color:#475569;"><div style="width:10px; height:10px; border-radius:50%; background-color:#ef4444; margin-right:10px;"></div>Nhóm 1 / Có rủi ro</div>
                    <div style="display:flex; align-items:center; margin-bottom:20px; font-size:14px; color:#475569;"><div style="width:10px; height:10px; border-radius:50%; background-color:#3b82f6; margin-right:10px;"></div>Nhóm 2 / An toàn</div>
                </div>
                """, unsafe_allow_html=True)

            explanation_text = f"Đang phân tích sự tác động qua lại giữa <strong>{x_col}</strong> (Trục X) và <strong>{y_col}</strong> (Trục Y)."
            if x_col == "Tenure" and y_col == "MonthlyCharges":
                explanation_text = "Biểu đồ thể hiện mối quan hệ giữa <b>thời gian gắn bó (Tenure)</b> và <b>chi phí hàng tháng (MonthlyCharges)</b>. Giúp quan sát xem khách hàng lâu năm có xu hướng trả cước cao hơn hay thấp hơn, kết hợp phân loại theo trạng thái rời bỏ (Churn)."
            elif x_col == "Tenure" and y_col == "TotalCharges":
                explanation_text = "Biểu đồ thể hiện sự tích lũy <b>tổng cước phí (TotalCharges)</b> theo <b>thời gian sử dụng (Tenure)</b>. Thông thường tổng chi phí sẽ tăng tiệm cận tuyến tính theo thời gian gắn bó của khách hàng."
            elif "Contract" in [x_col, y_col]:
                explanation_text = f"Đang so sánh sự khác biệt về phân phối giữa hình thức hợp đồng (Contract) và chỉ số {y_col if x_col == 'Contract' else x_col}, giúp đánh giá hành vi nhóm khách hàng dùng hợp đồng dài hạn so với ngắn hạn."

            st.markdown(
                f'''
                <div class="insight-box">
                    <div class="insight-title">Giải thích mối quan hệ giữa Trục X và Trục Y</div>
                    <div style="color: #1f2937; font-size: 14px; margin-bottom: 8px;">{explanation_text}</div>
                    <hr style="margin: 8px 0; border-color: #f3f4f6;">
                    <div style="color: #4b5563; font-size: 13px;">
                        • <b>Trục X ({x_col}):</b> Đại diện cho biến độc lập hoặc danh mục phân nhóm.<br>
                        • <b>Trục Y ({y_col}):</b> Đại diện cho giá trị đo lường hoặc biến phụ thuộc cần khảo sát.<br>
                        • <b>Phân loại màu ({color_col}):</b> Giúp khoanh vùng sự khác biệt rõ rệt giữa các nhóm đối tượng (ví dụ: nhóm khách hàng Rời bỏ so với Ở lại).
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

        # 2. Xử lý hiển thị Biểu đồ tròn  với giải thích tự động theo từng cột
        elif chart_type == "Biểu đồ tròn (Pie Chart)":
            st.markdown(
                f'<div class="sidebar-title" style="color:#0f172a; margin-top: 10px; margin-bottom: 10px;">TỶ LỆ PHÂN PHỐI GIÁ TRỊ: {x_col}</div>',
                unsafe_allow_html=True)

            with st.container():
                df_pie = df_db[x_col].value_counts().reset_index()
                df_pie.columns = [x_col, 'Count']

                fig_pie = px.pie(
                    df_pie,
                    names=x_col,
                    values='Count',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(
                    plot_bgcolor='white',
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=500
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                column_explanations = {
                    "CustomerID": "Mã định danh duy nhất của từng khách hàng. Phân phối đều 100% trên mỗi dòng.",
                    "gender": "Giới tính của khách hàng (Male/Female), giúp phân tích tỷ lệ nhân khẩu học.",
                    "SeniorCitizen": "Tỷ lệ khách hàng cao tuổi (1: Người lớn tuổi, 0: Khách hàng trẻ/trung niên).",
                    "Partner": "Tỷ lệ khách hàng có vợ/chồng hoặc bạn đời sống chung (Yes/No).",
                    "Dependents": "Tỷ lệ khách hàng có người phụ thuộc như con cái hoặc cha mẹ sống cùng (Yes/No).",
                    "Tenure": "Phân phối theo thời gian gắn bó (số tháng) của khách hàng sử dụng dịch vụ.",
                    "PhoneService": "Tỷ lệ khách hàng có đăng ký sử dụng dịch vụ điện thoại cố định hay không (Yes/No).",
                    "MultipleLines": "Tỷ lệ khách hàng đăng ký nhiều đường dây điện thoại cùng lúc (Yes/No/No phone service).",
                    "InternetService": "Tỷ lệ phân bổ các loại hình dịch vụ internet (Fiber optic, DSL, No internet service).",
                    "OnlineSecurity": "Tỷ lệ khách hàng sử dụng gói dịch vụ bảo mật trực tuyến bổ sung (Yes/No/No internet service).",
                    "OnlineBackup": "Tỷ lệ khách hàng sử dụng dịch vụ sao lưu dữ liệu trực tuyến (Yes/No/No internet service).",
                    "DeviceProtection": "Tỷ lệ khách hàng mua gói bảo vệ thiết bị (Yes/No/No internet service).",
                    "TechSupport": "Tỷ lệ khách hàng sử dụng gói hỗ trợ kỹ thuật trực tuyến (Yes/No/No internet service).",
                    "StreamingTV": "Tỷ lệ khách hàng xem truyền hình trực tuyến qua mạng internet (Yes/No/No internet service).",
                    "StreamingMovies": "Tỷ lệ khách hàng xem phim trực tuyến qua đường truyền internet (Yes/No/No internet service).",
                    "Contract": "Phân bổ hình thức cam kết hợp đồng (Month-to-month, One year, Two year).",
                    "PaperlessBilling": "Tỷ lệ khách hàng lựa chọn hình thức nhận hóa đơn điện tử không dùng giấy (Yes/No).",
                    "PaymentMethod": "Thống kê tỷ trọng các phương thức thanh toán cước (Electronic check, Mailed check, Bank transfer, Credit card).",
                    "MonthlyCharges": "Phân khúc mức chi phí cước dịch vụ khách hàng phải trả trung bình hàng tháng.",
                    "TotalCharges": "Phân phối tổng số tiền cước tích lũy mà khách hàng đã chi trả trong suốt thời gian sử dụng.",
                    "Churn": "Tỷ lệ % khách hàng thực tế đã rời bỏ (Yes) hoặc tiếp tục duy trì dịch vụ (No)."
                }

                default_desc = f"Biểu đồ tròn thể hiện tỷ trọng phần trăm (%) phân phối các nhóm giá trị bên trong cột dữ liệu <b>{x_col}</b>."
                current_explanation = column_explanations.get(x_col, default_desc)

                st.markdown(
                    f'''
                    <div class="insight-box" style="margin-top:0px;">
                        <div class="insight-title">Phân tích Tỷ lệ Phân phối ({x_col})</div>
                        <div style="color: #1f2937; font-size: 14px; line-height: 1.6;">
                            {current_explanation}
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

        # 3. Xử lý hiển thị Ma trận tương quan (Correlation Matrix)
        elif chart_type == "Ma trận tương quan (Correlation Matrix)":
            st.markdown(
                '<div class="sidebar-title" style="color:#0f172a; margin-top: 10px; margin-bottom: 10px;">MA TRẬN TƯƠNG QUAN (CORRELATION MATRIX)</div>',
                unsafe_allow_html=True)

            with st.container():
                if preprocess_training_data is not None:
                    try:
                        X, y, _, _, _, _ = preprocess_training_data(df_db.copy())
                        df_corr = X.copy()
                        df_corr['Churn'] = y

                        corr_matrix = df_corr.corr()

                        fig_corr = px.imshow(
                            corr_matrix,
                            text_auto=".2f",
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            labels=dict(color="Hệ số tương quan")
                        )
                        fig_corr.update_layout(
                            plot_bgcolor='white',
                            margin=dict(l=0, r=0, t=10, b=0),
                            height=650
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)

                        st.markdown(
                            '''
                            <div class="insight-box" style="margin-top:0px;">
                                <div class="insight-title">Phân tích Tương quan Chi tiết</div>
                                <div style="color: #1f2937; font-size: 14px; line-height: 1.6;">
                                    Ma trận được xây dựng dựa trên dữ liệu đã qua <b>Encoding và Scaling</b>. Giá trị càng gần <b>1</b> hoặc <b>-1</b> thể hiện sự tương quan càng mạnh. Đặc biệt chú ý đến hàng/cột <b>Churn</b> để tìm nguyên nhân rời bỏ.
                                    <br><br>
                                    <b>1. Tổng cước phí và Thời gian gắn bó (TotalCharges & Tenure):</b><br>
                                    • <i>Hệ số:</i> Tương quan thuận rất mạnh (gần 1). Khách hàng sử dụng dịch vụ càng lâu tháng (Tenure) thì tổng số tiền tích lũy cước phí họ đã thanh toán cho nhà mạng (TotalCharges) càng lớn. Đây là quy luật tuyến tính tự nhiên trong quản lý hợp đồng viễn thông.
                                    <br><br>
                                    <b>2. Thời gian gắn bó và Khả năng rời bỏ (Tenure & Churn):</b><br>
                                    • <i>Hệ số:</i> Tương quan nghịch (âm). Thời gian gắn bó càng dài thì xác suất rời bỏ dịch vụ càng thấp. Nhóm khách hàng mới sử dụng trong giai đoạn đầu là nhóm có nguy cơ cao nhất cần được chăm sóc đặc biệt.
                                    <br><br>
                                    <b>3. Hình thức hợp đồng dài hạn và Khả năng rời bỏ (Contract_Two year & Churn):</b><br>
                                    • <i>Hệ số:</i> Tương quan nghịch rõ rệt. Khách hàng ký hợp đồng dài hạn (2 năm) có xu hướng gắn bó rất bền chặt, làm giảm mạnh tỷ lệ rời bỏ so với nhóm dùng hợp đồng ngắn hạn theo tháng.
                                    <br><br>
                                    <b>4. Cước phí hàng tháng và Dịch vụ Cáp quang (MonthlyCharges & InternetService_Fiber optic):</b><br>
                                    • <i>Hệ số:</i> Tương quan thuận mạnh. Việc sử dụng gói Internet cáp quang gắn liền với mức chi trả hàng tháng cao hơn đáng kể so với các gói cước thông thường hoặc không dùng internet.
                                </div>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"Lỗi khi chạy tiền xử lý vẽ ma trận: {e}")
                else:
                    st.info("Không tìm thấy hàm tiền xử lý.")


# MÀN HÌNH 3: THÍ NGHIỆM MÁY HỌC
elif menu == "THÍ NGHIỆM MÁY HỌC":
    df_db = fetch_data_from_mysql()

    with st.sidebar:
        st.markdown('<div class="sidebar-title">THÍ NGHIỆM MÁY HỌC</div>', unsafe_allow_html=True)
        test_size_ui = st.slider("Tỷ lệ Test Size", 10, 50, 20, format="%d%%")

        st.markdown('<div class="sidebar-section">THUẬT TOÁN</div>', unsafe_allow_html=True)
        use_lr = st.checkbox("Logistic Regression", value=True)
        use_rf = st.checkbox("Random Forest", value=True)
        use_svm = st.checkbox("SVM", value=True)
        use_dt = st.checkbox("Decision Tree", value=True)

        st.markdown('<div class="sidebar-section">TÙY CHỈNH THAM SỐ MÔ HÌNH</div>', unsafe_allow_html=True)

        # Khởi tạo các biến tham số mặc định
        lr_C, lr_max_iter = 1.0, 1000
        n_estimators_ui, max_depth_rf = 100, 8
        svm_C, svm_kernel = 1.0, "rbf"
        dt_max_depth, dt_criterion = 10, "gini"

        # Chỉ hiển thị bảng tham số tương ứng nếu mô hình được tích chọn
        if use_lr:
            with st.expander("Tham số Logistic Regression"):
                lr_C = st.slider("C (Regularization)", 0.01, 10.0, 1.0, 0.1, key="lr_c_key")
                lr_max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100, key="lr_iter_key")

        if use_rf:
            with st.expander("Tham số Random Forest"):
                n_estimators_ui = st.slider("Số lượng cây (n_estimators)", 10, 200, 100, key="rf_n_key")
                max_depth_rf = st.slider("Độ sâu tối đa (max_depth)", 1, 30, 8, key="rf_depth_key")

        if use_svm:
            with st.expander("Tham số SVM"):
                svm_C = st.slider("SVM C", 0.1, 10.0, 1.0, 0.1, key="svm_c_key")
                svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="svm_kernel_key")

        if use_dt:
            with st.expander("Tham số Decision Tree"):
                dt_max_depth = st.slider("Tree Max Depth", 1, 30, 10, key="dt_depth_key")
                dt_criterion = st.selectbox("Criterion", ["gini", "entropy"], key="dt_crit_key")

        st.markdown("<br>", unsafe_allow_html=True)
        btn_train = st.button("BẮT ĐẦU HUẤN LUYỆN", type="primary", use_container_width=True)

    page_header("Kết quả Huấn luyện & Danh sách Dự báo",
                "Mô hình được huấn luyện dựa trên tập dữ liệu đã làm sạch lưu tại MySQL.")

    if df_db.empty:
        st.warning("Cơ sở dữ liệu MySQL đang trống. Vui lòng quay lại màn hình DỮ LIỆU KHÁCH HÀNG và chạy ETL Pipeline.")
    else:
        with st.expander("Bấm vào đây để xem toàn bộ dữ liệu đang dùng để huấn luyện", expanded=False):
            st.dataframe(df_db, use_container_width=True)

        if btn_train and preprocess_training_data is not None:
            with st.spinner("Đang tiền xử lý (Scaling, Encoding) và huấn luyện mô hình..."):
                try:
                    X, y, scaler, le_dict, target_encoder, features = preprocess_training_data(df_db)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ui / 100.0,
                                                                        random_state=42, stratify=y)

                    models_to_train = {}
                    if use_lr:
                        models_to_train["Logistic Regression"] = LogisticRegression(C=lr_C, max_iter=lr_max_iter,
                                                                                    random_state=42)
                    if use_rf:
                        models_to_train["Random Forest"] = RandomForestClassifier(n_estimators=n_estimators_ui,
                                                                                  max_depth=max_depth_rf,
                                                                                  random_state=42)
                    if use_svm:
                        models_to_train["SVM"] = CalibratedClassifierCV(
                            estimator=SVC(C=svm_C, kernel=svm_kernel, random_state=42), method="sigmoid", cv=5)
                    if use_dt:
                        models_to_train["Decision Tree"] = DecisionTreeClassifier(max_depth=dt_max_depth,
                                                                                  criterion=dt_criterion,
                                                                                  random_state=42)

                    results_list = []
                    trained_models = {}
                    for m_name, m_obj in models_to_train.items():
                        m_obj.fit(X_train, y_train)
                        y_pred = m_obj.predict(X_test)
                        results_list.append({
                            "Model": m_name,
                            "Accuracy": accuracy_score(y_test, y_pred),
                            "F1-score": f1_score(y_test, y_pred, zero_division=0),
                            "Precision": precision_score(y_test, y_pred, zero_division=0),
                            "Recall": recall_score(y_test, y_pred, zero_division=0)
                        })
                        trained_models[m_name] = m_obj

                    st.session_state["ui_ml_results"] = pd.DataFrame(results_list).sort_values(by="F1-score",
                                                                                               ascending=False)
                    st.session_state["ui_trained_models"] = trained_models
                    st.session_state["ui_sizes"] = {"total": len(X), "train": len(X_train), "test": len(X_test)}
                    st.session_state["ui_df_test_actual"] = df_db.iloc[X_test.index].copy()
                    st.session_state["ui_X_test"] = X_test
                    st.session_state["ui_y_test"] = y_test
                except Exception as e:
                    st.error(f"Lỗi huấn luyện: {e}")

        if "ui_ml_results" in st.session_state:
            result_df = st.session_state["ui_ml_results"]
            sizes = st.session_state["ui_sizes"]
            best_model_name = result_df.iloc[0]['Model']

            table_rows = ""
            for _, row in result_df.iterrows():
                m_name = row['Model']
                acc, f1, prec, rec = row['Accuracy'], row['F1-score'], row['Precision'], row['Recall']
                if m_name == best_model_name:
                    table_rows += f'<tr style="background-color:#f8fafc;"><td style="font-weight:600; color:#0f172a;">{m_name} <span class="badge-blue">BEST</span></td><td style="font-weight:600;">{acc:.4f}</td><td style="font-weight:600;">{f1:.4f}</td><td style="font-weight:600;">{prec:.4f}</td><td style="font-weight:600;">{rec:.4f}</td></tr>'
                else:
                    table_rows += f'<tr><td>{m_name}</td><td>{acc:.4f}</td><td>{f1:.4f}</td><td>{prec:.4f}</td><td>{rec:.4f}</td></tr>'

            st.markdown(
                f'<div class="custom-table-container"><table class="custom-table"><thead><tr><th style="width: 30%;">THUẬT TOÁN</th><th>ACCURACY</th><th>F1-SCORE</th><th>PRECISION</th><th>RECALL</th></tr></thead><tbody>{table_rows}</tbody></table></div>',
                unsafe_allow_html=True)

            col_bar, col_info = st.columns([2, 1])
            with col_bar:
                df_melted = result_df.melt(id_vars=['Model'],
                                           value_vars=['Accuracy', 'F1-score', 'Precision', 'Recall'],
                                           var_name='Metric', value_name='Score')
                fig_bar = px.bar(df_melted, x='Metric', y='Score', color='Model', barmode='group',
                                 color_discrete_sequence=['#cbd5e1', '#3b82f6', '#94a3b8', '#60a5fa'])
                fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0), height=200,
                                      legend=dict(orientation="h", y=-0.3, xanchor="center", x=0.5, title=None),
                                      xaxis=dict(title=None), yaxis=dict(title=None, showticklabels=False))
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)
            with col_info:
                st.markdown(f"""
                <div class="info-card">
                    <div style="font-weight:600; font-size:15px; margin-bottom:20px;">Thông tin Mô hình</div>
                    <div class="info-row"><span>Tổng mẫu:</span><span class="info-row-value">{sizes["total"]:,}</span></div>
                    <div class="info-row"><span>Tập Test:</span><span class="info-row-value">{sizes["test"]:,}</span></div>
                    <div class="info-row" style="margin-top:15px;"><span>Trạng thái:</span><span class="info-row-value" style="color:#10b981;">Thành công</span></div>
                </div>
                """, unsafe_allow_html=True)

            # Dự đoán thực tế và tính tỷ lệ Đúng / Sai (%)
            b_model = st.session_state["ui_trained_models"][best_model_name]
            X_test_all = st.session_state["ui_X_test"]
            y_test_all = st.session_state["ui_y_test"]
            all_preds = b_model.predict(X_test_all)

            correct_count = np.sum(all_preds == y_test_all)
            incorrect_count = len(y_test_all) - correct_count
            correct_pct = (correct_count / len(y_test_all)) * 100
            incorrect_pct = (incorrect_count / len(y_test_all)) * 100

            st.markdown(f"""
            <div style="display:flex; gap:15px; margin-top:20px; margin-bottom:15px;">
                <div class="metric-card" style="border-left: 4px solid ;"><div class="metric-title">Tỷ lệ Dự đoán ĐÚNG</div><div class="metric-value" style="color:#10b981;">{correct_pct:.2f}%</div></div>
                <div class="metric-card" style="border-left: 4px solid ;"><div class="metric-title">Tỷ lệ Dự đoán SAI</div><div class="metric-value" style="color:#ef4444;">{incorrect_pct:.2f}%</div></div>
            </div>
            """, unsafe_allow_html=True)

            df_test_actual = st.session_state["ui_df_test_actual"].head(10)
            X_test_samp = X_test_all.head(10)
            preds_samp = b_model.predict(X_test_samp)

            pred_rows = ""
            for i in range(len(preds_samp)):
                cid = df_test_actual.iloc[i].get("CustomerID", f"Sample-{i + 1}")
                actual_val = str(df_test_actual.iloc[i].get("Churn", "No")).strip().lower()
                pred_str = "Rời bỏ" if preds_samp[i] == 1 else "Ở lại"
                actual_str = "Rời bỏ" if actual_val in ["yes", "1"] else "Ở lại"
                status = "Đúng" if pred_str == actual_str else "Sai"
                badge_class = "badge-green" if status == "Đúng" else "badge-red"
                pred_rows += f"<tr><td style='color:#64748b;'>{cid}</td><td>{pred_str}</td><td>{actual_str}</td><td><span class='{badge_class}'>{status}</span></td></tr>"

            st.markdown(
                f'<br><div class="custom-table-container"><div class="custom-table-header"><span>Dự đoán Thực Tế (Tập Test mẫu)</span></div><table class="custom-table"><thead><tr><th>MÃ KH</th><th>DỰ ĐOÁN</th><th>THỰC TẾ</th><th>ĐÚNG/SAI</th></tr></thead><tbody>{pred_rows}</tbody></table></div>',
                unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("Đánh giá Chi tiết theo từng lớp ( 0 và 1).")

            from sklearn.metrics import classification_report

            # Lấy mô hình tốt nhất đang xét
            b_model = st.session_state["ui_trained_models"][best_model_name]
            X_test_all = st.session_state["ui_X_test"]
            y_test_all = st.session_state["ui_y_test"]
            y_pred_all = b_model.predict(X_test_all)

            # Tạo báo cáo phân loại dạng dictionary
            report_dict = classification_report(y_test_all, y_pred_all, target_names=['0 (Không rời bỏ)', '1 (Rời bỏ)'],
                                                output_dict=True, zero_division=0)

            # Chuyển đổi thành DataFrame
            class_report_df = pd.DataFrame(report_dict).transpose().reset_index()
            class_report_df = class_report_df.rename(
                columns={"index": "Class", "precision": "Precision", "recall": "Recall", "f1-score": "F1-Score",
                         "support": "Số lượng mẫu"})

            # Hiển thị bảng chi tiết từng lớp
            st.markdown(f"**Bảng chỉ số chi tiết theo từng lớp (Mô hình tốt nhất: `{best_model_name}`):**")
            st.dataframe(class_report_df[class_report_df['Class'].isin(['0 (Không rời bỏ)', '1 (Rời bỏ)'])],
                         use_container_width=True, hide_index=True)

            # Vẽ biểu đồ cột so sánh Precision, Recall, F1-Score giữa 2 lớp
            st.markdown(f"**Biểu đồ trực quan các chỉ số theo từng lớp:**")
            df_class_melted = class_report_df[
                class_report_df['Class'].isin(['0 (Không rời bỏ)', '1 (Rời bỏ)'])].melt(
                id_vars=['Class'],
                value_vars=['Precision', 'Recall', 'F1-Score'],
                var_name='Chỉ số',
                value_name='Giá trị'
            )

            fig_class_bar = px.bar(
                df_class_melted,
                x='Class',
                y='Giá trị',
                color='Chỉ số',
                barmode='group',
                color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b'],
                text_auto='.2f'
            )
            fig_class_bar.update_layout(
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=20, b=20),
                height=350,
                xaxis=dict(title=None),
                yaxis=dict(title="Điểm số (Score)", range=[0, 1.1])
            )
            st.plotly_chart(fig_class_bar, use_container_width=True)
        else:
            st.info("Hãy nhấn 'BẮT ĐẦU HUẤN LUYỆN' để thực thi thuật toán trên dữ liệu CSDL.")