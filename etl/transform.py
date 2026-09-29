import pandas as pd


def rename_columns(df):
    rename_map = {
        "customerID": "CustomerID",
        "gender": "Gender",
        "tenure": "Tenure"
    }
    df = df.rename(columns=rename_map)
    return df


def remove_duplicates(df, apply_removal=True):
    # Kiểm tra trùng lặp trên TOÀN BỘ các cột của dòng
    duplicate_rows = df[df.duplicated(keep=False)]
    print(f"Số dòng trùng lặp hoàn toàn: {len(duplicate_rows)}")

    if apply_removal:
        df = df.drop_duplicates(keep='first')
        print(f"Số dòng dữ liệu sau khi xoá trùng lặp: {len(df)}")
    return df


def clean_total_charges(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def clean_senior_citizen(df):
    df["SeniorCitizen"] = (
        df["SeniorCitizen"]
        .astype(str)
        .replace({
            "1": "Yes",
            "0": "No"
        })
    )
    return df


def clean_numeric_columns(df, fill_method="median"):
    numeric_columns = ["Tenure", "MonthlyCharges"]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df[column] = df[column].apply(lambda x: abs(x) if pd.notnull(x) else x)

            if fill_method == "median":
                fill_val = df[column].median()
            else:
                fill_val = df[column].mode().iloc[0] if not df[column].mode().empty else 0

            df[column] = df[column].fillna(fill_val)
    return df


def clean_categorical_columns(df, fill_method="mode"):
    categorical_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in categorical_columns:
        if fill_method == "mode":
            mode_values = df[column].mode()
            if not mode_values.empty:
                df[column] = df[column].fillna(mode_values.iloc[0])
        elif fill_method == "median":
            try:
                numeric_series = pd.to_numeric(df[column], errors="coerce")
                fill_val = numeric_series.median()
                if pd.isna(fill_val):
                    fill_val = df[column].mode().iloc[0]
            except Exception:
                fill_val = df[column].mode().iloc[0] if not df[column].mode().empty else ""
            df[column] = df[column].fillna(fill_val)
    return df


def transform_data(df, remove_dup=True, na_method="median"):
    print("TRANSFORM")
    df = rename_columns(df)
    df = remove_duplicates(df, apply_removal=remove_dup)
    df = clean_total_charges(df)
    df = clean_senior_citizen(df)
    df = clean_numeric_columns(df, fill_method=na_method)
    df = clean_categorical_columns(df, fill_method=na_method)

    print("Đã hoàn thành quá trình làm sạch dữ liệu.")
    return df