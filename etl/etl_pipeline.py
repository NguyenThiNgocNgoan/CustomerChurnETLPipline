from database.connection import get_engine
from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data


# extract - transform - kết nối db - load - kết quả mysql
def run_etl_pipeline(file_path=None, remove_dup=True, na_method="median"):
    print("\nBẮT ĐẦU ETL PIPELINE")

    df = extract_data(file_path)
    df_clean = transform_data(df, remove_dup=remove_dup, na_method=na_method)

    engine = get_engine()
    result = load_data(df_clean, engine)

    print("ETL KẾT QUẢ:")
    for table, count in result.items():
        print(f"{table}: {count} dòng")

    return result, df_clean


if __name__ == "__main__":
    run_etl_pipeline()