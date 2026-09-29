from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (BASE_DIR/ "data"/ "Telco-Customer-Churn.csv")

def load_eda_data():
    df = pd.read_csv(DATA_PATH)
    return df

def dataset_info(df):
    print("THÔNG TIN DATASET")

    print("\nKích thước:")
    print(df.shape)

    print("\nDanh sách cột:")
    print(df.columns.tolist())

    print("\nThông tin kiểu dữ liệu:")
    print(df.info())

    print("\nSố lượng missing:")
    print(
        df.isnull().sum()
    )

# THỐNG KÊ MÔ TẢ
def statistics(df):

    print("\n")
    print("THỐNG KÊ MÔ TẢ")

    print(df.describe(include="all"))

# TỶ LỆ CHURN
def churn_statistics(df):

    print("\n")
    print("THỐNG KÊ CHURN")
    churn_count = (df["Churn"].value_counts())
    print(churn_count)

    churn_rate = (df["Churn"].value_counts(normalize=True) * 10)
    print("\nTỷ lệ %:")
    print( churn_rate.round(2))

# BIỂU ĐỒ CHURN
def plot_churn(df):

    churn_count = (df["Churn"].value_counts())

    churn_count.plot(
        kind="bar",
        title="Phân bố khách hàng rời bỏ"
    )
    plt.xlabel("Churn")
    plt.ylabel("Số lượng khách hàng")
    plt.tight_layout()
    plt.show()

def run_eda_analysis():

    df = load_eda_data()

    dataset_info(df)

    statistics(df)

    churn_statistics(df)

    plot_churn(df)


if __name__ == "__main__":
    run_eda_analysis()