from sqlalchemy import create_engine

DB_USER = "root"
DB_PASSWORD = "Ntnn1009%40"
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_NAME = "CustomerChurnDB"

#Tạo kết nối đến MySQL Workbench
def get_engine():
    # Tạo chuỗi kết nối MySQL
    connection_string = (
        f"mysql+pymysql://"
        f"{DB_USER}:{DB_PASSWORD}@"
        f"{DB_HOST}:{DB_PORT}/"
        f"{DB_NAME}"
    )

    # Tạo SQLAlchemy Engine
    engine = create_engine(
        connection_string,
        pool_pre_ping=True
    )

    return engine