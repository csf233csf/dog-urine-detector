import sqlite3
import logging
import time
import os

# 配置日志记录
LOG_FILE = 'db_monitor.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=LOG_FILE)

def purge_log(file_path, max_size=5*1024*1024):
    """清理日志文件，如果超过指定大小则清空"""
    if os.path.exists(file_path) and os.path.getsize(file_path) > max_size:
        with open(file_path, 'w'):
            pass

def list_tables(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return cursor.fetchall()

def fetch_data_from_table(cursor, table_name):
    cursor.execute(f"SELECT * FROM {table_name}")
    return cursor.fetchall()

def monitor_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    logging.info(f"Connecting to database: {db_path}")

    tables = list_tables(cursor)
    if not tables:
        logging.warning(f"No tables found in the database {db_path}.")
        return

    for table in tables:
        table_name = table[0]
        logging.info(f"Table: {table_name}")
        rows = fetch_data_from_table(cursor, table_name)
        for row in rows:
            logging.info(row)
        logging.info("\n")

    conn.close()

def main():
    db_paths = ['instance/sensor_data.db', 'instance/water_sensor.db']
    for db_path in db_paths:
        monitor_db(db_path)

if __name__ == "__main__":
    while True:
        purge_log(LOG_FILE)
        main()
        time.sleep(5)
        logging.info('=' * 50)
