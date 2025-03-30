import datetime
import os
import sqlite3

def check_and_create_database(db_file):
    """
    检查数据库文件是否存在，如果不存在则创建数据库及相关的数据表（hub_info表和mold_info表）
    Check if the database file exists. If not, create the database and related tables (hub_info and mold_info).

    :param db_file: 数据库文件的路径
    :param db_file: The path of the database file.
    :return:
    """
    # 获取数据库文件所在的目录路径
    # Get the directory path of the database file
    db_dir = os.path.dirname(db_file)

    # 如果目录路径不存在，则创建目录
    # If the directory path does not exist, create the directory
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    if not os.path.exists(db_file):
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # 创建数据库中可能需要的其他表，这里先假设没有其他表，如有需要可按下面格式添加其他表的创建语句
        # Create other tables that may be needed in the database. Here, it is assumed that there are no other tables.
        # If needed, add the creation statements of other tables in the following format.
        create_tables = [
            '''
            CREATE TABLE IF NOT EXISTS recognition_record (
                hub_id INTEGER PRIMARY KEY AUTOINCREMENT,
                recognition_time TEXT,
                mold_number TEXT
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS mold_info (
                mold_number TEXT PRIMARY KEY,
                mold_name TEXT
            )
            '''
        ]

        for create_table_sql in create_tables:
            cursor.execute(create_table_sql)

        conn.commit()
        conn.close()
        return '数据检查成功...'

def insert_recognition_record(db_file, mold_number):
    """
    向轮毂信息表（recognition_record）插入一条数据记录，其中识别时间会自动生成当前时间
    Insert a data record into the hub information table (recognition_record). The recognition time will be automatically set to the current time.

    :param db_file: 数据库文件的路径，应与check_and_create_database函数中使用的数据库文件路径一致。
    :param db_file: The path of the database file, which should be the same as the one used in the check_and_create_database function.
    :param mold_number: 要插入的模具编号，通常是通过OCR识别出来的，会同时插入到hub_info表和mold_info表（若mold_info表中不存在该编号）。
    :param mold_number: The mold number to be inserted, usually recognized by OCR. It will be inserted into both the hub_info and mold_info tables (if the number does not exist in the mold_info table).
    """
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # 先在mold_info表中添加mold_number（若不存在）
    # First, add the mold_number to the mold_info table if it does not exist.
    cursor.execute("INSERT OR IGNORE INTO mold_info (mold_number) VALUES (?)", (mold_number,))

    cursor.execute("INSERT INTO recognition_record (recognition_time, mold_number) VALUES (?,?)",
                   (current_time, mold_number))

    conn.commit()
    conn.close()

def query_recognition_record_by_mold_number(db_file, mold_number):
    """
    查询轮毂信息表（recognition_record）中的数据记录
    Query the data records in the hub information table (recognition_record).

    :param db_file: 数据库文件的路径，应与其他函数中使用的数据库文件路径一致。
    :param db_file: The path of the database file, which should be the same as the one used in other functions.
    :param mold_number: 要查询的模具编号。
    :param mold_number: The mold number to be queried.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM recognition_record WHERE mold_number=?", (mold_number,))
    columns = [description[0] for description in cursor.description]
    results =  [columns] + cursor.fetchall()
    conn.close()
    return results

def query_mold_info_by_number(db_file, mold_number):
    """
    查询模具信息表（mold_info）中的数据记录
    Query the data records in the mold information table (mold_info).

    :param db_file:  数据库文件的路径，应与其他函数中使用的数据库文件路径一致。
    :param db_file: The path of the database file, which should be the same as the one used in other functions.
    :param mold_number: 要查询的模具编号。
    :param mold_number: The mold number to be queried.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM mold_info WHERE mold_number=?", (mold_number,))
    columns = [description[0] for description in cursor.description]
    row = cursor.fetchone()
    if row is not None:
        result = [columns] + [row]  # 把结果行也转换为列表
        # Convert the result row to a list.
    else:
        result = [columns]  # 如果没有结果，仅返回列名
        # If there is no result, only return the column names.
    conn.close()
    return result

def update_recognition_record(db_file, hub_id, recognition_time=None, mold_number=None):
    """
    修改轮毂信息表（recognition_record）中的数据记录
    Update the data records in the hub information table (recognition_record).

    :param db_file: 数据库文件的路径，应与其他函数中使用的数据库文件路径一致。
    :param db_file: The path of the database file, which should be the same as the one used in other functions.
    :param hub_id:  要修改的轮毂记录的ID，通过该ID定位到要修改的具体记录。
    :param hub_id: The ID of the hub record to be updated. Use this ID to locate the specific record to be modified.
    :param recognition_time: 可选的参数，要更新的识别时间值，如果为None则不更新该字段。
    :param recognition_time: Optional parameter. The recognition time value to be updated. If it is None, this field will not be updated.
    :param mold_number: 可选的参数，要更新的模具编号值，如果为None则不更新该字段。
    :param mold_number: Optional parameter. The mold number value to be updated. If it is None, this field will not be updated.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    update_fields = []
    update_values = []

    if recognition_time is not None:
        update_fields.append("recognition_time =?")
        update_values.append(recognition_time)
    if mold_number is not None:
        update_fields.append("mold_number =?")
        update_values.append(mold_number)

    if update_fields:
        update_query = "UPDATE recognition_record SET " + ", ".join(update_fields) + " WHERE hub_id =?"
        update_values.append(hub_id)
        cursor.execute(update_query, tuple(update_values))

    conn.commit()
    conn.close()

def update_mold_info(db_file, mold_number, mold_name=None):
    """
    修改模具信息表（mold_info）中的数据记录
    Update the data records in the mold information table (mold_info).

    :param db_file: 数据库文件的路径，应与其他函数中使用的数据库文件路径一致。
    :param db_file: The path of the database file, which should be the same as the one used in other functions.
    :param mold_number: 要删除的轮毂记录的ID，通过该ID定位到要删除的具体记录。
    :param mold_number: The ID of the hub record to be deleted. Use this ID to locate the specific record to be deleted.
    :param mold_name: 要更新的模具名称值，如果为None则不更新该字段。
    :param mold_name: Optional parameter. The mold name value to be updated. If it is None, this field will not be updated.
    :type db_file: str
    :type mold_number: str
    :type mold_name: str
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    update_fields = []
    update_values = []

    if mold_name is not None:
        update_fields.append("mold_name =?")
        update_values.append(mold_name)

    if update_fields:
        update_query = "UPDATE mold_info SET " + ", ".join(update_fields) + " WHERE mold_number =?"
        update_values.append(mold_number)
        cursor.execute(update_query, tuple(update_values))

    conn.commit()
    conn.close()

def delete_recognition_record_by_id(db_file, hub_id):
    """
    根据轮毂ID删除轮毂信息表（recognition_record）中的一条数据记录
    Delete a data record from the hub information table (recognition_record) based on the hub ID.

    :param db_file: 数据库文件的路径，应与其他函数中使用的数据库文件路径一致。
    :param db_file: The path of the database file, which should be the same as the one used in other functions.
    :param hub_id: 要删除的轮毂记录的ID，通过该ID定位到要删除的具体记录。
    :param hub_id: The ID of the hub record to be deleted. Use this ID to locate the specific record to be deleted.
    :type db_file: str
    :type hub_id: str
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM recognition_record WHERE hub_id=?", (hub_id,))
    conn.commit()
    conn.close()

def delete_mold_info_by_number(db_file, mold_number):
    """
    根据模具编号删除模具信息表（mold_info）中的一条数据记录,如果不存在，则在mold_info表中执行删除操作，删除指定mold_number的记录。
    Delete a data record from the mold information table (mold_info) based on the mold number.
    If the record does not exist, perform the deletion operation in the mold_info table to delete the record with the specified mold_number.

    :param db_file: 数据库文件的路径，应与其他函数中使用的数据库文件路径一致。
    :param db_file: The path of the database file, which should be the same as the one used in other functions.
    :param mold_number: 要删除的模具记录的编号，通过该编号定位到要删除的具体记录。
    :param mold_number: The number of the mold record to be deleted. Use this number to locate the specific record to be deleted.
    :type db_file: str
    :type mold_number: str
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # 检查mold_number在hub_info表中是否存在
    # Check if the mold_number exists in the hub_info table.
    cursor.execute("SELECT COUNT(*) FROM recognition_record WHERE mold_number=?", (mold_number,))
    count = cursor.fetchone()[0]

    if count == 0:
        cursor.execute("DELETE FROM mold_info WHERE mold_number=?", (mold_number,))

    conn.commit()
    conn.close()

def query_recognition_record_by_time_range(db_file, start_time, end_time):
    """
    筛选查询指定时间范围内的轮毂信息表（recognition_record）中的数据记录
    Query the data records in the hub information table (recognition_record) within the specified time range.

    :param db_file: 数据库文件的路径，应与其他函数中使用的数据库文件路径一致。
    :param db_file: The path of the database file, which should be the same as the one used in other functions.
    :param start_time: 指定时间范围的开始时间，格式应为 'YYYY-MM-DD HH:MM:SS'
    :param start_time: The start time of the specified time range, in the format 'YYYY-MM-DD HH:MM:SS'.
    :param end_time: 指定时间范围的结束时间，格式应为 'YYYY-MM-DD HH:MM:SS'
    :param end_time: The end time of the specified time range, in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM recognition_record WHERE recognition_time BETWEEN? AND?",
                   (start_time, end_time))
    columns = [description[0] for description in cursor.description]
    results = [columns]+cursor.fetchall()
    conn.close()
    return results

def execute_custom_sql(db_file, sql_command):
    """
    执行自定义SQL指令
    Execute a custom SQL command.

    :param db_file: 数据库文件的路径，应与其他函数中使用的数据库文件路径一致。
    :param db_file: The path of the database file, which should be the same as the one used in other functions.
    :param sql_command: 要执行的自定义SQL指令
    :param sql_command: The custom SQL command to be executed.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_command)
        if sql_command.strip().lower().startswith("select"):
            columns = [description[0] for description in cursor.description]
            results = [columns]+cursor.fetchall()
            conn.close()
            return results
        else:
            conn.commit()
            conn.close()
            return "操作成功执行"
    except sqlite3.Error as e:
        conn.close()
        return f"操作失败: {e}"

def query_all_recognition_record(db_file):
    """
    查询轮毂信息表（recognition_record）中的所有数据记录，返回所有轮毂的情况。
    Query all data records in the hub information table (recognition_record) and return the information of all hubs.

    :param db_file: 数据库文件的路径，应与其他函数中使用的数据库文件路径一致。
    :param db_file: The path of the database file, which should be the same as the one used in other functions.
    :type db_file: str

    功能：
    1. 连接到数据库。
    2. 在recognition_record表中执行查询操作，获取所有记录。
    3. 获取查询结果（可能是多条记录）并返回。
    4. 关闭数据库连接。
    Functions:
    1. Connect to the database.
    2. Execute a query operation in the recognition_record table to retrieve all records.
    3. Get the query results (possibly multiple records) and return them.
    4. Close the database connection.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM recognition_record")
    columns = [description[0] for description in cursor.description]
    results = [columns] + cursor.fetchall()
    conn.close()
    return results

if __name__ == "__main__":
    db_file = "./database.db"
    check_and_create_database(db_file)
    insert_recognition_record(db_file, "12")