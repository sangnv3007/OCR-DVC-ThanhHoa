import sqlite3
import datetime
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except:
        print("Lỗi kết nối db")

    return conn
def select_one_tasks(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    dict_values = {}
    cur = conn.cursor()
    cur.execute("SELECT * FROM thongketheongay")
    
    rows = cur.fetchall()
    cur.close()

    for row in rows:
        dict_values.update({str(row[0]):{'total_request': str(row[1]), 'successful': str(row[2]), 'failed': str(row[3]),
        'invalid_image': str(row[4]) }})
    return dict_values

def select_one_tasks_with_time(conn, time):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM thongketheongay WHERE TIME LIKE"+"'"+time+"'")
    rows = cur.fetchone()
    cur.close()
    return rows[0], rows[1], rows[2], rows[3], rows[4]
def select_time_last_record(conn):
    cur = conn.cursor()
    cur.execute('SELECT * FROM thongketheongay ORDER BY TIME DESC LIMIT 1 ')
    rows = cur.fetchone()
    cur.close()
    return rows[0]
def delete_one_record(conn, time):
    conn.execute("DELETE FROM thongketheongay WHERE TIME LIKE"+"'"+time+"'")
    conn.commit()
    print('Xoa ban ghi thanh cong')
def insert_record(conn,TIME,TOTAL_REQUEST,SUCCESSFUL,FAILED,INVALID_IMAGE):
    conn.execute(''' INSERT INTO thongketheongay(TIME, TOTAL_REQUEST,SUCCESSFUL,FAILED,INVALID_IMAGE)
            VALUES(?,?,?,?,?)''',(TIME, TOTAL_REQUEST,SUCCESSFUL,FAILED,INVALID_IMAGE))
    conn.commit()
    print('Record inserted')
def read_data(conn):
    data = conn.execute(''' SELECT * FROM thongketheongay''')
    for record in data:
        print('TIME : '+str(record[0]))
        print('TOTAL_REQUEST : '+str(record[1]))
        print('SUCCESSFUL : '+str(record[2]))
        print('FAILED : '+str(record[3]))
        print('INVALID_IMAGE : '+str(record[4])+'\n')   

def update_record(conn, TOTAL_REQUEST,SUCCESSFUL,FAILED,INVALID_IMAGE, TIME):
    conn.execute("UPDATE thongketheongay SET TOTAL_REQUEST=?,SUCCESSFUL=?,FAILED=?,INVALID_IMAGE=? WHERE TIME LIKE ?", (TOTAL_REQUEST,SUCCESSFUL,FAILED,INVALID_IMAGE,TIME))
    conn.commit()
    #print('Cap nhat thanh cong !')
def delete_all_record(conn):
    #dbase.execute("DELETE * FROM thongke")
    conn.execute('''DELETE from thongketheongay''')
    conn.commit()
    print('Xoa tat ca ban ghi thanh cong')
#dbase = sqlite3.connect('./model/thongke.pb') # Open a database File
#delete_one_record(dbase,'2023-01-06')
#delete_all_record(dbase)
# dbase.execute(''' CREATE TABLE thongketheongay(
#     TIME VARCHAR(10) PRIMARY KEY NOT NULL,
#     TOTAL_REQUEST INT NOT NULL,
#     SUCCESSFUL INT NOT NULL,
#     FAILED INT NOT NULL,
#     INVALID_IMAGE INT NOT NULL) ''')
#print('Table created')
###Insert values
#insert_record(dbase,'2023-01-06',0,0,0,0)
# a =  select_time_last_record(dbase)
# print(type(a))
#read_data(dbase)
# date_time = datetime.datetime.now().date()
#dict_values = select_one_tasks(dbase)
# print(a,b,c,d,e)
# print(total_request, successful, failed, invalid_image)
#update_record(dbase,0,0,0,0,'2023-01-01')
#read_data(dbase)
# dbase.close()
# print('Database Closed')