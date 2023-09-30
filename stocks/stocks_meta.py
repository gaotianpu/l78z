import sqlite3

conn = sqlite3.connect("file:data/stocks.db", uri=True)

def update_db(sql,data,conn=conn):
    cursor = conn.cursor()
    try:
        cursor.executemany(sql, data)
        conn.commit()
    except:
        print("exception")
        conn.rollback()

def update_stock_name():
    commit_id_list = []
    with open('today.bin', 'rb') as file:
        today_d = pickle.load(file)
        for k,v in today_d.items():
            commit_id_list.append((v[1],k))
            # print(k,v[1])
            
    sql = "update stock_basic_info set stock_name=? where stock_no=?;" 
    update_db(sql,commit_id_list)


if __name__ == "__main__":
    op_type = sys.argv[1] 
    print(op_type)
    if op_type == "update_stock_name":
        update_stock_name()
    