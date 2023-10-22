import pandas as pd 
import sqlite3

conn = sqlite3.connect("file:data/stocks_train_4.db?mode=ro", uri=True)
        
order_models = "pair_15,list_235,point_5,point_4,pair_11".split(",")
model_files = order_models + "point_high1,low1".split(",")
# ,low1.7

# pair_15     FLOAT NOT NULL,
# list_235    FLOAT NOT NULL,
# point_5     FLOAT NOT NULL,
# point_4     FLOAT NOT NULL,
# pair_11     FLOAT NOT NULL,
# point_high1 FLOAT NOT NULL,
# low1    FLOAT NOT NULL,

def process():
    df = pd.read_csv("data/predict/predict_merged.txt.20231019",sep=";",header=0) 
    li_all = []
    for index, row in df.iterrows():
        pk_date_stock = row['pk_date_stock']
        stock_no = str(pk_date_stock)[8:]
        # print(stock_no)
        li_ret = [pk_date_stock,stock_no]
        sql = "select * from stock_for_boost_v2 where stock_no='%s'" % (stock_no)
        df_stocks = pd.read_sql(sql, conn) 
        # print(df_stocks.describe())
        df_stocks_count = len(df_stocks)
        
        if df_stocks_count == 0:
            print(stock_no,"'s count is 0 ?")
            continue 
        
        # #有点特殊，low1.7和low1每对上
        row["low1"] = row["low1.7"]
        
        for model_name in model_files:
            predict_score = row[model_name] 
            after_filter_stocks = df_stocks[df_stocks[model_name]<predict_score]
            position_idx = round(len(after_filter_stocks)/df_stocks_count,4)
            li_ret.append(position_idx)
            # print(model_name,predict_score,len(after_filter_stocks),position_idx)
        
        # predict_score = row['low1.7'] 
        # after_filter_stocks = df_stocks[df_stocks["low1"]<predict_score]
        # position_idx = round(len(after_filter_stocks)/df_stocks_count,4)
        # li_ret.append(position_idx)
        # print("low1.7",predict_score,len(after_filter_stocks),position_idx)
        li_all.append(li_ret)
        
        # if index>2:
        #     print(li_all)
        #     break
        
    columns=["pk_date_stock","stock_no"] + model_files
    df = pd.DataFrame(li_all,columns=columns)
    df.to_csv("data/tmp.txt",sep=";",index=False)

if __name__ == "__main__":
    process()