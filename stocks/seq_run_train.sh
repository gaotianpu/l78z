#!/bin/bash
cur_date="`date +%Y%m%d`" 
echo $cur_date

start_time=$(date +%s)

echo "1. download history"
python download_history.py 0 &
python download_history.py 1 &
python download_history.py 2 &
# 再次下载，可以把首次下载不成功部分找补回来
python download_history.py -1 

cat data/history/* > history.data
awk -F '%' '{print $1$2}' history.data > history.data.new
sqlite3 data/stocks.db <<EOF
.separator ";"
.import history.data.new stock_raw_daily_2
EOF

echo "1.2. stocks_statistics"
python seq_statistics.py > data/per_stocks_mean_std.jsonl
# python seq_statistics.py > data/stocks_statistics.jsonl
# delete from stock_statistics_info where stock_no<>'0';
sqlite3 data/stocks.db <<EOF
.separator ";"
.import data/per_stocks_mean_std.jsonl stock_statistics_info
EOF

echo "2. generate stocks sequence data"
# rm -f data/seq_train_*.txt
# python seq_preprocess.py train 0 > data/seq_train_0.txt &
# python seq_preprocess.py train 1 > data/seq_train_1.txt &
# python seq_preprocess.py train 2 > data/seq_train_2.txt &
# python seq_preprocess.py train 3 > data/seq_train_3.txt &
# python seq_preprocess.py train 4 > data/seq_train_4.txt 
# python seq_preprocess.py predict > data/seq_predict.data #

# # 将股票的序列数据导入到sqlite3中
# echo "2.1. import into sqlite3.  stocks sequence data -> table:stock_for_transfomer"
# sqlite3 data/stocks.db <<EOF
# .separator ";"
# .import data/seq_train_0.txt stock_for_transfomer
# .import data/seq_train_1.txt stock_for_transfomer
# .import data/seq_train_2.txt stock_for_transfomer
# .import data/seq_train_3.txt stock_for_transfomer
# .import data/seq_train_4.txt stock_for_transfomer
# EOF

sqlite3 day_delta/stocks.db <<EOF
.separator ";"
.import day_delta.txt stock_simple_feathers
EOF

python seq_preprocess_v2.py predict > data3/seq_predict.data & #
python seq_preprocess_v2.py train 0 > data3/seq_train_0.txt &
python seq_preprocess_v2.py train 1 > data3/seq_train_1.txt &
python seq_preprocess_v2.py train 2 > data3/seq_train_2.txt &
python seq_preprocess_v2.py train 3 > data3/seq_train_3.txt &
python seq_preprocess_v2.py train 4 > data3/seq_train_4.txt &

# 全量
python compute_day_delta.py history 0
python compute_day_delta.py history 1
python compute_day_delta.py history 2
python compute_day_delta.py history 3
python compute_day_delta.py history 4
# 增量
python compute_day_delta.py incremental  

# cat data5/day_delta/* > day_delta.csv
sqlite3 newdb/stocks.db <<EOF
.separator ";"
.import day_delta.csv stock_with_delta_daily
EOF

sqlite3 data/stocks_train_3.db <<EOF
.separator ";"
.import data3/seq_train_0.txt stock_for_transfomer
.import data3/seq_train_1.txt stock_for_transfomer
.import data3/seq_train_2.txt stock_for_transfomer
.import data3/seq_train_3.txt stock_for_transfomer
.import data3/seq_train_4.txt stock_for_transfomer
EOF

python seq_preprocess_v3.py train 0 > data3_2/seq_train_0.txt &
python seq_preprocess_v3.py train 1 > data3_2/seq_train_1.txt &
python seq_preprocess_v3.py train 2 > data3_2/seq_train_2.txt &
python seq_preprocess_v3.py train 3 > data3_2/seq_train_3.txt &
python seq_preprocess_v3.py train 4 > data3_2/seq_train_4.txt &

sqlite3 data3/stocks_train_v3.db <<EOF
.separator ";"  
.import data3/seq_train_0.txt stock_for_transfomer
.import data3/seq_train_1.txt stock_for_transfomer
.import data3/seq_train_2.txt stock_for_transfomer
.import data3/seq_train_3.txt stock_for_transfomer
.import data3/seq_train_4.txt stock_for_transfomer
EOF


ll newdb/

sqlite3 newdb/stocks.db <<EOF
.separator "|"
.import newdb/stock_raw_daily.txt stock_raw_daily
EOF

sqlite3 newdb/stocks.db <<EOF
.separator "|"
.import newdb/stock_raw_daily_old.txt stock_raw_daily
EOF

sqlite3 newdb/stocks.db <<EOF
.separator ";"
.import stock_basic_info.txt stock_basic_info
EOF

# stocks_qianfuquan.txt
# .once newdb/stock_raw_daily_old.txt
# select trade_date,stock_no,OPEN_price,CLOSE_price,change_amount,change_rate,LOW_price,HIGH_price,TURNOVER,TURNOVER_amount,TURNOVER_rate from stock_raw_daily_2;


python seq_preprocess_v4.py train 0 > data4/seq_train_0.txt &
python seq_preprocess_v4.py train 1 > data4/seq_train_1.txt &
python seq_preprocess_v4.py train 2 > data4/seq_train_2.txt &
python seq_preprocess_v4.py train 3 > data4/seq_train_3.txt &
python seq_preprocess_v4.py train 4 > data4/seq_train_4.txt &


sqlite3 data4/stocks_train_v4.db <<EOF
.separator ";"
.import data4/seq_train_0.txt stock_for_transfomer
.import data4/seq_train_1.txt stock_for_transfomer 
.import data4/seq_train_2.txt stock_for_transfomer 
.import data4/seq_train_3.txt stock_for_transfomer
.import data4/seq_train_4.txt stock_for_transfomer 
EOF

sqlite3 day_minmax/stocks.db <<EOF
.separator ";"
.import all_day_minmax.txt stock_minmax
EOF

# data/stocks_train.db . 还要执行数据集拆分？
sqlite3 data/stocks.db <<EOF
.separator ";"
.import new_stocks_seq.txt.20230928 stock_for_transfomer
EOF

# boost 模型需要用到的数据
sqlite3 data/stocks_train_4.db <<EOF
.separator ";"
.import boost_data.all stock_for_boost_v2
EOF

sqlite3 day_delta/stocks_h20.db <<EOF
.separator ";"
.import day_delta.txt stock_simple_feathers
EOF

sqlite3 /mnt/d/stock.db <<EOF 
.separator ";" 
.import stock_basic_info.txt stock_basic_info 
EOF



# .import data/seq_train.txt.0915.2 stock_for_transfomer

echo "3. split dataset as train,validate,test"
python seq_data_split.py

echo "4. make pairs"
python seq_make_pairs.py 1 date > data3/pair.validate.date.txt &
python seq_make_pairs.py 1 stock > data3/pair.validate.stock.txt &
python seq_make_pairs.py 2 date > data3/pair.test.date.txt &
python seq_make_pairs.py 2 stock > data3/pair.test.stock.txt &
# python seq_make_pairs.py 0 f_high_mean_rate > f_high_mean_rate/train.txt 

# date pair
python seq_make_pairs.py 0 date 0 > data4/pair.train.date.txt_0 &
python seq_make_pairs.py 0 date 1 > data4/pair.train.date.txt_1 &
python seq_make_pairs.py 0 date 2 > data4/pair.train.date.txt_2 &
python seq_make_pairs.py 0 date 3 > data4/pair.train.date.txt_3 &
python seq_make_pairs.py 0 date 4 > data4/pair.train.date.txt_4 &

# stock pair
python seq_make_pairs.py 0 stock 0 > data3/pair.train.stock.txt_0 &
python seq_make_pairs.py 0 stock 1 > data3/pair.train.stock.txt_1 &
python seq_make_pairs.py 0 stock 2 > data3/pair.train.stock.txt_2 &
python seq_make_pairs.py 0 stock 3 > data3/pair.train.stock.txt_3 &
python seq_make_pairs.py 0 stock 4 > data3/pair.train.stock.txt_4 &

###
# python seq_make_pairs.py 1 date > data2/pair.validate.date.txt &
# python seq_make_pairs.py 1 stock > data2/pair.validate.stock.txt &
# python seq_make_pairs.py 2 date > data2/pair.test.date.txt &
# python seq_make_pairs.py 2 stock > data2/pair.test.stock.txt &
# python seq_make_pairs.py 0 f_high_mean_rate > f_high_mean_rate/train.txt 

# date pair
python seq_make_pairs.py 0 date 0 > data2/pair.train.date.txt_0 &
python seq_make_pairs.py 0 date 1 > data2/pair.train.date.txt_1 &
python seq_make_pairs.py 0 date 2 > data2/pair.train.date.txt_2 &
python seq_make_pairs.py 0 date 3 > data2/pair.train.date.txt_3 &
python seq_make_pairs.py 0 date 4 > data2/pair.train.date.txt_4 &

# stock pair
python seq_make_pairs.py 0 stock 0 > data2/pair.train.stock.txt_0 &
python seq_make_pairs.py 0 stock 1 > data2/pair.train.stock.txt_1 &
python seq_make_pairs.py 0 stock 2 > data2/pair.train.stock.txt_2 &
python seq_make_pairs.py 0 stock 3 > data2/pair.train.stock.txt_3 &
python seq_make_pairs.py 0 stock 4 > data2/pair.train.stock.txt_4 &

echo "5. training"
# pair
python seq_transfomer_train.py training > seq_transfomer_train.log.$cur_date 
# point
python seq_regress.py training > seq_regress.log.$cur_date 


echo "Done!"

