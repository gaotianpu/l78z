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
python statistics.py > data/per_stocks_mean_std.jsonl
# python statistics.py > data/stocks_statistics.jsonl
# delete from stock_statistics_info where stock_no<>'0';
sqlite3 data/stocks.db <<EOF
.separator ";"
.import data/per_stocks_mean_std.jsonl stock_statistics_info
EOF

echo "2. generate stocks sequence data"
rm -f data/seq_train_*.txt
python seq_preprocess.py train 0 > data/seq_train_0.txt &
python seq_preprocess.py train 1 > data/seq_train_1.txt &
python seq_preprocess.py train 2 > data/seq_train_2.txt &
python seq_preprocess.py train 3 > data/seq_train_3.txt &
python seq_preprocess.py train 4 > data/seq_train_4.txt 
python seq_preprocess.py predict > data/seq_predict.data #

# python seq_preprocess_v2.py predict > data2/seq_predict.data & #
# python seq_preprocess_v2.py train 0 > data2/seq_train_0.txt &
# python seq_preprocess_v2.py train 1 > data2/seq_train_1.txt &
# python seq_preprocess_v2.py train 2 > data2/seq_train_2.txt &
# python seq_preprocess_v2.py train 3 > data2/seq_train_3.txt &
# python seq_preprocess_v2.py train 4 > data2/seq_train_4.txt &
# sqlite3 data/stocks_train_2.db <<EOF
# .separator ";"
# .import data2/seq_train_0.txt stock_for_transfomer
# .import data2/seq_train_1.txt stock_for_transfomer
# .import data2/seq_train_2.txt stock_for_transfomer
# .import data2/seq_train_3.txt stock_for_transfomer
# .import data2/seq_train_4.txt stock_for_transfomer
# EOF

# 将股票的序列数据导入到sqlite3中
echo "2.1. import into sqlite3.  stocks sequence data -> table:stock_for_transfomer"
sqlite3 data/stocks.db <<EOF
.separator ";"
.import data/seq_train_0.txt stock_for_transfomer
.import data/seq_train_1.txt stock_for_transfomer
.import data/seq_train_2.txt stock_for_transfomer
.import data/seq_train_3.txt stock_for_transfomer
.import data/seq_train_4.txt stock_for_transfomer
EOF

# data/stocks_train.db . 还要执行数据集拆分？
sqlite3 data/stocks.db <<EOF
.separator ";"
.import new_stocks_seq.txt.20230928 stock_for_transfomer
EOF



# .import data/seq_train.txt.0915.2 stock_for_transfomer

echo "3. split dataset as train,validate,test"
python seq_data_split.py

echo "4. make pairs"
python seq_make_pairs.py 1 date > f_high_mean_rate/validate.date.txt &
python seq_make_pairs.py 1 stock > f_high_mean_rate/validate.stock.txt &
python seq_make_pairs.py 2 date > f_high_mean_rate/test.date.txt &
python seq_make_pairs.py 2 stock > f_high_mean_rate/test.stock.txt &
# python seq_make_pairs.py 0 f_high_mean_rate > f_high_mean_rate/train.txt 

# date pair
python seq_make_pairs.py 0 date 0 > f_high_mean_rate/train.date.txt_0 &
python seq_make_pairs.py 0 date 1 > f_high_mean_rate/train.date.txt_1 &
python seq_make_pairs.py 0 date 2 > f_high_mean_rate/train.date.txt_2 &
python seq_make_pairs.py 0 date 3 > f_high_mean_rate/train.date.txt_3 &
python seq_make_pairs.py 0 date 4 > f_high_mean_rate/train.date.txt_4 &

# stock pair
python seq_make_pairs.py 0 stock 0 > f_high_mean_rate/train.stock.txt_0 &
python seq_make_pairs.py 0 stock 1 > f_high_mean_rate/train.stock.txt_1 &
python seq_make_pairs.py 0 stock 2 > f_high_mean_rate/train.stock.txt_2 &
python seq_make_pairs.py 0 stock 3 > f_high_mean_rate/train.stock.txt_3 &
python seq_make_pairs.py 0 stock 4 > f_high_mean_rate/train.stock.txt_4 &

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

