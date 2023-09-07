#!/bin/bash
cur_date="`date +%Y%m%d`" 
echo $cur_date

start_time=$(date +%s)

echo "download history"
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

# echo "0. stocks_statistics"
# python statistics.py > data/stocks_statistics.jsonl
# sqlite3 data/stocks.db <<EOF
# .separator ";"
# delete from stock_statistics_info where stock_no<>'0';
# .import data/stocks_statistics.jsonl stock_statistics_info
# EOF

# echo "1. generate stocks sequence data"
# rm -f data/seq_train_*.txt
# python seq_preprocess.py train 0 > data/seq_train_0.txt &
# python seq_preprocess.py train 1 > data/seq_train_1.txt &
# python seq_preprocess.py train 2 > data/seq_train_2.txt &
# python seq_preprocess.py train 3 > data/seq_train_3.txt &
# python seq_preprocess.py train 4 > data/seq_train_4.txt 
# python seq_preprocess.py predict > data/seq_predict.data #
# #find data/ -name 'seq_train_*.txt' | xargs sed 'a\' | grep -v nan | sort -nr > data/seq_all_data_new.csv

# end_time=$(date +%s)
# cost_time=$[ $end_time-$start_time ]
# echo "1.done $(($cost_time/60))min $(($cost_time%60))s"
# start_time=$end_time 

# 将股票的序列数据导入到sqlite3中
echo "2. import into sqlite3.  stocks sequence data -> table:stock_for_transfomer"
sqlite3 data/stocks.db <<EOF
.separator ";"
.import data/seq_train_0.txt stock_for_transfomer
.import data/seq_train_1.txt stock_for_transfomer
.import data/seq_train_2.txt stock_for_transfomer
.import data/seq_train_3.txt stock_for_transfomer
.import data/seq_train_4.txt stock_for_transfomer
EOF

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "2.import into sqlite done $(($cost_time/60))min $(($cost_time%60))s"
start_time=$end_time 

# echo "3. split dataset as train,validate,test"
# python seq_data_split.py

# end_time=$(date +%s)
# cost_time=$[ $end_time-$start_time ]
# echo "3.split dataset done! $(($cost_time/60))min $(($cost_time%60))s"
# start_time=$end_time 

# echo "4. make pairs"
python seq_make_pairs.py 1 date > f_high_mean_rate/validate.date.txt &
python seq_make_pairs.py 2 date > f_high_mean_rate/test.date.txt &
python seq_make_pairs.py 1 stock > f_high_mean_rate/validate.stock.txt &
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


# end_time=$(date +%s)
# cost_time=$[ $end_time-$start_time ]
# echo "4. make pairs done! $(($cost_time/60))min $(($cost_time%60))s"
# start_time=$end_time 

# echo "5. training"
# python seq_transfomer_train.py training > seq_transfomer_train.log.$cur_date 

# end_time=$(date +%s)
# cost_time=$[ $end_time-$start_time ]
# echo "5.training done! $(($cost_time/60))min $(($cost_time%60))s"

echo "Done!"



# #导出
# sqlite3 data/stocks.db <<EOF
# .header off  
# .separator ";"
# .once data/seq_all_data.csv  
# SELECT * FROM stock_for_seq where trade_date>20180306 order by trade_date desc;
# EOF

# python tools/split_seq.py validate | sort -nr > data/seq_validate.txt &
# python tools/split_seq.py train | sort -nr > data/seq_train.txt 

# python sample_data.py seq train  80 

# python seq_group_pairwise.py validate 80 > data/seq_validate_pair_new.txt &
# python seq_group_pairwise.py train 80 > data/seq_train_pair_new.txt 

# # 导入，利用sqlite3的约束功能做数据merge
# sqlite3 data/stocks.db <<EOF
# .separator " "
# .import data/seq_validate_pair_new.txt stock_for_seq_pair;
# .import data/seq_train_pair_new.txt stock_for_seq_pair;
# EOF

# sqlite3 data/stocks.db <<EOF
# .header off  
# .separator ","
# .once data/seq_validate_pair.csv  
# SELECT * FROM stock_for_seq_pair where trade_date>20180306 and data_type='1' order by trade_date desc;
# .once data/seq_train_pair.csv  
# SELECT * FROM stock_for_seq_pair where trade_date>20180306 and data_type='0' order by trade_date desc limit 0,10;
# EOF


# #训练
# python seq_train_point.py > log/seq_train_point.${cur_date}.log 
# python seq_train_pair.py > log/seq_train_pair.${cur_date}.log 

# # 预测
# python seq_predict.py model_tmp