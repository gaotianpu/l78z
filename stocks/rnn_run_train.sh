#!/bin/bash
cur_date="`date +%Y%m%d`" 
echo $cur_date

#  2>&1 &

echo "1. generate stocks sequence data"
rm -f data/rnn_train_*.txt
python preprocess4rnn.py train 0 > data/rnn_train_0.txt &
python preprocess4rnn.py train 1 > data/rnn_train_1.txt &
python preprocess4rnn.py train 2 > data/rnn_train_2.txt &
python preprocess4rnn.py train 3 > data/rnn_train_3.txt &
python preprocess4rnn.py train 4 > data/rnn_train_4.txt 
python preprocess4rnn.py predict > data/rnn_predict.txt
#find data/ -name 'rnn_train_*.txt' | xargs sed 'a\' | grep -v nan | sort -nr  > data/rnn_all_data_new.csv
 
# #导入到sqlite3中，利用sqlite3的约束功能做数据merge
echo "2. import sequence data to sqlite3 stock_for_transfomer"
# sqlite3 data/stocks.db <<EOF
# .header off
# .separator "；"
# .import data/rnn_train_0.txt stock_for_transfomer
# .import data/rnn_train_1.txt stock_for_transfomer
# .import data/rnn_train_2.txt stock_for_transfomer
# .import data/rnn_train_3.txt stock_for_transfomer
# .import data/rnn_train_4.txt stock_for_transfomer
# EOF

echo "3. split train,validate,test dataset"
python rnn_data_split.py 

echo "4. make pairs"
# python rnn_make_pairs.py 1 f_mean_rate > f_mean_rate/validate.txt &
# python rnn_make_pairs.py 2 f_mean_rate > f_mean_rate/test.txt &
# python rnn_make_pairs.py 0 f_mean_rate > f_mean_rate/train.txt 

# sample 
# shuf -n10240 f_mean_rate/validate.txt > validate.txt
# shuf -n10240 f_mean_rate/test.txt > test.txt

echo "5. training"
python transfomer_train.py > transfomer_train.log.$cur_date 

# #导出
# sqlite3 data/stocks.db <<EOF
# .header off  
# .separator ";"
# .once data/rnn_all_data.csv  
# SELECT * FROM stock_for_rnn where trade_date>20180306 order by trade_date desc;
# EOF

# python tools/split_rnn.py validate | sort -nr > data/rnn_validate.txt &
# python tools/split_rnn.py train | sort -nr > data/rnn_train.txt 

# python sample_data.py rnn train  80 

# python rnn_group_pairwise.py validate 80 > data/rnn_validate_pair_new.txt &
# python rnn_group_pairwise.py train 80 > data/rnn_train_pair_new.txt 

# # 导入，利用sqlite3的约束功能做数据merge
# sqlite3 data/stocks.db <<EOF
# .separator " "
# .import data/rnn_validate_pair_new.txt stock_for_rnn_pair;
# .import data/rnn_train_pair_new.txt stock_for_rnn_pair;
# EOF

# sqlite3 data/stocks.db <<EOF
# .header off  
# .separator ","
# .once data/rnn_validate_pair.csv  
# SELECT * FROM stock_for_rnn_pair where trade_date>20180306 and data_type='1' order by trade_date desc;
# .once data/rnn_train_pair.csv  
# SELECT * FROM stock_for_rnn_pair where trade_date>20180306 and data_type='0' order by trade_date desc limit 0,10;
# EOF


# #训练
# python rnn_train_point.py > log/rnn_train_point.${cur_date}.log 
# python rnn_train_pair.py > log/rnn_train_pair.${cur_date}.log 

# # 预测
# python rnn_predict.py model_tmp