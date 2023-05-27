#!/bin/bash

# echo "preprocess"
# rm -f data/rnn_train_*.txt
# python preprocess4rnn.py train 0 > data/rnn_train_0.txt &
# python preprocess4rnn.py train 1 > data/rnn_train_1.txt &
# python preprocess4rnn.py train 2 > data/rnn_train_2.txt &
# python preprocess4rnn.py train 3 > data/rnn_train_3.txt &
# python preprocess4rnn.py train 4 > data/rnn_train_4.txt 
# python preprocess4rnn.py predict > data/rnn_predict.data
# find data/ -name 'rnn_train_*.txt' | xargs sed 'a\' | grep -v nan > data/rnn_all_data.csv
# python tools/split_rnn.py train > rnn_train.data &
# python tools/split_rnn.py validate > rnn_validate.data

# echo "download_daily"
# python download_daily.py > data/today_price.txt 
# echo "run_daily"
# python run_daily.py

# python sample_data.py all train  500 &
# python sample_data.py all validate & 
# python sample_data.py rnn train  100 &
# python sample_data.py rnn validate & 

# #rnn 训练
# python rnn_train.py > log/rnn_train.log
# # 预测
# python rnn_predict.py > data/rnn_predict.score

# xgboost train and predict
rm -f data/*.buffer
echo "train_regression.py 2"
python train_regression.py 2   # label-high
echo "train_regression.py 3"
python train_regression.py 3   # label-low
echo "train_rank_date.py 4"
python train_rank_date.py 4   # label-high
echo "train_rank_date.py 5"
python train_rank_date.py 5   # label-low

# 预测
echo "predict_test"
python predict.py

# echo "train_regression.py 4"
# python train_regression.py 4   # label-high
# echo "train_regression.py 5"
# python train_regression.py 5   # label-low
# echo "train_rank_date.py 4"
# python train_rank_date.py 4   # label-high
# echo "train_rank_date.py 5"
# python train_rank_date.py 5   # label-low

# sqlite3 data/stocks.db <<EOF
# .separator ","
# .import data/history_all.csv stock_raw_daily
# EOF

# sqlite3 data/stocks.db <<EOF
# .header off 
# .mode csv  
# .once test.csv  
# SELECT * FROM stock_for_xgb where trade_date>20180306 limit 0,10;
# EOF