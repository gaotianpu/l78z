#!/bin/bash
# 只下载数据、预测，不训练
#  sh run_predict.sh > log/run_predict.log  2>&1 &
cur_date="`date +%Y%m%d`" 
echo $cur_date

# 数据下载  
rm -f data/history/* 
rm -f data/history_all.csv 

echo "1. download history"
python download_history.py 0 &
python download_history.py 1 &
python download_history.py 2 &
# 再次下载，可以把首次下载不成功部分找补回来
python download_history.py -1 

mv log/download_history.log  log/download_history.log.$cur_date

cat data/history/* > history.data
awk -F '%' '{print $1$2}' history.data > history.data.new
sqlite3 data/stocks.db <<EOF
.separator ";"
.import history.data.new stock_raw_daily_2
EOF

# python download_daily.py > data/today_price.txt && cp data/today_price.txt data/today_price_${cur_date}.txt & 

# python download_history.py 0 &
# python download_history.py 1 &
# python download_history.py 2 
# # 再次下载，可以把首次下载不成功部分找补回来
# python download_history.py -1 

# 有换行符的情况 
# find data/history/ -name '*.csv' | xargs sed 'a\' > data/history_all.csv 
# 单个文件结尾没有换行符的情况
cat data/history/*.csv | sort -nr > data/history_all.csv 

# 数据导入到sqlite3中
sqlite3 data/stocks.db <<EOF
.separator ","
.import data/history_all.csv stock_raw_daily
EOF

# 数据预处理
echo "preprocess"
python preprocess4xgb.py predict > data/predict.txt 
python preprocess4rnn.py predict > data/rnn_predict.txt #sample这个有点怪异

# 预测
echo "predict"
python predict.py model
python rnn_predict.py model
python rnn_predict_pair.py  model 

python lastNdays.py

#out: predict_scores_final_today.csv
python predict_merge.py
