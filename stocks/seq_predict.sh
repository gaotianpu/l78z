#!/bin/bash
cur_date="`date +%Y%m%d`" 
echo $cur_date

rm -f data/history/*
rm -f log/download_history.log

echo "1.0 下载最新数据，导入stock_raw_daily"
python download_history.py 0 &
python download_history.py 1 &
python download_history.py 2 
# 再次下载，可以把首次下载不成功部分找补回来
python download_history.py -1 

mv log/download_history.log  log/download_history.log.$cur_date 

cat data/history/* > history.data.$cur_date
awk -F '%' '{print $1$2}' history.data.$cur_date > history.data.new

echo "1.1 导入stock_raw_daily"
sqlite3 data/stocks.db <<EOF
.separator ";"
.import history.data.new stock_raw_daily_2
EOF

# sqlite3 data/stocks.db <<EOF
# .separator ";"
# .import data/stock_raw_daily_3.txt stock_raw_daily
# EOF

# 更新统计信息
python statistics.py stocks

# sqlite3 data/stocks.db <<EOF
# .separator ","
# .import uncollect_stock_no.txt stock_basic_info
# EOF

# sqlite3 data/stocks.db <<EOF
# .separator ";"
# .import data/history/002913.csv stock_raw_daily_2
# EOF

echo "2. 生成predict需要的序列数据"
python seq_preprocess.py predict > data/seq_predict.data.$cur_date #
cp data/seq_predict.data.$cur_date seq_predict.data

echo "3. 调取模型，预测" # data/predict_merged.txt
python seq_model.py predict # > ret.seq_predict.txt.$cur_date  predict_buy_price.txt