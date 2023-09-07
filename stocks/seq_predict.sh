#!/bin/bash
cur_date="`date +%Y%m%d`" 
echo $cur_date

start_time=$(date +%s)

rm -f data/history/*

echo "1. 下载最新数据，导入stock_raw_daily"
python download_history.py 0 &
python download_history.py 1 &
python download_history.py 2 &
# 再次下载，可以把首次下载不成功部分找补回来
python download_history.py -1 

mv log/download_history.log  log/download_history.log.$cur_date 

cat data/history/* > history.data.$cur_date
awk -F '%' '{print $1$2}' history.data.$cur_date > history.data.new
sqlite3 data/stocks.db <<EOF
.separator ";"
.import history.data.new stock_raw_daily_2
EOF

echo "2. 生成predict需要的序列数据"
python seq_preprocess.py predict > data/seq_predict.data.$cur_date #
cp data/seq_predict.data.$cur_date seq_predict.data

echo "3. 调取模型，预测"
python seq_transfomer.py predict > ret.seq_predict.txt.$cur_date