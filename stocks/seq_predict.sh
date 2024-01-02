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

# python download_today.py all

echo "1.1 导入stock_raw_daily"
sqlite3 newdb/stocks.db <<EOF
.separator ";"
.import history.data.new stock_raw_daily
EOF

echo "1.2 生成并导入 stock_with_delta_daily"
python compute_day_delta.py incremental
sqlite3 newdb/stocks.db <<EOF
.separator ";"
.import day_delta_new.csv stock_with_delta_daily
EOF

python seq_preprocess_v4.py predict > data4/seq_predict/$cur_date.data #
cp data4/seq_predict/$cur_date.data seq_predict_v4.data

python seq_preprocess_v4.py predict_history $cur_date > data4/seq_predict/$cur_date.data

# echo "3. 调取模型，预测"
python seq_model_v4.py predict 

python download_today.py one_time

# python seq_train_class3.py evaluate_real 20231212

# python seq_model_merge.py intersection #将v1，v2结果取交集
# python seq_model_merge.py predict_true #