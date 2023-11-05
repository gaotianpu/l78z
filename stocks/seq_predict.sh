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

echo "1.1 导入stock_raw_daily_2"
sqlite3 data/stocks.db <<EOF
.separator ";"
.import history.data.new stock_raw_daily_2
EOF

echo "1.2 扩展字段，导入stock_raw_daily_1"
rm -f data/trade_dates/*
python seq_preprocess_v2.py convert_stock_raw_daily
cat data/trade_dates/*.txt > data/stock_raw_daily_1.txt
cp data/stock_raw_daily_1.txt data/stock_raw_daily_1.txt.$cur_date
sqlite3 data/stocks.db <<EOF
.separator ";"
.import data/stock_raw_daily_1.txt stock_raw_daily_1
EOF

# 废弃掉了，直接生成导入stock_raw_daily_1形式的数据
# sqlite3 data/stocks.db <<EOF
# .separator ";"
# .import data/stock_raw_daily_3.txt stock_raw_daily
# EOF

# sqlite3 data/stocks.db <<EOF
# .separator ";"
# .import data/stock_statics.txt stock_statistics_info
# EOF

echo "2. 生成predict需要的序列数据"
python seq_preprocess.py predict > data/seq_predict/$cur_date.data #
cp data/seq_predict/$cur_date.data seq_predict.data

python seq_preprocess_v2.py predict > data/seq_predict_v2/$cur_date.data #
cp data/seq_predict_v2/$cur_date.data seq_predict_v2.data

echo "3. 调取模型，预测"
python seq_model.py predict # > predict_merged_for_show.txt
python seq_model_v2.py predict # > predict_merged_for_show_v2.txt

python seq_model_merge.py intersection #将v1，v2结果取交集