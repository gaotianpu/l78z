#!/bin/bash
cur_date="`date +%Y%m%d`" 
echo $cur_date

echo "1. generate stocks sequence data"
rm -f data/seq_train_*.txt
python seq_preprocess.py train 0 > data/seq_train_0.txt &
python seq_preprocess.py train 1 > data/seq_train_1.txt &
python seq_preprocess.py train 2 > data/seq_train_2.txt &
python seq_preprocess.py train 3 > data/seq_train_3.txt &
python seq_preprocess.py train 4 > data/seq_train_4.txt 
python seq_preprocess.py predict > data/seq_predict.txt #
#find data/ -name 'seq_train_*.txt' | xargs sed 'a\' | grep -v nan | sort -nr > data/seq_all_data_new.csv


# 将股票的序列数据导入到sqlite3中
echo "2. import stocks sequence data into sqlite3 table:stock_for_transfomer"
sqlite3 data/stocks.db <<EOF
.separator ","
.import data/seq_train_0.csv stock_for_transfomer
.import data/seq_train_1.csv stock_for_transfomer
.import data/seq_train_2.csv stock_for_transfomer
.import data/seq_train_3.csv stock_for_transfomer
.import data/seq_train_4.csv stock_for_transfomer
EOF

echo "3. split dataset as train,validate,test"
python seq_data_split.py

echo "4. make pairs"
python seq_make_pairs.py 1 f_mean_rate> f_mean_rate/validate.txt &
python seq_make_pairs.py 2 f_mean_rate> f_mean_rate/test.txt &
python seq_make_pairs.py 0 f_mean_rate> f_mean_rate/train.txt

echo "5. training"
python seq_transfomer_train.py

# echo "Done!"



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