#!/bin/bash
# 30 19 * * 1,2,3,4,5 source /home/gaotianpu/.bashrc && cd /mnt/d/gitee/l78z/stocks/ && sh run_train.sh > log/run_train.log  2>&1 &

cur_date="`date +%Y%m%d`" 
echo $cur_date


# 数据预处理 step1
# 已经处理过的，不再处理了？
echo "preprocess"
rm -f data/train_*.txt
python preprocess4xgb.py train 0 > data/train_0.txt &
python preprocess4xgb.py train 1 > data/train_1.txt &
python preprocess4xgb.py train 2 > data/train_2.txt &
python preprocess4xgb.py train 3 > data/train_3.txt &
python preprocess4xgb.py train 4 > data/train_4.txt 
python preprocess4xgb.py test > data/test.txt
python preprocess4xgb.py predict > data/predict.txt

#cat data/train_*.txt > data/all_data.csv
find data/ -name 'train_*.txt' | xargs sed 'a\' | grep -v nan > data/all_data.csv

#导入到sqlite3中，利用sqlite3的约束功能做数据merge
sqlite3 data/stocks.db <<EOF
.separator ","
.import data/all_data.csv stock_for_xgb
EOF

#导出
sqlite3 data/stocks.db <<EOF
.header off  
.mode csv  
.once data/all.sort.csv  
SELECT * FROM stock_for_xgb where trade_date>20180306 order by trade_date desc;
EOF

#拆分出训练集合测试集
# split into train,validate,test? data
shuf -n 100000 data/all.sort.csv | sort -rn > data/all_validate.txt #固定住，一周内不再动？
python tools/split_data.py data/all.sort.csv data/all_validate.txt > data/all_train.txt 

#采样，保证样本的相对均衡？
python sample_data.py all train  500 &


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
python predict.py model_tmp


# source /home/gaotianpu/.bashrc && cd /mnt/d/gitee/l78z/stocks/ && sh run_predict.sh