#!/bin/bash
cur_date="`date +%Y%m%d%H%M`" 
echo $cur_date

shuf -n 102400  f_high_mean_rate/train.date.txt_0  > train.data_0 &
shuf -n 102400  f_high_mean_rate/train.date.txt_1 > train.data_1 &
shuf -n 102400  f_high_mean_rate/train.date.txt_2 > train.data_2 &
shuf -n 102400  f_high_mean_rate/train.date.txt_3 > train.data_3 &
shuf -n 102400  f_high_mean_rate/train.date.txt_4 > train.data_4 &
shuf -n 102400  f_high_mean_rate/train.stock.txt_0 > train.data_5 &
shuf -n 102400  f_high_mean_rate/train.stock.txt_1 > train.data_6 &
shuf -n 102400  f_high_mean_rate/train.stock.txt_2 > train.data_7 &
shuf -n 102400  f_high_mean_rate/train.stock.txt_3 > train.data_8
shuf -n 102400  f_high_mean_rate/train.stock.txt_4 > train.data_9 

cat train.data_* >  train.data

cp StockForecastModel.pth StockForecastModel.pth.$cur_date #备份
python seq_transfomer.py training > training.log.$cur_date

mv train.data train.data.$cur_date #备份
