#!/bin/bash

# shuf -n 409600 f_high_mean_rate/train.txt_0 > train.data_0
# shuf -n 409600 f_high_mean_rate/train.txt_1 > train.data_1
# shuf -n 409600 f_high_mean_rate/train.txt_2 > train.data_2
# shuf -n 409600 f_high_mean_rate/train.txt_3 > train.data_3
# shuf -n 409600 f_high_mean_rate/train.txt_4 > train.data_4

# shuf -n 204800 f_high_mean_rate/train.txt_0 > train.data_0 &
# shuf -n 204800 f_high_mean_rate/train.txt_1 > train.data_1 &
# shuf -n 204800 f_high_mean_rate/train.txt_2 > train.data_2 & 
# shuf -n 204800 f_high_mean_rate/train.txt_3 > train.data_3
# shuf -n 204800 f_high_mean_rate/train.txt_4 > train.data_4

# shuf -n 204800 f_high_mean_rate_s/train.txt_0 > train.data_0 &
# shuf -n 204800 f_high_mean_rate_s/train.txt_1 > train.data_1 &
# shuf -n 204800 f_high_mean_rate_s/train.txt_2 > train.data_2 &
# shuf -n 204800 f_high_mean_rate_s/train.txt_3 > train.data_3
# shuf -n 204800 f_high_mean_rate_s/train.txt_4 > train.data_4

# shuf -n 204800 f_high_mean_rate_1/train.txt_0 > train.data_0 &
# shuf -n 204800 f_high_mean_rate_1/train.txt_1 > train.data_1 &
# shuf -n 204800 f_high_mean_rate_1/train.txt_2 > train.data_2 &
# shuf -n 204800 f_high_mean_rate_1/train.txt_3 > train.data_3
# shuf -n 204800 f_high_mean_rate_1/train.txt_4 > train.data_4

shuf -n 102400 f_high_mean_rate_1/train.txt_0.s > train.data_0
shuf -n 204800 f_high_mean_rate_1/train.txt_4.s > train.data_4
cat train.data_* >  train.data

python seq_transfomer.py training # > training.log.00001.2 

# cp StockForecastModel.pth.0 StockForecastModel.pth_layer9_4
# cp StockForecastModel.pth.0 StockForecastModel.pth
#  rm -f StockForecastModel.pth.0.*
# mv train.data train_2.data
# mv train_4.data train.data
