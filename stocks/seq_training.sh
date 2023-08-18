#!/bin/bash

# shuf -n 409600 f_high_mean_rate/train.txt_0 > train.data_0
# shuf -n 409600 f_high_mean_rate/train.txt_1 > train.data_1
# shuf -n 409600 f_high_mean_rate/train.txt_2 > train.data_2
# shuf -n 409600 f_high_mean_rate/train.txt_3 > train.data_3
# shuf -n 409600 f_high_mean_rate/train.txt_4 > train.data_4

shuf -n 204800 f_high_mean_rate/train.txt_0 > train.data_0
shuf -n 204800 f_high_mean_rate/train.txt_1 > train.data_1
shuf -n 204800 f_high_mean_rate/train.txt_2 > train.data_2
shuf -n 204800 f_high_mean_rate/train.txt_3 > train.data_3
shuf -n 204800 f_high_mean_rate/train.txt_4 > train.data_4

shuf -n 204800 f_high_mean_rate_s/train.txt_0 > train.data_0
shuf -n 204800 f_high_mean_rate_s/train.txt_1 > train.data_1
shuf -n 204800 f_high_mean_rate_s/train.txt_2 > train.data_2
shuf -n 204800 f_high_mean_rate_s/train.txt_3 > train.data_3
shuf -n 204800 f_high_mean_rate_s/train.txt_4 > train.data_4

cat train.data_* >  train.data

python seq_transfomer.py training

# cp StockForecastModel.pth.0 StockForecastModel.pth.0.5
# cp StockForecastModel.pth.0 StockForecastModel.pth

# mv train.data train_2.data