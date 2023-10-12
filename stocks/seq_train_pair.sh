#!/bin/bash
run(){
    epochs=$1
    cur_date="`date +%Y%m%d%H%M`" 
    echo $cur_date , $epochs

    shuf -n 102400  f_high_mean_rate/train.date.txt_0  > train.data_0 &
    shuf -n 102400  f_high_mean_rate/train.date.txt_1 > train.data_1 &
    shuf -n 102400  f_high_mean_rate/train.date.txt_2 > train.data_2 &
    shuf -n 102400  f_high_mean_rate/train.date.txt_3 > train.data_3 
    shuf -n 102400  f_high_mean_rate/train.date.txt_4 > train.data_4 

    # shuf -n 102400  f_high_8/train.date.txt_0  > train.data_0 &
    # shuf -n 102400  f_high_8/train.date.txt_1 > train.data_1 &
    # shuf -n 102400  f_high_8/train.date.txt_2 > train.data_2 &
    # shuf -n 102400  f_high_8/train.date.txt_3 > train.data_3 
    # shuf -n 102400  f_high_8/train.date.txt_4 > train.data_4 

    # shuf -n 102400  f_high_mean_rate/train.stock.txt_0 > train.data_5 &
    # shuf -n 102400  f_high_mean_rate/train.stock.txt_1 > train.data_6 &
    # shuf -n 102400  f_high_mean_rate/train.stock.txt_2 > train.data_7 &
    # shuf -n 102400  f_high_mean_rate/train.stock.txt_3 > train.data_8
    # shuf -n 102400  f_high_mean_rate/train.stock.txt_4 > train.data_9 

    cat train.data_* >  train.data

    python seq_train_pair.py training > log/training.pair.log.$epochs

    cp StockForecastModel.pair.pth StockForecastModel.pair.pth.$epochs #备份
    # mv train.data train.data.$cur_date #备份

    echo $cur_date, $epochs, "`date +%Y%m%d%H%M`" 
}

run1(){
    epochs=$1
    echo $epochs
}

start=0 # 12之前为随机数据，15-8only,
run $(expr $start + 1) 
run $(expr $start + 2)
run $(expr $start + 3)  
run $(expr $start + 4)  
 

# run
# run
