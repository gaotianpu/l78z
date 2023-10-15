#!/bin/bash
run(){
    epochs=$1
    cur_date="`date +%Y%m%d%H%M`" 
    echo $cur_date , $epochs

    shuf -n 102400  data2/pair.train.date.txt_0  > train.data_0 &
    shuf -n 102400  data2/pair.train.date.txt_1 > train.data_1 &
    shuf -n 102400  data2/pair.train.date.txt_2 > train.data_2 
    shuf -n 102400  data2/pair.train.date.txt_3 > train.data_3 &
    shuf -n 102400  data2/pair.train.date.txt_4 > train.data_4 &

    shuf -n 102400  data2/pair.train.stock.txt_0 > train.data_5 
    shuf -n 102400  data2/pair.train.stock.txt_1 > train.data_6 &
    shuf -n 102400  data2/pair.train.stock.txt_2 > train.data_7 &
    shuf -n 102400  data2/pair.train.stock.txt_3 > train.data_8
    shuf -n 102400  data2/pair.train.stock.txt_4 > train.data_9 

    cat train.data_* >  data2/pair_dates_stocks.train.data_$epochs

    # pair_dates_0.txt, pair_stocks_0.txt  pair_dates_stocks_0.txt
    # ln -sf /mnt/d/github/l78z/stocks/data2/pair_dates_stocks.train.data_$epochs data2/pair_dates_stocks_0.txt
    # python -u seq_train_pair.py training
    # cp model_pair.pth model_pair.pth.$epochs #备份
    # mv train.data train.data.$cur_date #备份

    echo $cur_date, $epochs, "`date +%Y%m%d%H%M`" 
}

start=0 # 12之前为随机数据，15-8only,
run $(expr $start + 1)
run $(expr $start + 2)
run $(expr $start + 3)
run $(expr $start + 4)