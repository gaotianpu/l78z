#!/bin/bash
run(){
    epochs=$1
    cur_date="`date +%Y%m%d%H%M`" 
    echo $cur_date , $epochs

    # rm -f train.data_*
    rm -f train_pair_date.data_*

    # 1.pair.date
    
    # ln -sf /mnt/d/github/l78z/stocks/data4/pair.test.date.txt  data4/pair_dates_2.txt
    # ln -sf /mnt/d/github/l78z/stocks/data4/pair.validate.date.txt data4/pair_dates_1.txt 

    if [ -f "data4/shuf.pair.train.date.txt_$epochs" ];then
        echo "文件存在"
    else
        echo "文件不存在"
        shuf -n 204800  data4/pair.train.date.txt_0  > train_pair_date.data_0 &
        shuf -n 204800  data4/pair.train.date.txt_1 > train_pair_date.data_1 &
        shuf -n 204800  data4/pair.train.date.txt_2 > train_pair_date.data_2 &
        shuf -n 204800  data4/pair.train.date.txt_3 > train_pair_date.data_3 
        shuf -n 204800  data4/pair.train.date.txt_4 > train_pair_date.data_4
        cat train_pair_date.data_* >  data4/shuf.pair.train.date.txt_$epochs
    fi

    ln -sf /mnt/d/github/l78z/stocks/data4/shuf.pair.train.date.txt_$epochs data4/pair.train.date.txt

    ## 2.pair.stock
    #
    # ln -sf /mnt/d/github/l78z/stocks/data4/pair.test.date.txt data4/pair_stocks_2.txt
    # ln -sf /mnt/d/github/l78z/stocks/data4/pair.validate.date.txt data4/pair_stocks_1.txt 

    # shuf -n 204800  data4/pair.train.stock.txt_0 > train_pair_stock.data_0 &
    # shuf -n 204800  data4/pair.train.stock.txt_1 > train_pair_stock.data_1 &
    # shuf -n 204800  data4/pair.train.stock.txt_2 > train_pair_stock.data_2 &
    # shuf -n 204800  data4/pair.train.stock.txt_3 > train_pair_stock.data_3 
    # shuf -n 204800  data4/pair.train.stock.txt_4 > train_pair_stock.data_4

    # cat train_pair_stock.data_* >  data4/pair_stocks.train.data_$epochs
    # ln -sf /mnt/d/github/l78z/stocks/data4/pair_stocks.train.data_$epochs data4/pair_stocks_0.txt

    ## 3.pair_dates_stocks
    # 
    # cat train_pair_* >  data4/pair_dates_stocks.train.data_$epochs
    # ln -sf /mnt/d/github/l78z/stocks/data4/pair_dates_stocks.train.data_$epochs data4/pair_dates_stocks_0.txt
    ## 验证集、测试集
    # cat data4/pair.test.date.txt data4/pair.test.stock.txt > data4/pair_dates_stocks_2.txt
    # cat data4/pair.validate.date.txt data4/pair.validate.stock.txt > data4/pair_dates_stocks_1.txt
    
    python seq_train_pair.py training $epochs

    # 模型文件备份，多此一举？
    # cp model_pair_dates_stocks.pth model_pair_dates_stocks.pth.$epochs #备份
    # cp model_pair_dates.pth model_pair_dates.pth.$epochs #备份
    # cp model_pair_date.pth model_pair_date.pth.$epochs #备份
    # cp model_pair_stocks.pth model_pair_stocks.pth.$epochs #备份 
    cp model_pair.pth  model_pair.pth.$epochs 
}

start=4 #
run $(expr $start + 1)
run $(expr $start + 2)
run $(expr $start + 3)
run $(expr $start + 4)