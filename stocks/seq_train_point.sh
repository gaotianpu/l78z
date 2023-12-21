#!/bin/bash

# 
#   1722652 data2/point_0_1.txt
#   1734547 data2/point_0_2.txt
#    857286 data2/point_0_3.txt
#    858519 data2/point_0_4.txt
#    425545 data2/point_0_5.txt
#    428279 data2/point_0_6.txt
#    428111 data2/point_0_7.txt
#    429861 data2/point_0_8.txt

# cat point_0_5.txt point_0_6.txt point_0_7.txt point_0_8.txt > point_0_5678.txt
# cat data2/point_1_*.txt > data2/point_sampled_1.txt
# cat data2/point_2_*.txt > data2/point_sampled_2.txt

run(){
    epochs=$1
    cur_date="`date +%Y%m%d%H%M`" 
    echo $cur_date , $epochs

    # shuf -n 428279 data2/point_0_1.txt > data2/point_0_1.txt_shuf
    # shuf -n 428279 data2/point_0_2.txt > data2/point_0_2.txt_shuf 
    # shuf -n 428279 data2/point_0_3.txt > data2/point_0_3.txt_shuf
    # shuf -n 428279 data2/point_0_4.txt > data2/point_0_4.txt_shuf
    # cat data2/point_0_5678.txt data2/point_0_*.txt_shuf > data2/point_sampled_0.txt.$epochs
    # mv data2/point_sampled_0.txt data2/point_sampled_0.txt.$epochs

    ln -sf /mnt/d/github/l78z/stocks/data2/point_sampled_0.txt.$epochs data2/point_0.txt

    ln -sf /mnt/d/github/l78z/stocks/data3/point_0_all.txt data3/point_0.txt
    ln -sf /mnt/d/github/l78z/stocks/data3/point_0_9.txt data3/point_0.txt
    ln -sf /mnt/d/github/l78z/stocks/data3/point_0_0_9.txt data3/point_0.txt
    ln -sf /mnt/d/github/l78z/stocks/data3/point_0_10.txt data3/point_0.txt
    ln -sf /mnt/d/github/l78z/stocks/data3/point_0_11.txt data3/point_0.txt
    ln -sf /mnt/d/github/l78z/stocks/data3/point_0_12.txt data3/point_0.txt

    echo "f_high_mean_rate"
    python seq_train_point.py training f_high_mean_rate #next_low_rate #next_high_rate #f_high_mean_rate 
    cp model_point_high_f2.pth model_point_high_f2.pth.$epochs
    # cp model_point_high1.pth model_point_high1.pth.$epochs #备份
    # cp model_point_low1.pth model_point_low1.pth.$epochs #备份
    # cp model_point_sampled.pth model_point_sampled.pth.$epochs #备份
}

start=6 #
run $(expr $start + 1) 
run $(expr $start + 2)
run $(expr $start + 3)
# run $(expr $start + 4) 
# run $(expr $start + 5)
# run $(expr $start + 6)
# run $(expr $start + 7)
# run $(expr $start + 8)
# run $(expr $start + 9)          

# sh seq_train_point.sh > log/seq_train_point.log.20231011 &