#!/bin/bash
echo "1"
python seq_make_pairs.py 1 date > f_high_mean_rate/validate.date.txt &
python seq_make_pairs.py 1 stock > f_high_mean_rate/validate.stock.txt 
python seq_make_pairs.py 2 date > f_high_mean_rate/test.date.txt &
python seq_make_pairs.py 2 stock > f_high_mean_rate/test.stock.txt 
# python seq_make_pairs.py 0 f_high_mean_rate > f_high_mean_rate/train.txt 

echo "2"
# date pair
python seq_make_pairs.py 0 date 0 > f_high_mean_rate/train.date.txt_0 &
python seq_make_pairs.py 0 date 1 > f_high_mean_rate/train.date.txt_1 
python seq_make_pairs.py 0 date 2 > f_high_mean_rate/train.date.txt_2 &
python seq_make_pairs.py 0 date 3 > f_high_mean_rate/train.date.txt_3 
python seq_make_pairs.py 0 date 4 > f_high_mean_rate/train.date.txt_4 &

echo "3"
# stock pair
python seq_make_pairs.py 0 stock 0 > f_high_mean_rate/train.stock.txt_0 
python seq_make_pairs.py 0 stock 1 > f_high_mean_rate/train.stock.txt_1 &
python seq_make_pairs.py 0 stock 2 > f_high_mean_rate/train.stock.txt_2 
python seq_make_pairs.py 0 stock 3 > f_high_mean_rate/train.stock.txt_3 &
python seq_make_pairs.py 0 stock 4 > f_high_mean_rate/train.stock.txt_4 