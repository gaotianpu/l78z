#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import random
from datetime import datetime
import pandas as pd
import numpy as np

X_START_IDX = 4
Y_IDX = 3
LABEL_TOP_IDX = 6

def convert_csv2svm_line(line, y_idx):
    fields = line.strip().split(",")

    svm_fields = ["%d:%s" % (f_idx+1, f)
                  for f_idx, f in enumerate(fields[X_START_IDX:])]
    # svm_fields.insert(0, "qid:%s%s" % (fields[0], fields[1]))
    svm_fields.insert(0, fields[y_idx])

    print(" ".join(svm_fields))


def process_one_date(li, label_idx, current_date):
    top_li = []
    other_li = []
    line_count = 0

    for i, line in enumerate(li):
        line_count = i
        parts = line.strip().split(",")
        label = int(parts[label_idx])
        if label > LABEL_TOP_IDX:
            top_li.append(line.strip())
        else:
            other_li.append(line.strip())

    group_count = int(line_count/8)
    for group_id in range(group_count):
        top_k = min(len(top_li), 3)
        other_k = min(len(other_li), 17)
        top_sample = random.choices(top_li, k=top_k)
        other_sample = random.choices(other_li, k=other_k)
        group_sample = top_sample + other_sample
        random.shuffle(group_sample)
        for line in group_sample:
            convert_csv2svm_line(line, label_idx)


def process(filename, label_idx=4):
    with open(filename, 'r') as f:
        last_date = 0
        last_date_rows = []
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            current_date = parts[0]

            if current_date != last_date and i != 0:
                process_one_date(last_date_rows, label_idx, last_date)
                last_date_rows = []

            last_date = current_date
            last_date_rows.append(line.strip())

        process_one_date(last_date_rows, label_idx, last_date)


# python preprocess_v4.py data/train.txt 4 > data/rank_high_train_date.txt
# python preprocess_v4.py data/train.txt 5 > data/rank_low_train_date.txt
# python preprocess_v4.py sort.train.shuf.txt 5 > sort.train.shuf.svm.txt
if __name__ == "__main__":
    filename = sys.argv[1]
    lable_idx = int(sys.argv[2])
    process(filename, lable_idx)
