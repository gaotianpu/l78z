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


def process(filename, label_idx=4):
    top_li = []
    other_li = []
    line_count = 0
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            line_count = i
            parts = line.strip().split(",")
            label = int(parts[label_idx])
            if label > LABEL_TOP_IDX:
                top_li.append(line.strip())
            else:
                other_li.append(line.strip())

    # top_group_count = len(top_li)
    # for group_id in range(top_group_count):
    #     tmp = random.sample(top_li,20)
    #     print("\n".join(tmp))

    group_count = int(line_count/8)
    for group_id in range(group_count):
        top_sample = random.choices(top_li, k=3)
        other_sample = random.choices(other_li, k=17)
        group_sample = top_sample + other_sample
        random.shuffle(group_sample)
        for line in group_sample:
            convert_csv2svm_line(line, label_idx)
        # print("\n".join(group_sample))


# python preprocess_v3.py data/validate.txt 4 > data/rank_high_validate.txt
# python preprocess_v3.py data/train.txt 4 > data/rank_high_train.txt
# python preprocess_v3.py data/validate.txt 5 > data/rank_low_validate.txt
# python preprocess_v3.py data/train.txt 5 > data/rank_low_train.txt
if __name__ == "__main__":
    filename = sys.argv[1]
    lable_idx = int(sys.argv[2])
    process(filename, lable_idx)
