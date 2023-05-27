#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

def load_vali():
    l = []
    with open('data/all_validate.txt','r') as f:
        for i,line in enumerate(f):
            parts = line.strip().split(",")
            l.append(parts[0]+parts[1])
    return set(l)

def process_all(datatype="train"):
    vali_keys = load_vali()
    
    with open('data/rnn_all_data.csv', 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split(";")
            key = parts[0]+parts[1]
            # print(key, key in vali_keys)
            if key in vali_keys:
                if datatype=="validate":
                    print(line.strip())
            else:
                if datatype=="train":
                    print(line.strip())

# python tools/split_rnn.py train > rnn_train.txt
# python tools/split_rnn.py validate > rnn_validate.txt
if __name__ == "__main__":
    datatype = sys.argv[1]
    assert datatype in ['train','validate'], "not train or validate"
    process_all(datatype)