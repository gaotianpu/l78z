#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汉字字符级别的词嵌入, 预处理文件
"""
import os
import sys 

def process(line):
    char_labels = line.split("  ")
    # print(char_labels)
    for char_label in char_labels:
        # print(char_label)
        char,label = char_label.split(r"/")
        print(char,label)

# head -1 data/msr_train.txt | python preprocess.py
if __name__ == "__main__":
    for i,line in enumerate(sys.stdin):
        process(line.strip())