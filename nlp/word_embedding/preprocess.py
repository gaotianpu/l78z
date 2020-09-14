#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汉字字符级别的词嵌入, 预处理文件
"""
import os
import random
import numpy as np
import math
from collections import Counter
import pickle
import time


class Preprocess():
    def __init__(self, corp_file, out_root, context_size=4):
        self.corp_file = corp_file
        self.out_root = out_root
        self.context_size = context_size
        self.vocab_file = self.out_root + "vocab.txt"
        self.cbow_file = self.out_root + "cbow4.txt"
        self.word_counts = {}

    def gen_vocabulary(self):
        """生成词典
        语料库很大情况下统计？
        """
        with open(self.corp_file, 'r') as f:
            corp_content = f.read().replace(" ", "").replace("\n", '')
            self.word_counts = dict(Counter(corp_content))

        # word  word_count  doc_count
        # https://blog.csdn.net/qq_21997625/article/details/85641246
        # word_info = {}
        # with open(self.corp_file, 'r') as f:
        #     for i,line in enumerate(f):
        #         line = line.replace(" ", "")
        #         line_word_count = dict(Counter(line))
        #         for k,v in line_word_count.items():
        #             word_count,doc_count = word_info.get(k,[0,0])
        #             word_count = word_count + v
        #             doc_count = doc_count + 1
        #             word_info[k] = [word_count,doc_count]

        with open(self.vocab_file, 'w') as f:
            li = ["\t".join([word, str(count)])
                  for word, count in self.word_counts.items()]
            li.insert(0, "\t".join(["UNK", "0"]))
            f.write("\n".join(li))

    def generate_cbow_data(self, raw_text):
        """一段分词好的wordlist，产出cbow需要的训练数据"""
        data = []
        for i in range(self.context_size, len(raw_text) - self.context_size):
            context = []
            for ii in range(-self.context_size, self.context_size+1):
                if ii != 0:
                    context.append(raw_text[i+ii])
            target = raw_text[i]
            context.append(target)
            data.append(context)
        return data

    def gen_cbow_data(self):
        with open(self.cbow_file, 'w') as fw:
            with open(self.corp_file, 'r') as fr:
                for i, line in enumerate(fr):
                    l = line.replace(" ", "").strip()
                    data = self.generate_cbow_data(l)
                    if data:
                        fw.write("\n".join(["\t".join(d) for d in data]))
                        fw.write("\n")

    def generate(self):
        self.gen_vocabulary()
        self.gen_cbow_data()


def run():
    p = Preprocess("data/zhihu.txt", "data/")
    p.generate()


if __name__ == "__main__":
    run()
