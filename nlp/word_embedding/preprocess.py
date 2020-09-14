#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汉字字符级别的词嵌入, 预处理文件
"""
import os
import random
import math
import numpy as np
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
        """生成词典"""  
        with open(self.corp_file, 'r') as f:
            corp_content = f.read().replace(" ", "").replace("\n",'')
            # corp_content = "".join(
            #     [line.replace(" ", "").strip() for line in f])
            self.word_counts = dict(Counter(corp_content)) 

        words = sorted(set(self.word_counts.keys()))
        words.insert(0, "UNK")
        with open(self.vocab_file, 'w') as f:
            f.write("\n".join(words))

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

    def sample_negative(self, sample_size):
        """负样本采样"""
        sample_probability = {}
        normalizing_factor = sum([v**0.75 for v in self.word_counts.values()])
        for word in self.word_counts:
            sample_probability[word] = self.word_counts[word]**0.75 / \
                normalizing_factor
        words = np.array(list(self.word_counts.keys()))
        while True:
            word_list = []
            sampled_index = np.array(np.random.multinomial(
                sample_size, list(sample_probability.values())))
            for index, count in enumerate(sampled_index):
                for _ in range(count):
                    word_list.append(words[index])
            yield word_list

    def generate(self):
        self.gen_vocabulary()
        self.gen_cbow_data()


def run():
    p = Preprocess("data/zhihu.txt", "data/")
    p.generate()
    x = p.sample_negative(8)
    print(next(x))


if __name__ == "__main__":
    run()
