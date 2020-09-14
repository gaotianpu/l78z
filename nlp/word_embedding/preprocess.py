#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汉字字符级别的词嵌入, 预处理文件
"""
import os
import pickle
import time


class Preprocess():
    def __init__(self, corp_file, out_root):
        self.corp_file = corp_file
        self.out_root = out_root
        self.vocab_file = self.out_root + "vocab.txt"
        self.cbow_file = self.out_root + "cbow4.txt"

    def gen_vocabulary(self):
        """生成词典"""
        vocab_set = set()
        with open(self.corp_file, 'r') as f:
            for i, line in enumerate(f):
                l = line.replace(" ", "").strip()
                for char in l:
                    if char not in vocab_set:
                        vocab_set.add(char)
        words = sorted(vocab_set)

        words.insert(0, "UNK")
        with open(self.vocab_file, 'w') as f:
            f.write("\n".join(words))

    def generate_cbow_data(self, raw_text, context_size=2):
        """一段分词好的wordlist，产出cbow需要的训练数据"""
        data = []
        for i in range(context_size, len(raw_text) - context_size):
            context = []
            for ii in range(-context_size, context_size+1):
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
                    data = self.generate_cbow_data(l, 4)
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
