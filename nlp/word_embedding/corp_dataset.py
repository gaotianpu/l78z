#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dataset

WRITING CUSTOM DATASETS, DATALOADERS AND TRANSFORMS
https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html?highlight=custom%20dataset
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html?highlight=custom%20dataset
"""
import torch
from torch.utils.data import Dataset, DataLoader
import time
import random
import numpy as np
from itertools import (takewhile, repeat)
import linecache


class CbowDataSet(Dataset):
    def __init__(self, corp_file, vocab_file, context_size):
        self.corp_file = corp_file
        self.vocab_file = vocab_file
        self.context_size = context_size

        self.word_counts = {}
        self.word_li = []
        self.word_dict = {}
        self.load_vocab()
        self.vocab_size = len(self.word_li)

    def load_vocab(self):
        """加载词典"""
        with open(self.vocab_file, 'r') as f:
            for idx, line in enumerate(f):
                word, count = line.strip().split("\t")
                self.word_counts[word] = int(count)
                self.word_li.append(word)
                self.word_dict[word] = idx

    def get_vocab_size(self):
        """词典大小"""
        return self.vocab_size

    def word2idx(self, words):
        """根据词列表找到对应"""
        idx_li = []
        for word in words:
            # 不存在的情况, 默认0-Unkown
            idx_li.append(self.word_dict.get(word, 0))
        return idx_li

    def sample_negative(self):
        """负样本采样"""
        sample_size = self.context_size * 2
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

    def __len__(self):
        """大文件统计行数
        https://www.cnblogs.com/jhao/p/13488867.html
        """
        buffer = 1024 * 1024
        with open(self.corp_file, 'r') as f:
            buf_gen = takewhile(lambda x: x, (f.read(buffer)
                                              for _ in repeat(None)))
            return sum(buf.count('\n') for buf in buf_gen)

    def __getitem__(self, idx):
        """大文件读取指定行, use linecache
        """
        line = linecache.getline(self.corp_file, idx).strip()
        parts_idx = self.word2idx(line.split("\t"))

        context = torch.tensor(parts_idx[:-1], dtype=torch.long)
        target = torch.tensor([parts_idx[-1]], dtype=torch.long)

        negative_ctx_idx = self.word2idx(next(self.sample_negative()))
        negative_ctx = torch.tensor(negative_ctx_idx, dtype=torch.long)

        return (negative_ctx, context, target)


def unit_test():
    st = time.time()
    data = CbowDataSet(corp_file='data/cbow4.txt',
                       vocab_file="data/vocab.txt", context_size=4)

    print(len(data))

    neg = data.sample_negative()
    print(next(neg))

    for i, row in enumerate(data):
        if i > 4:
            break
        print(i, row)

    print("========")

    batch_size = 30
    # dataloader = DataLoader(data, batch_size=batch_size,shuffle=False, num_workers=0)
    dataloader = DataLoader(data, batch_size, shuffle=True, num_workers=0)
    for i_batch, (neg_ctx, context, target) in enumerate(dataloader):
        if i_batch > 1:
            break
        for i in range(batch_size):
            print(context[i], target[i])

    print(len(dataloader))
    et = time.time()
    print("--- %s seconds ---" % (et-st))


if __name__ == "__main__":
    unit_test()
