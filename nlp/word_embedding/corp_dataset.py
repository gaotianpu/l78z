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


class CbowDataSet(Dataset):
    def __init__(self, corp_file, vocab_file, context_size):
        self.corp_file = corp_file
        self.vocab_file = vocab_file
        self.context_size = context_size
        self.word_li, self.word_dict = self.load_vocab()
        self.vocab_size = len(self.word_li)
        self.rows = self.load_data()

    def load_vocab(self):
        """加载词典"""
        word_li = []
        word_dict = {}
        with open(self.vocab_file, 'r') as f:
            for i, line in enumerate(f):
                word = line.strip()
                word_li.append(word)
                word_dict[word] = i
        return word_li, word_dict
    
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

    def load_data(self):
        """加载训练数据"""
        rows = []
        with open(self.corp_file, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) < self.context_size*2:
                    continue
                parts_idx = self.word2idx(parts)
                context = torch.tensor(parts_idx[:-1], dtype=torch.long)
                target = torch.tensor([parts_idx[-1]], dtype=torch.long)
                rows.append((context, target))
        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def unit_test():
    st = time.time()
    data = CbowDataSet(corp_file='data/cbow4.txt',
                       vocab_file="data/vocab.txt", context_size=4)
    for i, row in enumerate(data):
        if i > 4:
            break
        print(i, row)

    print("========")

    batch_size = 30
    # dataloader = DataLoader(data, batch_size=batch_size,shuffle=False, num_workers=0)
    dataloader = DataLoader(data, batch_size, shuffle=True, num_workers=0)
    for i_batch, (context, target) in enumerate(dataloader):
        if i_batch > 1:
            break
        for i in range(batch_size):
            print(context[i], target[i])

    print(len(dataloader))
    et = time.time()
    print("--- %s seconds ---" % (et-st))


if __name__ == "__main__":
    unit_test()
