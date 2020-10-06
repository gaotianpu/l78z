#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dataset
"""
import torch
import torch.nn as nn
from torchtext import data
from torch.utils.data import Dataset, DataLoader
import time
import random
import numpy as np
from itertools import (takewhile, repeat)
import linecache
import jieba
from torchtext.vocab import GloVe
# import spacy
# spacy_en = spacy.load('en')


class CorpusDataSet(Dataset):
    def __init__(self, corp_file, vocab_file, context_size):
        pass

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
        pass


def test():
    """https://pytorch.org/text/examples.html
    https://zhuanlan.zhihu.com/p/65833208
    https://www.jianshu.com/p/0f7107db2f3a
    https://blog.csdn.net/u012436149/article/details/79310176
    https://blog.csdn.net/qq_25992377/article/details/105012948
    https://state-of-art.top/2018/11/28/torchtext%E8%AF%BB%E5%8F%96%E6%96%87%E6%9C%AC%E6%95%B0%E6%8D%AE%E9%9B%86/
    https://zhuanlan.zhihu.com/p/31139113
    https://torchtext.readthedocs.io/en/latest/data.html
    https://docs.python.org/3/library/csv.html#csv.reader
    https://pytorch.org/text/_modules/torchtext/data/dataset.html
    4085224503
    4085224503
    """
    def tokenizer(text):  # create a tokenizer function
        return list(text)
        # return list(jieba.cut(text))
        # return [tok.text for tok in spacy_en.tokenizer(text)]

    LABELS = data.Field(sequential=False, use_vocab=False) 
    TEXT = data.Field(sequential=True, tokenize=tokenizer,
                      init_token='<SOS>', eos_token='<EOS>', lower=True, fix_length=200)
    
    RES_URL = data.Field(sequential=True, use_vocab=False)
    USER_URL = data.Field(sequential=True, use_vocab=False)

    # TEXT = data.Field()
    # LABEL = data.Field()
    fields = [('label', LABELS), ('res_url', None),
              ('user_url', None), ('text', TEXT)]

    # train, val, test = data.TabularDataset.splits(
    #     path='./data/', train='train.tsv',
    #     validation='val.tsv', test='test.tsv', format='tsv',
    #     fields=[('Text', TEXT), ('Label', LABELS)])

    train, val, test = data.TabularDataset.splits(path='data/',
                                     train='train.tsv',
                                     validation='val.tsv',
                                     test='test.tsv',
                                     format='csv',
                                     csv_reader_params={"delimiter": "\t"},
                                     fields=fields,
                                     skip_header=False)
    

    

    # https://zhuanlan.zhihu.com/p/94941514
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train, val, test),
        batch_size=32,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
        device=torch.device('cpu'))

    # for batch in train_iterator:
    #     # text, text_len = batch.text
    #     print(batch.label.shape)
    #     print(batch.text.shape)
    #     break
    
    # vectors=GloVe(name='6B', dim=128)
    TEXT.build_vocab(train)
    LABELS.build_vocab(train)

    print(dir(TEXT.vocab))
    print(TEXT.vocab.freqs)
    
    # vocab = TEXT.vocab
    # print(TEXT.vocab)
    # print(len(TEXT.vocab))
    # print(TEXT.vocab.vectors.shape)
    # embed = nn.Embedding(len(vocab), 128)
    # embed.weight.data.copy_(TEXT.vocab.vectors)
    

    # print(TEXT.vocab.freqs.most_common(20))
    # print(TEXT.vocab.itos[:10])  #列表 index to word
    # print(TEXT.vocab.stoi)  # 字典 word to index

    # print(next(iter(train)))

    # print(dir(TEXT.vocab))
    # for v in TEXT.vocab:
    #     print(v)
    # print(dir(TEXT))
    # print(dir(LABELS))

    # # print(dir(train.examples[0]))
    # print(dir(train.examples[0]))

    # print(train.examples[0].label) 
    # print(train.examples[0].text)
    # print(train.examples[0].res_url)
    # print(train.examples[0].user_url)
    # print(dir(train))
    # VOCAB_SIZE = len(train.get_vocab())
    # NUN_CLASS = len(train.get_labels())
    # print(VOCAB_SIZE,NUN_CLASS)

    # for i,x in enumerate(training_data.examples):
    #     if i>5:break
    #     print(x)
    # print(list(x))
    # print(training_data.examples[0])


if __name__ == "__main__":
    test()
