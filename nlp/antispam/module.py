#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model
https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
https://zhuanlan.zhihu.com/p/94941514
https://juejin.im/post/6854573220176068622
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextSentiment(nn.Module):
    """https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html"""
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

    def predict(self,x):
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)
