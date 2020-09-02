#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from model import Word2Vec
from preprocess import CorpusData

embedding_size = 8

data = CorpusData(corpus_file="data/zhihu.txt",line_count=10,allow_cache=False)
data.load_corpus()
train_data = data.get_bag_words()


model = Word2Vec(embedding_size, data.vocab_len)
optimizer = optim.SGD(model.parameters(),lr=0.01)

loss_function = nn.NLLLoss()

losses = []
for epoch in range(1000):
    total_loss = 0  
    for context,target in train_data: 
        context_idxs = torch.tensor([data.get_idx_by_word(context)],dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs,torch.tensor([data.get_idx_by_word(target)],dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
    losses.append(total_loss)
    if epoch % 10 == 0:
        print(epoch,total_loss/len(train_data))
print(epoch,total_loss/len(train_data))
print("final")

