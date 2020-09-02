#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字符级别的词嵌入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2Vec(nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, context_word):
        emb = self.embeddings(context_word)
        hidden = self.linear(emb)
        out = F.log_softmax(hidden,dim=1)
        return out

def unit_test():
    word_to_ix = {"hello": 0, "world": 1}
    embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
    lookup_tensor = torch.tensor([word_to_ix["world"]], dtype=torch.long)
    print(lookup_tensor)
    hello_embed = embeds(lookup_tensor)
    print(hello_embed)
    return 

if __name__ == "__main__":
    unit_test()