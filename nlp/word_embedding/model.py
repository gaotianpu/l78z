#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字符级别的词嵌入
https://github.com/weberrr/pytorch_word2vec
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NGram(nn.Module):
    """https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html"""

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, 128)
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.fc1(embeds))
        out = self.fc2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


class CBOW(nn.Module):
    """https://github.com/smafjal/continuous-bag-of-words-pytorch/blob/master/cbow_model_pytorch.py
    """

    def __init__(self, vocab_size, embedding_size, context_size):
        """
        vocab_size: 词典大小
        embedding_size: 词向量维度
        context_size: 上下文大小
        """
        super(CBOW, self).__init__() 
        HIDDEN_SIZE = 512
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.fc1 = nn.Linear(context_size * 2 * embedding_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, vocab_size)

    def forward(self, inputs):
        batch_size = inputs.shape[0] 
        out = self.embeddings(inputs) 
        out = out.view(batch_size, -1) 
        out = self.fc1(out) 
        out = F.relu(out) 
        out = self.fc2(out) 
        out = F.log_softmax(out, dim=1) 
        return out


class SkipGram(nn.Module):
    """https://github.com/Andras7/word2vec-pytorch/blob/master/word2vec/model.py
    https://github.com/fanglanting/skip-gram-pytorch/blob/master/model.py
    https://github.com/blackredscarf/pytorch-SkipGram
    https://github.com/weberrr/pytorch_word2vec
    https://github.com/n0obcoder/Skip-Gram-Model
    https://github.com/tqvinhcs/SkipGram/blob/master/m_word2vec_pt.py
    https://github.com/PengFoo/word2vec-pytorch/blob/master/word2vec/model.py
    """

    def __init__(self, vocab_size, embedding_size, context_size):
        super(SkipGram, self).__init__()
        # self.emb_size = emb_size
        # self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(vocab_size, embedding_size, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_size, sparse=True)

        initrange = 1.0 / self.embedding_size
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)


class Word2Vec(nn.Module):
    def __init__(self, embedding_size, vocab_size, context_size):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, context_word):
        emb = self.embeddings(context_word)
        hidden = self.linear(emb)
        out = F.log_softmax(hidden, dim=1)
        return out


def unit_test():
    word_to_ix = {"hello": 0, "world": 1}
    embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
    # print(dir(embeds.weight))
    # print(embeds.state_dict)
    return 
    x,y = embeds.weight.shape
    print(x,y)
    lookup_tensor = torch.tensor([word_to_ix["world"]], dtype=torch.long)
    print(lookup_tensor)
    hello_embed = embeds(lookup_tensor)
    print(hello_embed)
    return


if __name__ == "__main__":
    unit_test()
