#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字符级别的词嵌入
https://github.com/weberrr/pytorch_word2vec

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from numpy.random import multinomial
import numpy as np
from collections import Counter
import re
import itertools
# import nltk
# nltk.download('brown')
# from nltk.corpus import brown

# corpus = []
# for cat in ['news']:
#     for text_id in brown.fileids(cat):
#         raw_text = list(itertools.chain.from_iterable(brown.sents(text_id)))
#         text = ' '.join(raw_text)
#         text = text.lower()
#         text.replace('\n', ' ')
#         text = re.sub('[^a-z ]+', '', text)
#         corpus.append([w for w in text.split() if w != ''])
# print(len(corpus))
# print(corpus[:5])

corpus = ["""This notebook introduces how to implement the NLP technique, so-called word2vec, using Pytorch. The main goal of word2vec is to build a word embedding, i.e a latent and semantic free representation of words in a continuous space. To do so, this approach exploits a shallow neural network with 2 layers""".split()]
vocabulary = set(corpus[0])
word_to_index = {w: idx for (idx, w) in enumerate(vocabulary)}
index_to_word = {idx: w for (idx, w) in enumerate(vocabulary)}
# print(word_to_index)
# context_tuple_list = []
# print(corpus)


def sample_negative(corpus, sample_size):
    sample_probability = {}
    word_counts = dict(Counter(list(itertools.chain.from_iterable(corpus))))
    normalizing_factor = sum([v**0.75 for v in word_counts.values()])
    for word in word_counts:
        sample_probability[word] = word_counts[word]**0.75 / normalizing_factor
    words = np.array(list(word_counts.keys()))
    while True:
        word_list = []
        sampled_index = np.array(multinomial(
            sample_size, list(sample_probability.values())))
        for index, count in enumerate(sampled_index):
            for _ in range(count):
                word_list.append(words[index])
        yield word_list


def get_context_tuple_list(corpus, context_size=4):
    context_tuple_list = []
    negative_samples = sample_negative(corpus, 8)
    for text in corpus:
        for i, word in enumerate(text):
            first_context_word_index = max(0, i-context_size)
            last_context_word_index = min(i+context_size, len(text))
            for j in range(first_context_word_index, last_context_word_index):
                if i != j:
                    context_tuple_list.append(
                        (word, text[j], next(negative_samples)))
                    # context_tuple_list.append((word, text[j]))
    print("There are {} pairs of target and context words".format(
        len(context_tuple_list)))
    return context_tuple_list


class Word2Vec(nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super(Word2Vec, self).__init__()
        self.embeddings_target = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_context = nn.Embedding(vocab_size, embedding_size)

    def forward(self, target_word, context_word, negative_example):
        emb_target = self.embeddings_target(target_word)
        emb_context = self.embeddings_context(context_word)
        emb_product = torch.mul(emb_target, emb_context)
        emb_product = torch.sum(emb_product, dim=1)
        out = torch.sum(F.logsigmoid(emb_product))
        emb_negative = self.embeddings_context(negative_example)
        emb_product = torch.bmm(emb_negative, emb_target.unsqueeze(2))
        emb_product = torch.sum(emb_product, dim=1)
        out += torch.sum(F.logsigmoid(-emb_product))
        return -out


def get_batches(context_tuple_list, batch_size=100):
    random.shuffle(context_tuple_list)
    # print(context_tuple_list)
    batches = []
    batch_target, batch_context, batch_negative = [], [], []
    for i in range(len(context_tuple_list)):
        # print(context_tuple_list[i])
        batch_target.append(word_to_index[context_tuple_list[i][0]])
        batch_context.append(word_to_index[context_tuple_list[i][1]])
        batch_negative.append([word_to_index[w]
                               for w in context_tuple_list[i][2]])
        if (i+1) % batch_size == 0 or i == len(context_tuple_list)-1:
            tensor_target = torch.tensor(batch_target, dtype=torch.long)
            tensor_context = torch.tensor(batch_context, dtype=torch.long)
            tensor_negative = torch.tensor(batch_negative, dtype=torch.long)

            # tensor_target = autograd.Variable(
            #     torch.from_numpy(np.array(batch_target)).long())
            # tensor_context = autograd.Variable(
            #     torch.from_numpy(np.array(batch_context)).long())
            # tensor_negative = autograd.Variable(
            #     torch.from_numpy(np.array(batch_negative)).long())

            batches.append((tensor_target, tensor_context, tensor_negative))
            batch_target, batch_context, batch_negative = [], [], []
    return batches


class EarlyStopping():
    def __init__(self, patience=5, min_percent_gain=0.05):
        self.patience = patience
        self.loss_list = []
        self.min_percent_gain = min_percent_gain / 100.

    def update_loss(self, loss):
        self.loss_list.append(loss)
        if len(self.loss_list) > self.patience:
            del self.loss_list[0]

    def stop_training(self):
        if len(self.loss_list) == 1:
            return False
        gain = (max(self.loss_list) - min(self.loss_list)) / \
            max(self.loss_list)
        print("Loss gain: {}%".format(round(100*gain, 2)))
        if gain < self.min_percent_gain:
            return True
        else:
            return False


def train():
    context_tuple_list = get_context_tuple_list(corpus, context_size=4)
    vocabulary_size = len(vocabulary)

    loss_function = nn.CrossEntropyLoss()
    net = Word2Vec(embedding_size=200, vocab_size=vocabulary_size)
    optimizer = optim.Adam(net.parameters())
    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    early_stopping = EarlyStopping(patience=5, min_percent_gain=1)

    while True:
        losses = []
        context_tuple_batches = get_batches(context_tuple_list, batch_size=100)
        for target, context, negative in context_tuple_batches:
            net.zero_grad()
            loss = net(target, context, negative)
            loss.backward()
            optimizer.step()
            losses.append(loss.data)
        print("Loss: ", np.mean(losses))
        early_stopping.update_loss(np.mean(losses))
        if early_stopping.stop_training():
            break


if __name__ == "__main__":
    train()
    # x = get_context_tuple_list(corpus,4)
    # print(x[:5])
