#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from model import Word2Vec, CBOW
from corp_dataset import CbowDataSet
# from preprocess import CorpusData

embedding_size = 8
context_size = 4
epoch_size = 10
batch_size = 500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = CbowDataSet('data/cbow4.txt', "data/vocab.txt", context_size)
dataloader = DataLoader(data, batch_size=batch_size,
                        shuffle=True, num_workers=0)

model = CBOW(data.get_vocab_size(), embedding_size, context_size)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_function = nn.NLLLoss()


# 使用 TensorBoard 可视化模型，数据和训练
# https://pytorch.apachecn.org/docs/1.4/6.html
# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
writer = SummaryWriter('out/cbow')

# 存储embedding


def save_embedding(embeddings, id2word_dict, file_name):
    emb_size, emb_dimension = embeddings.weight.shape
    embedding = embeddings.weight.data.numpy()
    # file_output = open(file_name, 'w')
    with open(file_name, 'w') as file_output:
        file_output.write('%d %d\n' % (emb_size, emb_dimension))
        # for idx, word in id2word_dict.items():
        for idx, word in enumerate(id2word_dict):
            e = embedding[idx]
            e = ' '.join(map(lambda x: str(x), e))
            file_output.write('%s %s\n' % (word, e))


class EarlyStopping():
    """https://rguigoures.github.io/word2vec_pytorch/"""

    def __init__(self, patience=5, min_percent_gain=0.1):
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
    for epoch in range(epoch_size):
        total_loss = 0
        for context, target in data:
            model.zero_grad()
            log_probs = model(context)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(epoch, total_loss)
    print("final")


def train_2():
    for epoch in range(epoch_size):
        for i_batch, (context, target) in enumerate(dataloader):
            total_loss = 0
            for i, txt in enumerate(context):
                model.zero_grad()
                log_probs = model(txt)
                loss = loss_function(log_probs, target[i])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(epoch, i_batch, total_loss/i)
            save_embedding(embeddings=model.embeddings,
                           id2word_dict=data.word_li, file_name="out/embddings")
            writer.add_scalar('training loss',
                              total_loss/i,
                              i_batch * i)

        break


if __name__ == "__main__":
    train()
    # train_2()
