#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from model import Word2Vec, CBOW
from corp_dataset import CbowDataSet

CONTEXT_SIZE = 4
EMBEDDING_SIZE = 256
BATCH_SIZE = 64
EPOCH_SIZE = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = CbowDataSet('data/cbow4.txt', "data/vocab.txt", CONTEXT_SIZE)
dataloader = DataLoader(data, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)

model = CBOW(vocab_size=data.get_vocab_size(), embedding_size=EMBEDDING_SIZE,
             context_size=CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.NLLLoss()

# use gpu
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=gpu
# https://pytorch.org/tutorials/beginner/saving_loading_models.html?highlight=gpu

# 使用 TensorBoard 可视化模型，数据和训练
# https://pytorch.apachecn.org/docs/1.4/6.html
# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
# tensorboard --logdir=out/cbow/ --bind_all
tb_writer = SummaryWriter('out/cbow')


def save_embedding(embeddings, id2word_dict, file_name):
    """存储embedding"""
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
    """only dataset"""
    for epoch in range(EPOCH_SIZE):
        total_loss = 0
        i = 0
        for neg_ctx, context, target in data:
            print(context, target)
            print(context.shape, target.shape)

            optimizer.zero_grad()
            log_probs = model(context)
            loss = criterion(log_probs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 300 == 0:
                print(epoch, i, total_loss/(i+1))
            i = i + 1

            break
        break
    print("final")


def train_2():
    for epoch in range(EPOCH_SIZE):
        for i_batch, (neg_ctx, context, target) in enumerate(dataloader):
            total_loss = 0
            for i, txt in enumerate(context):
                optimizer.zero_grad()
                log_probs = model(txt)
                loss = criterion(log_probs, target[i])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(epoch, i_batch, total_loss/BATCH_SIZE)
            tb_writer.add_scalar('training loss v2',
                                 total_loss/BATCH_SIZE,
                                 i_batch * BATCH_SIZE)
            save_embedding(embeddings=model.embeddings,
                           id2word_dict=data.word_li, file_name="out/embddings")


def train_3():
    for epoch in range(EPOCH_SIZE):
        total_loss = 0.0
        for i_batch, (neg_ctx, context, target) in enumerate(dataloader):
            optimizer.zero_grad()
            log_probs = model(context)
            loss = criterion(log_probs, target.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i_batch % 20 == 1:
                print(epoch, i_batch, loss.item(), total_loss/(i_batch+1))
                tb_writer.add_scalar('training loss v3',
                                     loss.item(),
                                     epoch*len(dataloader) + i_batch)

        save_embedding(embeddings=model.embeddings,
                       id2word_dict=data.word_li,
                       file_name="out/embddings_2")

def train_4():
    model = Word2Vec(vocab_size=data.get_vocab_size(), embedding_size=EMBEDDING_SIZE)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion1 = nn.CrossEntropyLoss()

    for epoch in range(EPOCH_SIZE):
        total_loss = 0.0
        for i_batch, (neg_ctx, context, target) in enumerate(dataloader):
            optimizer.zero_grad()
            log_probs = model(target.to(device),context.to(device),neg_ctx.to(device))
            # loss = criterion1(log_probs)
            loss = log_probs
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # break
            if i_batch % 20 == 1:
                print(epoch, i_batch, loss.item()/BATCH_SIZE, total_loss/(i_batch+1)/BATCH_SIZE)
                tb_writer.add_scalar('training loss v4',
                                     loss.item(),
                                     epoch*len(dataloader) + i_batch) 
        save_embedding(embeddings=model.embeddings_target,
                       id2word_dict=data.word_li,
                       file_name="out/embddings_4")
        break 

if __name__ == "__main__":
    # train()
    # train_2()
    # train_3()
    train_4()
