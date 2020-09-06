#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Word2Vec, CBOW
from preprocess import CorpusData

writer = SummaryWriter('out/cbow')

embedding_size = 8
context_size = 4
epoch_size = 1000

data = CorpusData(corpus_file="data/zhihu.txt",
                  line_count=1000, allow_cache=True)
data.load_corpus()
train_data = data.get_bag_words()
# print(len(train_data))
# print(data.vocab_len)

# 使用 TensorBoard 可视化模型，数据和训练
# https://pytorch.apachecn.org/docs/1.4/6.html
# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

# model = Word2Vec(embedding_size, data.vocab_len)
model = CBOW(data.vocab_len, embedding_size, context_size)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_function = nn.NLLLoss()

# 存储embedding
def save_embedding(embeddings, id2word_dict, file_name):
    emb_size,emb_dimension = embeddings.weight.shape 
    embedding = embeddings.weight.data.numpy()
    # file_output = open(file_name, 'w')
    with open(file_name, 'w') as file_output:
        file_output.write('%d %d\n' % (emb_size, emb_dimension))
        for idx, word in id2word_dict.items():
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
        gain = (max(self.loss_list) - min(self.loss_list)) / max(self.loss_list)
        print("Loss gain: {}%".format(round(100*gain,2)))
        if gain < self.min_percent_gain:
            return True
        else:
            return False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train():
    for epoch in range(epoch_size):
        total_loss = 0
        for context, target in train_data:
            context_idxs = torch.tensor(
                data.get_idx_by_words(context), dtype=torch.long)
            target_idx = torch.tensor(
                data.get_idx_by_words([target]), dtype=torch.long)

            model.zero_grad()
            log_probs = model(context_idxs)
            loss = loss_function(log_probs, target_idx)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 2 == 0 or epoch == (epoch_size-1):
            print(epoch, total_loss/len(train_data))

            # PATH = './cbow.pth'
            # print(model.embeddings.data[:3])
            # torch.save(model.state_dict(), PATH)
            #model.load_state_dict(torch.load(PATH))

            writer.add_scalar('training loss',
                            total_loss/len(train_data),
                            epoch * len(train_data))

    print("final")


def train_2():
    pass


if __name__ == "__main__":
    train()
