#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchtext import data
import jieba

import corpus
# from corpus import load_data
from module import TextSentiment

LEARNING_RATE=4.0
EMBEDDING_SIZE = 128
BATCH_SIZE = 16
NUN_CLASS = 2

# CONTEXT_SIZE = 4
EPOCH_SIZE = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tb_writer = SummaryWriter('out/antispam')


# 
train_iterator, valid_iterator, test_iterator = corpus.load_data(BATCH_SIZE)
train_len = len(train_iterator.dataset)
val_len = len(valid_iterator.dataset)
vocab_size = len(train_iterator.dataset.fields['text'].vocab)

model = TextSentiment(vocab_size, EMBEDDING_SIZE, NUN_CLASS).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
# optimizer = optim.SparseAdam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

# print(model.embedding.weight.shape)
# print(VOCAB_SIZE,NUN_CLASS)

def process_text_offsets(batch):
    text = batch.text.permute(1,0) 
    # text = [entry for entry in text] 
    text = [entry[entry>1] for entry in text]  

    offsets = [0] + [len(entry) for entry in text] 
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)  

    text = torch.cat(text) 
    return text.to(device), offsets.to(device) 

def train_batch():
    train_loss = 0
    train_acc = 0
    for i,batch in enumerate(train_iterator):  
        label, text, offsets = corpus.process_text_offsets(batch) 

        optimizer.zero_grad() 
        output = model(text, offsets) 
        loss = criterion(output, label)
        train_loss += loss.item()
        train_acc += (output.argmax(1) == label).sum().item()
        loss.backward()
        optimizer.step() 

        t_len = (i+1)*BATCH_SIZE
        if i % 100 == 0:
            print('middle',i,train_loss/t_len,train_acc/t_len)

    
    return train_loss/train_len,train_acc/train_len

def vali_batch():
    valid_loss = 0
    valid_acc = 0
    for i,batch in enumerate(valid_iterator):  
        label, text, offsets = corpus.process_text_offsets(batch) 
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, label)
            valid_loss += loss.item()
            valid_acc += (output.argmax(1) == label).sum().item()
    return valid_loss/val_len,valid_acc/val_len



def main():
    for epoch in range(EPOCH_SIZE):
        train_loss,train_acc = train_batch()  
        valid_loss,valid_acc = vali_batch()  
        scheduler.step()
        
        tb_writer.add_scalar('training loss',
                                    train_loss,
                                    epoch)
        tb_writer.add_scalar('training acc',
                                    train_acc,
                                    epoch)
        tb_writer.add_scalar('validate loss',
                                    valid_loss,
                                    epoch)
        tb_writer.add_scalar('validate acc',
                                    valid_acc,
                                    epoch)
        print(epoch,train_loss,train_acc,valid_loss,valid_acc) 
        # break 
        torch.save(model.state_dict(), "out/model.%d.dict"%(epoch))

if __name__ == "__main__":
    main()

    