#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchtext import data

from corpus import CorpusDataSet
from module import TextSentiment

LEARNING_RATE=4.0
EMBEDDING_SIZE = 128
BATCH_SIZE = 16
NUN_CLASS = 2

# CONTEXT_SIZE = 4
EPOCH_SIZE = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tb_writer = SummaryWriter('out/antispam')

def tokenizer(text):  # create a tokenizer function
    return list(text)

LABELS = data.Field(sequential=False, use_vocab=False) 
TEXT = data.Field(sequential=True, tokenize=tokenizer,
        fix_length=200,
                     lower=True)
fields = [('label', LABELS), ('res_url', None),
              ('user_url', None), ('text', TEXT)]

train, val, test = data.TabularDataset.splits(path='data/',
                                     train='train.tsv',
                                     validation='val.tsv',
                                     test='test.tsv',
                                     format='csv',
                                     csv_reader_params={"delimiter": "\t"},
                                     fields=fields,
                                     skip_header=False)

train_len = len(train)
val_len = len(val)
# print(train_len)



train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train, val, test),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
        device=device)

TEXT.build_vocab(train)
LABELS.build_vocab(train)

VOCAB_SIZE = len(TEXT.vocab)

model = TextSentiment(VOCAB_SIZE, EMBEDDING_SIZE, NUN_CLASS).to(device)
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
    # for t in text:
    #     print(t,t.shape)

    offsets = [0] + [len(entry) for entry in text] 
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0) 
    # print(offsets)
    text = torch.cat(text)
    # print(text.shape,offsets.shape)
    return text.to(device), offsets.to(device) 

for epoch in range(EPOCH_SIZE):
    train_loss = 0
    train_acc = 0
    for i,batch in enumerate(train_iterator):  
        label = batch.label.to(device)
        text, offsets = process_text_offsets(batch) 

        optimizer.zero_grad() 
        output = model(text, offsets) 
        loss = criterion(output, label)
        train_loss += loss.item()
        train_acc += (output.argmax(1) == label).sum().item()
        loss.backward()
        optimizer.step()  
        # break 
    
    valid_loss = 0
    valid_acc = 0
    for i,batch in enumerate(valid_iterator):  
        label = batch.label.to(device)
        text, offsets = process_text_offsets(batch) 
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, label)
            valid_loss += loss.item()
            valid_acc += (output.argmax(1) == label).sum().item()

    scheduler.step()
    
    tb_writer.add_scalar('training loss',
                                 train_loss/train_len,
                                 epoch)
    tb_writer.add_scalar('training acc',
                                 train_acc/train_len,
                                 epoch)
    tb_writer.add_scalar('validate loss',
                                 valid_loss/val_len,
                                 epoch)
    tb_writer.add_scalar('validate acc',
                                 valid_acc/val_len,
                                 epoch)
    print(epoch,train_loss/train_len,train_acc/train_len,valid_loss/val_len,valid_acc/val_len) 
    # break 
    torch.save(model.state_dict(), "out/model.%d.dict"%(epoch))

     

# data = CorpusDataSet('data/cbow4.txt', "data/vocab.txt", CONTEXT_SIZE)
# dataloader = DataLoader(data, batch_size=BATCH_SIZE,
#                         shuffle=True, num_workers=0)

# model = AntispamModule_1(vocab_size=data.get_vocab_size(), embedding_size=EMBEDDING_SIZE,
#              context_size=CONTEXT_SIZE)
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# # criterion = nn.NLLLoss()
# criterion1 = nn.CrossEntropyLoss().to(device)




# def train(): 
#     model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
    

#     for epoch in range(EPOCH_SIZE):
#         total_loss = 0.0
#         for i_batch, (neg_ctx, context, target) in enumerate(dataloader):
#             optimizer.zero_grad()
#             log_probs = model(target.to(device),context.to(device),neg_ctx.to(device))
#             # loss = criterion1(log_probs)
#             loss = log_probs
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#             # break
#             if i_batch % 20 == 1:
#                 print(epoch, i_batch, loss.item()/BATCH_SIZE, total_loss/(i_batch+1)/BATCH_SIZE)
#                 tb_writer.add_scalar('training loss v4',
#                                      loss.item(),
#                                      epoch*len(dataloader) + i_batch) 

# if __name__ == "__main__":
#     train()