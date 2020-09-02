#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汉字字符级别的词嵌入
"""
import os
import pickle
import time


class CorpusData():
    def __init__(self, corpus_file, line_count=None,allow_cache=True):
        self.corpus_file = corpus_file
        self.line_count = line_count
        self.allow_cache = allow_cache 

        self.idx2word_file = 'data/idx2word.pickle'
        self.word2idx_file = 'data/word2idx.pickle'
        self.bagwords_file = 'data/bagwords4.txt'

        self.idx2word = []
        self.word2idx = {}
        self.vocab_len = 0

    def load_corpus(self):
        if self.allow_cache and os.path.exists(self.idx2word_file) and os.path.exists(self.word2idx_file):
            with open(self.idx2word_file, 'rb') as f:
                self.idx2word = pickle.load(f)
            with open(self.word2idx_file, 'rb') as f:
                self.word2idx = pickle.load(f)
            self.vocab_len = len(self.idx2word)
            return

        vocab_set = set()
        with open(self.corpus_file, 'r') as f:
            for i, line in enumerate(f):
                if self.line_count and i>self.line_count:
                    break 
                l = line.replace(" ", "").strip()
                for char in l:
                    if char not in vocab_set:
                        vocab_set.add(char)

        self.idx2word = sorted(vocab_set)
        self.idx2word.append("UNK")

        for i, t in enumerate(self.idx2word):
            self.word2idx[t] = i
        self.vocab_len = i

        # 持久化
        with open(self.idx2word_file, 'wb') as f:
            pickle.dump(self.idx2word, f)
        with open(self.word2idx_file, 'wb') as f:
            pickle.dump(self.word2idx, f)

    def get_word_by_idx(self, idx):
        # 越界情况？
        return self.idx2word[idx] if idx < self.vocab_len else self.idx2word[-1]

    def get_idx_by_word(self, word):
        idx = self.word2idx.get(word, -1)
        if idx < 0:
            idx = self.vocab_len
        return idx

    def get_one_hot(self, word):
        idx = self.get_idx_by_word(word)

    def get_bag_words(self, context_size=4):
        """CBOW"""
        context_tuple_list = []
        if self.allow_cache and os.path.exists(self.bagwords_file):
            print("cache")
            with open(self.bagwords_file, 'r') as f:
                for line in f:
                    context_tuple_list.append(line.strip().split("\t"))
            return context_tuple_list 
        
        with open(self.corpus_file, 'r') as f:
            with open(self.bagwords_file, 'w') as fw:
                for i, line in enumerate(f):
                    if self.line_count and i>self.line_count:
                        break 
                    txt = line.replace(" ", "").strip()
                    for ii, char in enumerate(txt):
                        start_idx = max(0, ii-context_size)
                        end_idx = min(ii+context_size, len(txt))
                        for iii in range(start_idx, end_idx):
                            if iii != ii:
                                item = [char, txt[iii]]
                                context_tuple_list.append(item)
                                fw.write("\t".join(item))
                                fw.write("\n")
        return context_tuple_list


def unit_test():
    corpus = ["中华人民共和国","印度阿三骚扰边境啦"]
    context_tuple_list = []
    w = 4

    for text in corpus:
        for i, word in enumerate(text):
            first_context_word_index = max(0,i-w)
            last_context_word_index = min(i+w, len(text))
            for j in range(first_context_word_index, last_context_word_index):
                if i!=j:
                    context_tuple_list.append((word, text[j]))
    print("There are {} pairs of target and context words".format(len(context_tuple_list)))
    print(context_tuple_list)
    return 

    raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

    # By deriving a set from `raw_text`, we deduplicate the array
    vocab = set(raw_text)
    vocab_size = len(vocab)

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    data = []
    for i in range(2, len(raw_text) - 2):
        context = [raw_text[i - 2], raw_text[i - 1],
                raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))
    print(data[:5])
    return 

    test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
    # we should tokenize the input, but we will ignore that for now
    # build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
    trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
                for i in range(len(test_sentence) - 2)]
    # print the first 3, just so you can see what they look like
    print(trigrams[:3])
    return 

    # word_to_ix = {"hello": 0, "world": 1}
    # embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
    # lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
    # hello_embed = embeds(lookup_tensor)
    # print(hello_embed)
    # return

    o = CorpusData(corpus_file="data/zhihu.txt",line_count=10,allow_cache=False)
    start_time = time.time()
    o.load_corpus()
    # o.get_bag_words()
    print(o.idx2word)
    print("--- %s seconds ---" % (time.time() - start_time))
    # whitout cache: 10.6s, 
    return

    
    print(o.idx2word)
    print(o.word2idx)
    print(o.get_word_by_idx(4))
    print(o.get_word_by_idx(99))
    print(o.get_idx_by_word("一"))
    print(o.get_idx_by_word("王"))


if __name__ == "__main__":
    unit_test()
