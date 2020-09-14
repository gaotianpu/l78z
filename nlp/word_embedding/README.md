1. Word Embedding
1.1 字符级，词汇级，通用工具
a. 字符/词汇转成词典, word2idx, idx2word
b. 生成one-hot vector? 不用，nn.embedding提供
c. CBOW (Continuous Bag Of Words) 数据生成。 选定词->预测上下文词汇
d. Skip-gram 数据生成   。上下文->目标词
训练数据是['a','b'], 还是[(['a','b'],c)]?
https://srijithr.gitlab.io/post/word2vec/
https://github.com/smafjal/continuous-bag-of-words-pytorch/blob/master/cbow_model_pytorch.py 
e. 训练好的词向量如何导出？ parameters = m.weight.numpy()
f. tensorbard 显示训练进度？
g. dataset,dataloader,小批量训练
h. 除了损失函数，效果如何衡量？
i. 负采样,Negative examples
https://rguigoures.github.io/word2vec_pytorch/

torch.mul


RuntimeError: stack expects each tensor to be equal size, but got [8] at entry 0 and [0] at entry 1


a. 文本转字符、文本分词