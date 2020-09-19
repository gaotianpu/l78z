1. Word Embedding
1.1 字符级
1.2 词汇级
1.3 通用工具
a. 文本转字符、文本分词
b. 字符、词汇转成词典
c. word2idx, idx2word
d. 生成one-hot vector? 不用nn.embedding提供
e. CBOW (Continuous Bag Of Words) 数据生成。 选定词->预测上下文词汇
f. Skip-gram 数据生成   。上下文->目标词
训练数据是['a','b'], 还是[(['a','b'],c)]?
g. Negative examples
https://rguigoures.github.io/word2vec_pytorch/
h. 训练好的词向量如何导出？ parameters = m.weight.numpy()
https://srijithr.gitlab.io/post/word2vec/
https://discuss.pytorch.org/t/how-to-implement-skip-gram-or-cbow-in-pytorch/47625/3
https://github.com/smafjal/continuous-bag-of-words-pytorch/blob/master/cbow_model_pytorch.py
https://github.com/FraLotito/pytorch-continuous-bag-of-words/blob/master/cbow.py 
j. 除了损失函数，效果如何衡量？
h. tensorbard 显示训练进度？
2. 中文分词
3. 词性标注 

1. torch.mul(a, b) 矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵
    https://pytorch.org/docs/master/generated/torch.mul.html
2. torch.mm(a, b) 矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵
3. torch.bmm(a,b) a 的size为(b,h,w), b的size为(b,w,h), 注意a,b的维度必须为3. bmm的b意思是批量？
4. torch.sum
https://pytorch.org/docs/master/generated/torch.sum.html




RuntimeError: stack expects each tensor to be equal size, but got [8] at entry 0 and [0] at entry 1
