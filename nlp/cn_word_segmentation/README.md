基于RNN中文分词
核心思路：基于双向RNN(GRU,LSTM等)实现seq2seq
1. 如何构建样本 
BMES(B词首,M词中,E词尾,S单字词), 举例：
“/s  人/b  们/e  常/s  说/s  生/b  活/e  是/s  一/s  部/s  教/b  科/m  书/e  ，/s  而/s  血/s  与/s  火/s  的/s  战/b  争/e  更/s  是/s  不/b  可/m  多/m  得/e  的/s  教/b  科/m  书/e  ，/s  她/s  确/b  实/e  是/s  名/b  副/m  其/m  实/e  的/s  ‘/s  我/s  的/s  大/b  学/e  ’/s  。/s 
2. PyTorch 训练 RNN 时，序列长度不固定怎么办？
https://zhuanlan.zhihu.com/p/59772104
https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html
3. data,label的形式
data 是字符在词典中的idx
label 是bmes构成的one-hot？ 
4. 损失函数？
The negative log likelihood loss
https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
neg_log_likelihood(inputs, targets) 在1.6版本中，改为NLLLoss


10分钟快速入门 PyTorch (5) – RNN：
https://www.pytorchtutorial.com/10-minute-pytorch-5/


文言文-白话文翻译？

开源项目参考：
https://github.com/Tasselkk/ChineseWordSegmentation
https://github.com/YaooXu/Chinese_seg_ner_pos
https://github.com/VenRay/ChineseSegmentationPytorch


1. 语料获得，可以先用jieba分词软件处理好分词作为训练数据
2. 构建训练样本？
正则表达式，识别中英文标点符号，
3. 开源语料库
http://sighan.cs.uchicago.edu/bakeoff2005/
https://spaces.ac.cn/archives/3924


BMES(B词首,M词中,E词尾,S单字词)，使用结巴分词产出训练数据(学习用)
使用RNN,LSTM,GRU,Bi-RNN,Bi-LSTM,Bi-GRU等实现
http://bcmi.sjtu.edu.cn/~zhaohai/pubs/huangchangning2006.pdf
http://www.strawman.site/public/index/singlepage/100




https://github.com/pytorch/examples/blob/master/word_language_model/model.py

