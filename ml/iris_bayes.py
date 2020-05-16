#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贝叶斯分类：https://blog.csdn.net/luanpeng825485697/article/details/78967139
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import datasets

# 加载数据集
iris = datasets.load_iris()

# a.高斯朴素贝叶斯
clf = GaussianNB()
clf = clf.fit(iris.data, iris.target)
y_pred=clf.predict(iris.data)
print("高斯朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))

# b.多项分布朴素贝叶斯
# 多项分布数据的朴素贝叶斯算法，也是用于文本分类(这个领域中数据往往以词向量表示，尽管在实践中 tf-idf 向量在预测时表现良好)的两大经典朴素贝叶斯算法之一
# 适用于文本分类
clf = MultinomialNB()
clf = clf.fit(iris.data, iris.target)
y_pred=clf.predict(iris.data)
print("多项分布朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))

# c. 伯努利朴素贝叶斯
# 每个特征 都假设是一个二元 (Bernoulli, boolean) 变量
clf = BernoulliNB()
clf = clf.fit(iris.data, iris.target)
y_pred=clf.predict(iris.data)
print("伯努利朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))


