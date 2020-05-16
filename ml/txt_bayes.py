#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贝叶斯分类：https://blog.csdn.net/luanpeng825485697/article/details/78967139
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer #文本特征提取组件中，导入特征向量计数函数
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
print(twenty_train.target)
print(twenty_train.target_names)  # 训练集中类别的名字，这里只有四个类别
print(len(twenty_train.data))  # 训练集中数据的长度
print(len(twenty_train.filenames))  # 训练集文件名长度
print('-----')
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print('-----')
print(twenty_train.target_names[twenty_train.target[0]])
print('-----')
print(twenty_train.target[:10])  # 前十个的类别
print('-----')
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])  # 类别的名字
print('-----') 

count_vect = CountVectorizer()  # 特征向量计数函数
X_train_counts = count_vect.fit_transform(twenty_train.data)  # 对文本进行特征向量处理
print(X_train_counts)  # 特征向量和特征标签
print(X_train_counts.shape)  # 形状
print('-----') 
print(count_vect.vocabulary_.get(u'algorithm'))
print('-----')


tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)  # 建立词频统计函数,注意这里idf=False
print(tf_transformer)  # 输出函数属性 TfidfTransformer(norm=u'l2', smooth_idf=True, sublinear_tf=False, use_idf=False)
print('-----')
X_train_tf = tf_transformer.transform(X_train_counts)  # 使用函数对文本文档进行tf-idf频率计算
print(X_train_tf)
print('-----')
print(X_train_tf.shape)
print('-----') 

tfidf_transformer = TfidfTransformer()  # 这里使用的是tf-idf
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf)
print(X_train_tfidf.shape)
print('-----') 


# 中的MultinomialNB多项式函数
clf = MultinomialNB()  # 加载多项式函数
x_clf = clf.fit(X_train_tfidf, twenty_train.target)  # 构造基于数据的分类器
print(x_clf)  # 分类器属性：MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
print('-----')

docs_new = ['God is love', 'OpenGL on the GPU is fast']  # 文档
X_new_counts = count_vect.transform(docs_new)  # 构建文档计数
X_new_tfidf = tfidf_transformer.transform(X_new_counts)  # 构建文档tfidf
predicted = clf.predict(X_new_tfidf)  # 预测文档
print(predicted)  # 预测类别 [3 1]，一个属于3类，一个属于1类
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))  # 将文档和类别名字对应起来
print('-----')


twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))  # 预测的值和测试值的比例，mean就是比例函数
print('-----')  # 精度已经为0.834886817577

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
print(text_clf)  # 构造分类器，分类器的属性
predicted = text_clf.predict(docs_new)  # 预测新文档
print(predicted)  # 获取预测值
print('-----') 

text_clf = Pipeline(
    [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
# _ = text_clf.fit(twenty_train.data, twenty_train.target)  # 和下面一句的意思一样，一个杠，表示本身
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))  # 精度 0.912782956059
print('-----')

print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))




