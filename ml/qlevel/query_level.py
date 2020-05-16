#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE


def lower_sample_data(df, percent=1):
    '''
    percent:多数类别下采样的数量相对于少数类别样本数量的比例
    https://blog.csdn.net/xiaoxy97/article/details/82898812
    '''
    data1 = df[df['label'] == 1]  # 将多数类别的样本放在data1
    data0 = df[df['label'] == 0]  # 将少数类别的样本放在data0
    index = np.random.randint(
        len(data1), size=percent * (len(df) - len(data1)))  # 随机给定下采样取出样本的序号
    lower_data1 = data1.iloc[list(index)]  # 下采样
    return(pd.concat([lower_data1, data0]))


df = pd.read_csv('data/query.data', sep="\t")
print(df.shape)
# df = lower_sample_data(df) #下采样
# print(df.shape)


# ??? ValueError: could not convert string to float: ','
# model_smote = SMOTE() # 建立SMOTE模型对象
# x_smote_resampled, y_smote_resampled = model_smote.fit_sample(df['query'], df['label']) # 输入数据并作过抽样处理
# x_smote_resampled = pd.DataFrame(x_smote_resampled, columns=['query']) # 将数据转换为数据框并命名列名
# y_smote_resampled = pd.DataFrame(y_smote_resampled,columns=['label']) # 将数据转换为数据框并命名列名
# df = pd.concat([x_smote_resampled, y_smote_resampled],axis=1) # 按列合并数据框
# groupby_data_smote = df.groupby('label').count() # 对label做分类汇总
# print (groupby_data_smote) # 打印输出经过SMOTE处理后的数据集样本分类分布


train_x, valid_x, train_y, valid_y = train_test_split(df['query'], df['label'], random_state = 10)
train_x = df['query']
train_y = df['label']
print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)


tfidf_vect = TfidfVectorizer(
    analyzer='char', decode_error='ignore', max_features=5000)
tfidf_vect.fit(df['query'].values.astype('U'))
# print(tfidf_vect.vocabulary_)
# for idx, word in enumerate(tfidf_vect.get_feature_names()):
#     print("{}\t{}".format(word, tfidf_vect.idf_[idx]))

xtrain_tfidf = tfidf_vect.transform(train_x.values.astype('U'))
xvalid_tfidf = tfidf_vect.transform(valid_x.values.astype('U')) 


def gauss():
    clf = GaussianNB().fit(xtrain_tfidf.toarray(), train_y)
    # print(clf.feature_log_prob_)
    # print(clf.feature_log_prob_.shape)
    # print(clf.coef_)
    
    predicted = clf.predict(xvalid_tfidf.toarray())
    predicted_proba = clf.predict_proba(xvalid_tfidf.toarray())
    # for i, pred_label in enumerate(predicted):
    #     if pred_label != valid_y.values[i]:
    #         print(i, pred_label, predicted_proba[i][1],
    #               valid_y.values[i], valid_x.values[i])
    print(clf.score(xtrain_tfidf.toarray(),train_y),clf.score(xvalid_tfidf.toarray(),valid_y))
    print(metrics.classification_report(valid_y.values, predicted,
                                    target_names=["0", "1"]))
    print(valid_y.values.shape)
    print(predicted.shape)

def multi():
    # class_prior=[.4, .6]
    clf = MultinomialNB().fit(xtrain_tfidf, train_y)
    print(clf.feature_log_prob_)
    print(clf.feature_log_prob_.shape)
    print(clf.coef_) 

    predicted = clf.predict(xvalid_tfidf)
    predicted_proba = clf.predict_proba(xvalid_tfidf)
    # for i, pred_label in enumerate(predicted):
    #     if pred_label != valid_y.values[i]:
    #         print("%s\t%s\t%s\t%s\t%s" % (i, pred_label, predicted_proba[i][1],
    #               valid_y.values[i], valid_x.values[i]))

    # predicted = clf.predict(xvalid_tfidf.toarray())
    # predicted_proba = clf.predict_proba(xvalid_tfidf.toarray())
    
    # for i, pred_label in enumerate(predicted):
    #     if pred_label != valid_y.values[i]:
    #         print(i, pred_label, predicted_proba[i][1],
    #               valid_y.values[i], valid_x.values[i])

    print(clf.score(xtrain_tfidf.toarray(),train_y),clf.score(xvalid_tfidf.toarray(),valid_y))
    print(valid_y.values.shape)
    print(predicted.shape)

    print(metrics.classification_report(valid_y.values, predicted,
                                    target_names=["0", "1"]))

def gbdt():
    clf = GradientBoostingClassifier()
    clf.fit(xtrain_tfidf, train_y)
    predicted = clf.predict(xvalid_tfidf)
    predicted_proba = clf.predict_proba(xvalid_tfidf)
    for i, pred_label in enumerate(predicted):
        if pred_label != valid_y.values[i]:
            print("%s\t%s\t%s\t%s\t%s" % (i, pred_label, predicted_proba[i][1],
                  valid_y.values[i], valid_x.values[i]))

    print(clf.score(xtrain_tfidf.toarray(),train_y),clf.score(xvalid_tfidf.toarray(),valid_y))
    print(metrics.classification_report(valid_y.values, predicted,
                                    target_names=["0", "1"]))


# gauss()
multi()
# gbdt()
     




