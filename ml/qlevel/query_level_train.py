#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump, load
import pickle
import scipy as sp

# #for py2, not for py3
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

df = pd.read_csv('data/query.data', sep="\t")
print(df.shape)

# ngram_range=(2, 3),
tfidf_vect = TfidfVectorizer(
    analyzer='char', decode_error='ignore', ngram_range=(1, 4), max_features=10000)
X = tfidf_vect.fit_transform(df['query'].values.astype('U'))
dump(tfidf_vect, 'model/TfidfVectorizer.joblib')
pickle.dump(tfidf_vect, open(
    'model/TfidfVectorizer.pickle', "wb"))  # , protocol=2

quey_len_list = np.array([[len(q)] for q in df['query'].values.astype('U')])
print(X.shape, quey_len_list.shape)
X = sp.sparse.hstack((quey_len_list, X))
print(X.shape)

model_smote = SMOTE()  # 建立SMOTE模型对象
X, Y = model_smote.fit_sample(X, df['label'])  # 输入数据并作过抽样处理
print(sum(Y), len(Y), sum(Y)*100/len(Y))

train_x, valid_x, train_y, valid_y = train_test_split(X, Y, random_state=10)
print(train_x.shape, valid_x.shape, train_y.shape, valid_y.shape)


def multi():
    # class_prior=[.4, .6]
    clf = MultinomialNB().fit(train_x, train_y)
    dump(clf, 'model/MultinomialNB.joblib')
    pickle.dump(clf, open('model/MultinomialNB.pickle', "wb"))  # , protocol=2

    predicted = clf.predict(valid_x)
    predicted_proba = clf.predict_proba(valid_x)
    # for i, pred_label in enumerate(predicted):
    #     if pred_label != valid_y.values[i]:
    #         print("%s\t%s\t%s\t%s\t%s" % (i, pred_label, predicted_proba[i][1],
    #               valid_y.values[i], valid_x.values[i]))

    print("MultinomialNB:")
    print(clf.score(train_x.toarray(), train_y),
          clf.score(valid_x.toarray(), valid_y))
    print(metrics.classification_report(valid_y, predicted,
                                        target_names=["0", "1"]))


def gbdt():
    clf = GradientBoostingClassifier().fit(train_x, train_y)
    dump(clf, 'model/GradientBoostingClassifier.joblib')
    pickle.dump(
        clf, open('model/GradientBoostingClassifier.pickle', "wb"))  #

    predicted = clf.predict(valid_x)
    predicted_proba = clf.predict_proba(valid_x)
    # for i, pred_label in enumerate(predicted):
    #     if pred_label != valid_y.values[i]:
    #         print("%s\t%s\t%s\t%s\t%s" % (i, pred_label, predicted_proba[i][1],
    #               valid_y.values[i], valid_x.values[i]))

    print("GradientBoostingClassifier:")
    print(clf.score(train_x.toarray(), train_y),
          clf.score(valid_x.toarray(), valid_y))
    print(metrics.classification_report(valid_y, predicted,
                                        target_names=["0", "1"]))


multi()
gbdt()
