#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import re
import numpy as np
import pandas as pd
import joblib
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
import scipy as sp

# for py2, not for py3
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# rm -f data/query.sum_*
# split -l5000  data/query.sum  data/query.sum_

tfidf_vect = joblib.load('model/TfidfVectorizer.joblib')
clf = joblib.load('model/MultinomialNB.joblib')
# clf = joblib.load('model/GradientBoostingClassifier.joblib')

# tfidf_vect = pickle.load(open('model/TfidfVectorizer.pickle', 'rb'))
# clf = pickle.load(open('model/MultinomialNB.pickle', 'rb'))


def _fullmatch(regex, string, flags=0):
    if hasattr(re, 'fullmatch'):
        return re.fullmatch(regex, string)
    return re.match("(?:" + regex + r")\Z", string, flags=flags)


def process(fname):
    df = pd.read_csv(fname, sep="\t", quotechar=None,
                     quoting=3, header=None, names=['query'])

    X = tfidf_vect.transform(df["query"].values.astype('U'))
    print(X.shape)
    quey_len_list = np.array([[len(q)]
                              for q in df['query'].values.astype('U')])
    X = sp.sparse.hstack((quey_len_list, X))

    predicted = clf.predict(X)
    predicted_proba = clf.predict_proba(X)

    for i, pred_label in enumerate(predicted):
        if pred_label:
            q = df['query'].values[i]
            # 过滤单字符、纯标点符号类
            if len(str(q)) < 2:
                continue
            # matchObj = re.match(r"^[\W]+$",q)   #_fullmatch( r'[\W]+$', q, re.M|re.I)
            # if matchObj:
            #     continue
            # 去除#query# ？
            print(q)


if __name__ == "__main__":
    dirs = glob.glob("data/query.sum_*")
    for f in dirs:
        process(f)
