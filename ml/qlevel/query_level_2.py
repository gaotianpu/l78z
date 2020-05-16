#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics

df = pd.read_csv('data/query.data', sep="\t")
train_x = df['query']
train_y = df['label']

df1 = pd.read_csv('data/query.2w', sep="\t")
valid_x = df1['query']

tfidf_vect = TfidfVectorizer(
    analyzer='char', decode_error='ignore', max_features=5000)
tfidf_vect.fit(df['query'].values.astype('U'))
xtrain_tfidf = tfidf_vect.transform(train_x)

xvalid_tfidf = tfidf_vect.transform(valid_x.values.astype('U'))

clf = MultinomialNB().fit(xtrain_tfidf, train_y)
predicted = clf.predict(xvalid_tfidf)

clf1 = GaussianNB().fit(xtrain_tfidf.toarray(), train_y)
predicted1 = clf1.predict(xvalid_tfidf.toarray())

for i, pred_label in enumerate(predicted):
    # print( "%s\t%s" % (valid_x.values[i],pred_label))
    print("%s\t%s\t%s" % (valid_x.values[i], pred_label, predicted1[i]))


    #     if pred_label != valid_y.values[i]:
    #         print(i, pred_label, predicted_proba[i][1],
    #               valid_y.values[i], valid_x.values[i])

