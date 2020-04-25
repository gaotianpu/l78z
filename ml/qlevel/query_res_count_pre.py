#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd
import joblib
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
import scipy as sp

tfidf_vect = joblib.load('model/TfidfVectorizer.res_count.joblib')
clf = joblib.load('model/GradientBoostingClassifier.res_count.joblib')
# clf = joblib.load('model/MultinomialNB.res_count.joblib')

# tfidf_vect = pickle.load(open('model/TfidfVectorizer.res_count.pickle', 'rb'))
# clf = pickle.load(open('model/MultinomialNB.res_count.pickle', 'rb'))
# clf = pickle.load(open('model/GradientBoostingClassifier.res_count..pickle', 'rb'))

if __name__ == "__main__":
    df = pd.read_csv('data/query.after_level', sep="\t", header=None, names=['query'])

    X = tfidf_vect.transform(df["query"].values.astype('U'))
    quey_len_list = np.array([[len(q)] for q in df['query'].values.astype('U')]) 
    X = sp.sparse.hstack((quey_len_list,X))
    # print(X.shape)

    predicted = clf.predict(X)
    predicted_proba = clf.predict_proba(X)

    for i, proba in enumerate(predicted_proba):
        if proba[1] > 0.45:
            query = df['query'].values[i] 
            print(query,predicted_proba[i][1])
        # label = df['label'].values[i]
        # print( "%s\t%s\t%s\t%s" % (query,pred_label,predicted_proba[i][1],label)) 
