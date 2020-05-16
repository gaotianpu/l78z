
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
# from imblearn.over_sampling import SMOTE
from joblib import dump, load
import pickle


df = pd.read_csv('data/query.res.count', sep="\t")
print(df.shape)

train_x, valid_x, train_y, valid_y = train_test_split(
    df['query'], df['label'], random_state=10)

tfidf_vect = TfidfVectorizer(
    analyzer='char', decode_error='ignore', ngram_range=(1, 4), max_features=10000)
tfidf_vect.fit(df['query'].values.astype('U'))

xtrain_tfidf = tfidf_vect.transform(train_x.values.astype('U'))
xvalid_tfidf = tfidf_vect.transform(valid_x.values.astype('U'))


def multi():
    # class_prior=[.4, .6]
    clf = MultinomialNB().fit(xtrain_tfidf, train_y)
    dump(clf, 'model/MultinomialNB.res_count.joblib')
    # , protocol=2
    pickle.dump(clf, open('model/MultinomialNB.res_count.pickle', "wb"))

    predicted = clf.predict(xvalid_tfidf)
    predicted_proba = clf.predict_proba(xvalid_tfidf)
    # for i, pred_label in enumerate(predicted):
    # if pred_label != valid_y.values[i]:
    # print("%s\t%s\t%s\t%s\t%s" % (i, pred_label, predicted_proba[i][1],
    #         valid_y.values[i], valid_x.values[i]))

    print("MultinomialNB:")
    print(clf.score(xtrain_tfidf.toarray(), train_y),
          clf.score(xvalid_tfidf.toarray(), valid_y))
    print(metrics.classification_report(valid_y.values, predicted,
                                        target_names=["0", "1"]))


def gbdt():
    clf = GradientBoostingClassifier().fit(xtrain_tfidf, train_y)
    dump(clf, 'model/GradientBoostingClassifier.res_count.joblib')
    pickle.dump(
        clf, open('model/GradientBoostingClassifier.res_count.pickle', "wb"))  #

    predicted = clf.predict(xvalid_tfidf)
    predicted_proba = clf.predict_proba(xvalid_tfidf)
    # for i, pred_label in enumerate(predicted):
    #     if pred_label != valid_y.values[i]:
    #         print("%s\t%s\t%s\t%s\t%s" % (i, pred_label, predicted_proba[i][1],
    #               valid_y.values[i], valid_x.values[i]))

    print("GradientBoostingClassifier:")
    print(clf.score(xtrain_tfidf.toarray(), train_y),
          clf.score(xvalid_tfidf.toarray(), valid_y))
    print(metrics.classification_report(valid_y.values, predicted,
                                        target_names=["0", "1"]))


multi()
gbdt()
