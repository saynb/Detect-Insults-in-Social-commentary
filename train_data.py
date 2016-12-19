# Using sklearn features to train the data
import pandas as pd
import sys
import string
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.datasets
import sklearn.metrics
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
import os
import nltk
import re
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.datasets import load_files
import numpy as np
from sklearn import metrics
import _pickle as cPickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import TransformerMixin

bw_list = []

class BadWordTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.map(lambda x: len(list(filter((lambda y: y in bw_list), x) ))))
        

    def fit(self, X, y=None, **fit_params):
        return self

target_names = ['Not an insult', 'Insult']

# Training
dataframe_dataset = pd.read_csv("./Train_clean.csv", na_values='unknown', encoding="utf-8")
bw = open('bad_words.txt', 'r')
inp_text = bw.read()  # raeding file generated from puzzleenerator.py
inp_text = re.split('\n|,', inp_text)

bw_list= [word.replace( u"\xa0",u"") for word in inp_text]

dataframe_dataset["replaced stemmed"] = \
        dataframe_dataset["replaced stemmed"].astype('str'). \
        map(lambda x: x[1:-1].split(',') if (x!='nan') else '')

dataframe_dataset["replaced stemmed"] = \
        dataframe_dataset["replaced stemmed"]. \
        map(lambda x: list(map(lambda s:str(s.strip("' ")), x)))

print(dataframe_dataset["replaced stemmed"])
print("Using Naive Baye's")
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
'''
text_clf = Pipeline([#('vect', CountVectorizer(ngram_range=(1, 1))),
                     ('tfidf', BadWordTransformer()),
                     ('clf', MultinomialNB()),
                     ])
'''
text_clf = text_clf.fit(dataframe_dataset["replaced stemmed"],dataframe_dataset["Insult"])

print("Print here the mean value")
print("Print metric report")


# Testing
dataframe_dataset_test = pd.read_csv("./Test_clean.csv", na_values='unknown', encoding="utf-8")
dataframe_dataset_test["replaced stemmed"] = \
        dataframe_dataset_test["replaced stemmed"].astype('str'). \
        map(lambda x: x[1:-1].split(',') if (x!='nan') else '')
dataframe_dataset_test["replaced stemmed"] = \
        dataframe_dataset_test["replaced stemmed"]. \
        map(lambda x: list(map(lambda s:str(s.strip("' ")), x)))

predicted = text_clf.predict(dataframe_dataset_test["replaced stemmed"])
print(np.mean(predicted == dataframe_dataset_test["Insult"]))
print(metrics.classification_report(dataframe_dataset_test["Insult"], predicted, target_names=target_names))


'''
# save the classifier
with open('my_classifier.pkl', 'wb') as fid:
        cPickle.dump(text_clf, fid)

SVC_values = []

for n in range(100, train_data_length, 200):
    text_clf = text_clf.fit(twenty_train.data[:n], twenty_train.target[:n])
    predicted = text_clf.predict(docs_test)
    SVC_values.append(metrics.f1_score(twenty_test.target, predicted, average='weighted'))
'''
