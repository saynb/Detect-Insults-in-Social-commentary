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
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.datasets import load_files
import numpy as np
from sklearn import metrics
import _pickle as cPickle
from sklearn.naive_bayes import MultinomialNB

target_names = ['Not an insult', 'Insult']

# Training
dataframe_dataset = pd.read_csv("./Train_clean.csv", na_values='unknown', encoding="utf-8")

print("Using Naive Baye's")
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
print(len(dataframe_dataset["replaced stemmed"]))
text_clf = text_clf.fit(dataframe_dataset["replaced stemmed"],dataframe_dataset["Insult"])

print("Print here the mean value")
print("Print metric report")


# Testing

dataframe_dataset_test = pd.read_csv("./Test_clean.csv", na_values='unknown', encoding="utf-8")

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
