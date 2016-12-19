# Using sklearn features to train the data
import pandas as pd
import sys
import string
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
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
from sklearn.cluster import KMeans
from sklearn import metrics
import _pickle as cPickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.base import TransformerMixin
import math
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

bw_list = []

class BadWordTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        X = X.astype('str').map(lambda x: x.split() if (x!='nan') else '')
        
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


dataframe_dataset["joined replaced stemmed"].fillna(" ", inplace=True)

print(dataframe_dataset["joined replaced stemmed"])
print("Using Naive Baye's")
'''
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
'''

# Naive Baye's
text_clf = Pipeline([
    ('features', FeatureUnion([
        ('ngram_tf_idf', Pipeline([
            ('counts', CountVectorizer(ngram_range=(2, 2))),
            ('tf_idf', TfidfTransformer())
            ])),
        ('badwords', BadWordTransformer())
        ])),
    ('classifier', MultinomialNB())
    ])

text_clf = text_clf.fit(dataframe_dataset["joined replaced stemmed"],dataframe_dataset["Insult"])

print("Print here the mean value of Naive Baye's")
print("Print metric report of Naive Baye's")

# Testing
dataframe_dataset_test = pd.read_csv("./Test_clean.csv", na_values='unknown', encoding="utf-8")
dataframe_dataset_test["joined replaced stemmed"].fillna(" ", inplace=True)

predicted = text_clf.predict(dataframe_dataset_test["joined replaced stemmed"])
#print(np.mean(predicted == dataframe_dataset_test["Insult"]))
#print(metrics.classification_report(dataframe_dataset_test["Insult"], predicted, target_names=target_names))


# LinearSVC
text_clf = Pipeline([
    ('features', FeatureUnion([
        ('ngram_tf_idf', Pipeline([
            ('counts', CountVectorizer(ngram_range=(2, 2))),
            ('tf_idf', TfidfTransformer())
            ])),
        ('badwords', BadWordTransformer())
        ])),
    ('classifier', LinearSVC())
    ])

text_clf = text_clf.fit(dataframe_dataset["joined replaced stemmed"],dataframe_dataset["Insult"])

print("Print here the mean value of LinearSVC")
print("Print metric report of LinearSVC")

# Testing
dataframe_dataset_test = pd.read_csv("./Test_clean.csv", na_values='unknown', encoding="utf-8")
dataframe_dataset_test["joined replaced stemmed"].fillna(" ", inplace=True)

predicted = text_clf.predict(dataframe_dataset_test["joined replaced stemmed"])
#print(np.mean(predicted == dataframe_dataset_test["Insult"]))
#print(metrics.classification_report(dataframe_dataset_test["Insult"], predicted, target_names=target_names))

#Logistic Regresson
text_clf = Pipeline([
    ('features', FeatureUnion([
        ('ngram_tf_idf', Pipeline([
            ('counts', CountVectorizer(ngram_range=(2, 2))),
            ('tf_idf', TfidfTransformer())
            ])),
        ('badwords', BadWordTransformer())
        ])),
    ('classifier', LogisticRegression())
    ])

text_clf = text_clf.fit(dataframe_dataset["joined replaced stemmed"],dataframe_dataset["Insult"])

print("Print here the mean value of Logistic Regression")
print("Print metric report of Logistic Regression")

# Testing
dataframe_dataset_test = pd.read_csv("./Test_clean.csv", na_values='unknown', encoding="utf-8")
dataframe_dataset_test["joined replaced stemmed"].fillna(" ", inplace=True)

predicted = text_clf.predict(dataframe_dataset_test["joined replaced stemmed"])
#print(np.mean(predicted == dataframe_dataset_test["Insult"]))
#print(metrics.classification_report(dataframe_dataset_test["Insult"], predicted, target_names=target_names))

#Random Forest Classifier
text_clf = Pipeline([
    ('features', FeatureUnion([
        ('ngram_tf_idf', Pipeline([
            ('counts', CountVectorizer(ngram_range=(2, 2))),
            ('tf_idf', TfidfTransformer())
            ])),
        ('badwords', BadWordTransformer())
        ])),
    ('classifier', RandomForestClassifier())
    ])

text_clf = text_clf.fit(dataframe_dataset["joined replaced stemmed"],dataframe_dataset["Insult"])

print("Print here the mean value of Random Forest Classifier")
print("Print metric report of Random Forest Classifier")

# Testing
dataframe_dataset_test = pd.read_csv("./Test_clean.csv", na_values='unknown', encoding="utf-8")
dataframe_dataset_test["joined replaced stemmed"].fillna(" ", inplace=True)

predicted = text_clf.predict(dataframe_dataset_test["joined replaced stemmed"])
#print(np.mean(predicted == dataframe_dataset_test["Insult"]))
#print(metrics.classification_report(dataframe_dataset_test["Insult"], predicted, target_names=target_names))

 
#KMeans Clustering
text_clf = Pipeline([
    ('features', FeatureUnion([
        ('ngram_tf_idf', Pipeline([
            ('counts', CountVectorizer(ngram_range=(2, 2))),
            ('tf_idf', TfidfTransformer())
            ])),
        ('badwords', BadWordTransformer())
        ])),
    ('classifier', KMeans(n_clusters = 2))
    ])

text_clf = text_clf.fit(dataframe_dataset["joined replaced stemmed"],dataframe_dataset["Insult"])

print("Print here the mean value of KMeans Clustering")
print("Print metric report of KMeans Clustering")

# Testing
dataframe_dataset_test = pd.read_csv("./Test_clean.csv", na_values='unknown', encoding="utf-8")
dataframe_dataset_test["joined replaced stemmed"].fillna(" ", inplace=True)

predicted = text_clf.predict(dataframe_dataset_test["joined replaced stemmed"])
#print(np.mean(predicted == dataframe_dataset_test["Insult"]))
#print(metrics.classification_report(dataframe_dataset_test["Insult"], predicted, target_names=target_names))


# My_Best_Configuration
text_clf = Pipeline([
    ('features', FeatureUnion([
        ('ngram_tf_idf', Pipeline([
            ('counts', CountVectorizer(ngram_range=(1, 2))),
            ('tf_idf', TfidfTransformer())
            ])),
        ('badwords', BadWordTransformer())
        ])),
    ('classifier', LinearSVC(penalty='l2', loss='squared_hinge',  C=1.0, fit_intercept=True, intercept_scaling=1, max_iter=1000))
    ])

text_clf = text_clf.fit(dataframe_dataset["joined replaced stemmed"],dataframe_dataset["Insult"])

print("Print here the mean value of Best Configuration")
print("Print metric report of Best Configuration")

# Testing
dataframe_dataset_test = pd.read_csv("./Test_clean.csv", na_values='unknown', encoding="utf-8")
dataframe_dataset_test["joined replaced stemmed"].fillna(" ", inplace=True)

predicted = text_clf.predict(dataframe_dataset_test["joined replaced stemmed"])
#print(np.mean(predicted == dataframe_dataset_test["Insult"]))
#print(metrics.classification_report(dataframe_dataset_test["Insult"], predicted, target_names=target_names))



SVC_values = []

for n in range(100, len(dataframe_dataset), 200):
    text_clf = text_clf.fit(dataframe_dataset[:n]["joined replaced stemmed"],dataframe_dataset[:n]["Insult"])
    predicted = text_clf.predict(dataframe_dataset_test["joined replaced stemmed"])
    SVC_values.append(metrics.f1_score(datafrane_dataset_test["Insult"], predicted, average='weighted'))

xaxis = [n for n in range(100, len(dataframe_dataset),200)]
plt.plot(xaxis,SVC_values,'r-', label='Best Configuration')
plt.xlabel('Training samples')
plt.ylabel('F1 scores')
plt.show()
'''
# save the classifier

with open('my_classifier.pkl', 'wb') as fid:
        cPickle.dump(text_clf, fid)

'''
SVC_values = []

for n in range(100, train_data_length, 200):
    text_clf = text_clf.fit(twenty_train.data[:n], twenty_train.target[:n])
    predicted = text_clf.predict(docs_test)
    SVC_values.append(metrics.f1_score(twenty_test.target, predicted, average='weighted'))
'''


