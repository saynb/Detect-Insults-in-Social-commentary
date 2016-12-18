# Using sklearn features to train the data
# Training
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import MultinomialNB




print("Using Naive Baye's")
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(2, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
print(len(insulting_list_1d))
text_clf = text_clf.fit(insulting_list_1d)

print("Print here the mean value")
from sklearn import metrics
print("Print metric report")
