
import sys, math, re
from operator import itemgetter
import math
import os
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import sklearn.datasets
import sklearn.metrics
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



print("Where is the dataset ? Specify the path")

path = r'./train.csv'
print("Path to dataset is \n" + path)

class Comment:
    def __init__(self, content):
        self.text = content

inp = open(path, 'r')
inp_header = inp.readline()
header_list = re.split("[,\n]" , inp_header)


#Reading Dataset
dataframe_dataset = pd.read_csv(path, na_values='unknown', encoding="utf-8")

# The list for the corresponding 1 and 0 values of the comments
dataframe_target = dataframe_dataset[['Insult']]
insult_target = dataframe_target.ix[:,0].tolist()
#print(insult_target)

col3 = dataframe_dataset[['Comment']]
# Removing \n
dataframe_dataset['Comment'] = dataframe_dataset['Comment'].str.replace('\n', '')
#print(dataframe_dataset['Comment'])
#example1 = BeautifulSoup(dataframe_dataset['Comment'][0], "lxml")

#print(example1.get_text)
# Removing all HTML Tags
dataframe_dataset['Comment'] = dataframe_dataset['Comment'].str.replace('[^\w\s]','')
#print(dataframe_dataset['Comment'])

# Converting all uppercase letters to lower case
dataframe_dataset['Comment'] = dataframe_dataset['Comment'].str.lower()
#print(dataframe_dataset['Comment'])

#Tokenizing all the words in a given sentence
dataframe_dataset['tokenized_sents'] = dataframe_dataset.apply(lambda row: nltk.word_tokenize(row['Comment']), axis=1)
#print(dataframe_dataset['tokenized_sents'])

#print(dataframe_dataset)

#Removing all the stop words which have no meaning
stop = stopwords.words('english')
#print(stop)
dataframe_dataset['tokenized_sents'] = dataframe_dataset['tokenized_sents'].apply(lambda x: [item for item in x if item not in stop])
#print(dataframe_dataset['tokenized_sents'])

# Stemming all the words
stemmer = SnowballStemmer('english')
dataframe_dataset['stemmed'] = dataframe_dataset["tokenized_sents"].apply(lambda x: [stemmer.stem(y) for y in x])
#print(dataframe_dataset['stemmed'])

#This is the list for all the words
list_of_all_words = []
list_of_all_words = dataframe_dataset['stemmed'].tolist()

# Separating the list of both insulting and non insulting comments
'''for i in dataframe_dataset['Insult']:
    if dataframe_dataset.iloc[i]['Insult'] == 1:

        insulting_list.append(dataframe_dataset['stemmed'])

print(insulting_list)'''
'''dataframe_dataset_filter = dataframe_dataset.loc[dataframe_dataset['Insult'] == 1]
insulting_list = dataframe_dataset_filter['stemmed'].tolist()
print(insulting_list)'''

map_words = {"u": "you", "em":"them", "da":"the", "yo":"you",
        "ur":"you", "won't": "will not", "won't": "will not",
        "can't": "can not", "i'm": "i am", "i'm": "i am", "ain't": "is not",
        "'ll": "will", "'t": "not", "'ve": "have", "'s": "is", "'re": "are",
        "'d": "would"}

dataframe_dataset['replaced stemmed'] = dataframe_dataset["stemmed"].map(lambda x: [map_words[x[i]] if x[i] in map_words else x[i] for i in range(len(x)) ])


# Taking google list of bad words
bw = open('full-list-of-bad-words.txt', 'r')

#inp_text = bw.read()  # raeding file generated from puzzleenerator.py
#inp_text = re.split('\n|,', inp_text)
#print(inp_text)

#print dataframe_dataset.applymap(lambda x: isinstance(x, (int, float))).all(0)

dataframe_dataset.to_csv('Train_clean.csv')


