
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


def clean_dataset(dataframe_dataset):

    dataframe_dataset['Backup Comment'] = dataframe_dataset['Comment']
# Converting all uppercase letters to lower case
    dataframe_dataset['Comment'] = dataframe_dataset['Comment'].str.lower()
# Removing \n
    dataframe_dataset['Comment'] = dataframe_dataset['Comment'].str.replace('\n', '')

# Removing all HTML Tags
    dataframe_dataset['Comment'] = dataframe_dataset['Comment'].str.replace('[^\w\s]','')
    dataframe_dataset['Comment'] = dataframe_dataset['Comment'].str.replace('http\S+','')
    dataframe_dataset['Comment'] = dataframe_dataset['Comment'].str.replace('www\S+','')

# Removing all the non ASCII characters
    dataframe_dataset['Comment'] = dataframe_dataset["Comment"].map(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    dataframe_dataset['Comment'] = dataframe_dataset['Comment'].str.replace('xa0',' ')
    dataframe_dataset['Comment'] = dataframe_dataset['Comment'].str.replace('xc2',' ')
# Removing the _ with ' '
    dataframe_dataset['Comment'] = dataframe_dataset['Comment'].str.replace('_',' ')
    
# Trimming all the white spaces
    #dataframe_dataset["Comment"] = dataframe_dataset["Comment"].map(str.strip)
    dataframe_dataset['Comment'] = pd.core.strings.str_strip(dataframe_dataset['Comment'])
    print(dataframe_dataset['Comment'])
#Tokenizing all the words in a given sentence
    dataframe_dataset['tokenized_sents'] = dataframe_dataset['Comment'].map(lambda row: nltk.word_tokenize(row))

# A custom stop words list which does not exclude words which have a referral meaning like you, we etc
    cust_list = open("Stop_words_custom_list.txt", 'r')
    cust_list = cust_list.read()
    cust_stop_list = re.split('\n|,', cust_list)
    #print(cust_stop_list)
    dataframe_dataset['tokenized_sents'] = dataframe_dataset['tokenized_sents'].apply(lambda x: [item for item in x if item not in cust_stop_list])

# Stemming all the words
    stemmer = SnowballStemmer('english')
    dataframe_dataset['stemmed'] = dataframe_dataset["tokenized_sents"].apply(lambda x: [stemmer.stem(y) for y in x])

# Separating the list of both insulting and non insulting comments
    '''for i in dataframe_dataset['Insult']:
        if dataframe_dataset.iloc[i]['Insult'] == 1:

            insulting_list.append(dataframe_dataset['stemmed'])

    print(insulting_list)'''
    '''dataframe_dataset_filter = dataframe_dataset.loc[dataframe_dataset['Insult'] == 1]
    insulting_list = dataframe_dataset_filter['stemmed'].tolist()
    print(insulting_list)'''

# Replacing shorthand and other internet slangs with correct phrases
    map_words = {"u": "you", "em":"them", "da":"the", "yo":"you",
            "ur":"you", "won't": "will not", "won't": "will not",
            "can't": "can not", "i'm": "i am", "i'm": "i am", "ain't": "is not",
            "'ll": "will", "'t": "not", "'ve": "have", "'s": "is", "'re": "are",
            "'d": "would"}

    dataframe_dataset['replaced stemmed'] = dataframe_dataset["stemmed"].map(lambda x: [map_words[x[i]] if x[i] in map_words else x[i] for i in range(len(x)) ])

    dataframe_dataset['joined replaced stemmed'] = \
            dataframe_dataset["replaced stemmed"]. \
            map(lambda x: (' '.join(word for word in x)))

    return


# Taking google list of bad words
bw = open('full-list-of-bad-words.txt', 'r')

#inp_text = bw.read()  # raeding file generated from puzzleenerator.py
#inp_text = re.split('\n|,', inp_text)
#print(inp_text)
#print dataframe_dataset.applymap(lambda x: isinstance(x, (int, float))).all(0)

#Reading Train Dataset
path = r'./train.csv'
print("Path to dataset is \n" + path)
dataframe_dataset = pd.read_csv(path, na_values='unknown', encoding="utf-8")
clean_dataset(dataframe_dataset)
dataframe_dataset.to_csv('Train_clean.csv')

#Reading Test Dataset
path = r'./Test/test_with_solutions.csv'
print("Path to test dataset is \n" + path)
dataframe_dataset_test = pd.read_csv(path, na_values='unknown', encoding="utf-8")
clean_dataset(dataframe_dataset_test)
dataframe_dataset_test.to_csv('Test_clean.csv')


