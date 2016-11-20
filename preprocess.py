import sys, math, re
from operator import itemgetter
import math
import os
from stemming.porter2 import stem
import nltk

ext = '.dtst_train'

#set to your working directory
dir_path = ''
#set to the name of your output file
results = 'final.out'

files = os.listdir(dir_path)
dicti = {}
class Word:
    def __init__(self,w):
        self.text = w
        self.alist = []
        self.final = []
    def create(self,feature_number,vector):
        self.alist.append(vector)
    def display_vector(self):
        print self.text,
        print self.final
        
    def make_input_autoencoder(self):
        for i in range(0,len(self.alist)):
            self.final.extend(self.alist[i])
        dicti[self.text] = self.final
            
    

#reading the files -->

for f in files:
  if f.endswith(ext):
    data = open(f,'r')
    open(results, 'a').write('\n' + data.read())
    data.close()

filename = 'final.out'
open('final.out','r')
word_list= word_list2 = []
word_list = re.split('\s+|[,":;\']+', file(filename).read())
print 'Words in text:', len(word_list)
os.remove('final.out')
keywordList1 = []
keywordList = ["UNK"]
keywordList1 = list(word_list)
kWF = open('first.dict','w')

for item in keywordList1:
    if len(item) >2:
        if item[-1] == 's' and item[-2].isalpha() == 0:
            item = item[:-2]
    keywordList.append(item)
    if len(item) >1:
        if item[-1] =='.':
            keywordList.append("UNK")
    
        


i=0
for item in keywordList:
    kWF.write("%s \n" % item)
    
#stemming-->
newkey = []
for item in keywordList:
    item = re.sub("[^'A-Za-z0-9\-\&]+", '', item)
#   w = stem(item)
    newkey.append(item)
i=0
newkey = [i for i in newkey if i != '' and i != '-']



print newkey
w=word=[]
i=0
kWF = open('second.dict','w')
for item in newkey:
    kWF.write("%s \n" % item)
    word.append(Word(item))         #appending objects to the object array word[]
    i+=1

Word.total_words=i
print Word.total_words



i=0 
'''
#feature 1 (one hot of the word itself)--> 
for w in newkey:
    bits = ['0']*len(newkey)
    bits[i] = '1'
    word[i].create(1,bits)
    i+=1
i=0
'''

#feature 2 (only 1st letter capital or not) -->
for w in newkey:
    if w[0].isupper() and w[1].islower():
        bits = [1]
    else:
        bits = [0]
    word[i].create(2,bits)
    i+=1

i=0
#feature 3 (all letters capital or not) -->
for w in newkey:
    if w.isupper():
        bits = [1]
    else:
        bits = [0]
    word[i].create(3,bits)
    i+=1
i=0
#feature 4 (alphanumeric) -->
flag = 0

for w in newkey:
    bits = [1]
    for k in range(1,len(w)):
        if (w[k-1].isdigit()):
            flag = 1
    if (flag == 1): 
            if (w.isdigit()):
                flag = 0
    if flag == 0:
        bits = [0]
    word[i].create(4,bits)
    i+=1
    flag=0
    
    
i=0

#feature 5 (digits) -->
for w in newkey:
    bits = [0]*4
    if w.isdigit():
        bits[len(w)-1]=1
    word[i].create(5,bits)
    i+=1
i=0

#feature 6 (POS tag)

tagged=nltk.pos_tag(newkey)
f = open ("POS.dict","r")
word_list = re.split('\s+|[.?!,":;\']+', f.read())
del word_list[-1]
f.close()
counter=0
for w in newkey:
    bits = [0]*(len(word_list))
    check = tagged[i][1]
    for k in word_list:
        if check == k:
            bits[counter]=1
        counter+=1
    if counter == 0:
        bits[16]=1
    counter=0
    word[i].create(6,bits)
    i+=1
i=0
    
            
    
for w in newkey:
    word[i].make_input_autoencoder()
    i+=1

str2=str(dicti)
f = open('output_makdict.txt','w')
f.write(str2)

