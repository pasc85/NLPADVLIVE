# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:07:04 2020

@author: s-minhas
"""

import nltk
import matplotlib
import re
from nltk import ngrams
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.probability import FreqDist
import gensim
from gensim import models, corpora
from gensim.models import TfidfModel
from gensim.models import Word2Vec
import sklearn
from sklearn.decomposition import PCA
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import wordcloud
from wordcloud import WordCloud
import string
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
import operator 
import pandas as pd
import numpy as np
import os
import sys
import codecs
from nltk.corpus import stopwords
import csv
from collections import Counter


inverted_index_example = ["He likes to wink, He likes to drink!", "He likes to drink, and drink, and drink.", "The thing he likes to drink is ink","The ink he likes to drink is pink","He likes to wink, and drink pink ink" ] 


def set_tokens_to_lowercase(data):
    for index, entry in enumerate(data):
        data[index] = entry.lower()
    return data


def remove_punctuation(data):
    symbols = ",.!"
    for index, entry in enumerate(symbols):
        for index2, entry2 in enumerate (data):
            data[index2] = re.sub(r'[^\w]', ' ', entry2)
    return data

def remove_stopwords_from_tokens(data):
       stop_words = set(stopwords.words("english"))
       new_list = []
       for index, entry in enumerate(data):
           no_stopwords = ""
           entry = entry.split()
           for word in entry:
               if word not in stop_words:
                    no_stopwords = no_stopwords + " " + word 
           new_list.append(no_stopwords)
       return new_list


inverted_index_example = remove_stopwords_from_tokens(remove_punctuation(set_tokens_to_lowercase(inverted_index_example)))

vectorizer = CountVectorizer()
inverted_index_vectorised = vectorizer.fit_transform(inverted_index_example)

#if u want to look at it
tdm = pd.DataFrame(inverted_index_vectorised.toarray(), columns = vectorizer.get_feature_names())
print (tdm.transpose())


data = ["He likes to wink, He likes to drink!", "He likes to drink, and drink, and drink.", "The thing he likes to drink is ink","The ink he likes to drink is pink","He likes to wink, and drink pink ink" ] 

data = remove_stopwords_from_tokens(remove_punctuation(set_tokens_to_lowercase(data)))

binary_vectorizer = CountVectorizer(binary=True)
counts = binary_vectorizer.fit_transform(data)

#if u want to look at it
tdm = pd.DataFrame(counts.toarray(), columns = binary_vectorizer.get_feature_names())
tdm=tdm.transpose()
print (tdm)


def NOT(pterm): 
    for a in range(len(pterm)):
        if(pterm[a] == 0): 
            pterm[a] = 1
        elif(pterm[a] == 1): 
           pterm[a] = 0
    return pterm


term1 =  list(tdm.loc['drink'])
term2 = list(tdm.loc['ink'])
term3 =  NOT(list(tdm.loc['pink']))
terms = list(zip(term1, term2, term3))

vector= [terms[item][0] & terms[item][1] & terms[item][2]for item in range(len(terms))] 

for i in vector:
    if i == 1:
        print ("Document", vector.index(i), "meets search term criteria")




def set_tokens_to_lowercase(data):
    for index, entry in enumerate(data):
        data[index] = entry.lower()
    return data


def remove_punctuation(data):
    symbols = ",.!"
    for index, entry in enumerate(symbols):
        for index2, entry2 in enumerate (data):
            data[index2] = re.sub(r'[^\w]', ' ', entry2)
    return data

def remove_stopwords_from_tokens(data):
       stop_words = set(stopwords.words("english"))
       new_list = []
       for index, entry in enumerate(data):
           no_stopwords = ""
           entry = entry.split()
           for word in entry:
               if word not in stop_words:
                    no_stopwords = no_stopwords + " " + word 
           new_list.append(no_stopwords)
       return new_list
   
    
def stemming (data):
    st = PorterStemmer()
    for index, entry in enumerate(data):
        data[index] = st.stem(entry)
    return data
  
def read_data():
    raw_data_orig = pd.read_csv("C:/IR Course/Adv -IR/IATI.csv") 
    raw_data_orig = raw_data_orig.sample(500)
    #raw_data_orig  = open("C:/IR Course/Adv -IR/IATI3.pkl","rb")
    #raw_data_orig = pickle.load(raw_data_orig, encoding='iso-8859-1')
    raw_data_orig = raw_data_orig[raw_data_orig['description'].notnull()]
    return raw_data_orig


query ="climate change and environmental degradation"

def preprocess(pdf):
    for index, row in pdf.iterrows():
            row['description'] = " ".join(stemming(remove_stopwords_from_tokens(remove_punctuation(set_tokens_to_lowercase(row['description'].split(" "))))))
    return pdf

#preprocess documents
raw_data= preprocess(read_data())


#now preprocess query
query = " ".join(stemming(remove_stopwords_from_tokens(remove_punctuation(set_tokens_to_lowercase(query.split(" "))))))
rownames = raw_data["iati-identifier"]


#vectorise and get tfidf values
vectorizer = TfidfVectorizer()
vectorized_iati = vectorizer.fit_transform(raw_data["description"])
tdm = pd.DataFrame(vectorized_iati.toarray(), columns = vectorizer.get_feature_names())
tdm=tdm.set_index(rownames)

#now vectorise query
vectorized_query=vectorizer.transform(pd.Series(query))
query = pd.DataFrame(vectorized_query.toarray(), columns = vectorizer.get_feature_names())

# get cosine similarity

def cos_sim (pdf, qdf):
    f_similarity={}   
    for index, row in qdf.iterrows():
        for index2, row2 in pdf.iterrows():
             cos_result = cosine_similarity(np.array(row).reshape(1, row.shape[0]), np.array(row2).reshape(1, row2.shape[0]))
             f_similarity[index2] = round(float(cos_result),5)
    return f_similarity

cosine_scores=cos_sim (tdm, query)
#now rank
final_rank= sorted(cosine_scores.items(), key=operator.itemgetter(1), reverse=True)
final_rank = final_rank[0:5]
rownames = rownames.tolist()
unprocessed  = read_data()

for item in final_rank:
    if item[0] in rownames:
         
        print('IATI-IDENTIFIER {0} DESCRIPTION {1}'.format(item[0],unprocessed.iloc[rownames.index(item[0]),2])) 


   
df = pandas.read_csv('C:/IR Course/Adv -IR/IATI10k.csv', header = 0, encoding="iso-8859-1")
df = df[df.description.notnull()]

def set_tokens_to_lowercase(data):
    for index, entry in enumerate(data):
        data[index] = entry.lower()
    
    return data

def remove_punctuation(data):
    symbols = ",.!"
    for index, entry in enumerate(symbols):
        for index2, entry2 in enumerate (data):
            data[index2] = re.sub(r'[^\w]', ' ', entry2)
            data[index2] = entry2.strip()
            
    return data

def remove_stopwords_from_tokens(data):
       stop_words = set(stopwords.words("english"))
       stop_words.add(" ")
       new_list = []
       for index, entry in enumerate(data):
               if entry not in stop_words:
                    new_list.append(entry)
       return new_list

def clean_df(pdf):
  
    for index, row in pdf.iterrows():
         row['description'] =  remove_stopwords_from_tokens(remove_punctuation(set_tokens_to_lowercase(row['description'].split())))  
         row['description'] = " ".join(x for x in row['description'])
    return pdf


def calc_docscore(pdf, pqry):
    col_names =  ['Description', 'score']
    f_df2  = pandas.DataFrame(columns = col_names)
    for index, row in pdf.iterrows(): 
        rank = []
        docscore = 0
        scored = score(row['description'])
        for word in pqry.split(" "):
            if word in scored.keys():
                rank.append(float(scored[word] )+ float(allcounts[word]/total)/2)
        
        if rank != []:
            docscore = np.prod(np.array(rank)) 
           
        f_df2.loc[index] = pandas.Series({'Description':row['description'], 'score':docscore})
    return f_df2
               
def score (pstr):
    fdict = {}
    flist = pstr.split()
    fdict = dict(nltk.FreqDist(flist))
    for  key, value in fdict.items():
        fdict[key] = round(fdict[key]/len(flist),2)
    return fdict

df = clean_df(df)
qry = "reduce transmission of HIV"
qry=  remove_stopwords_from_tokens(remove_punctuation(set_tokens_to_lowercase(qry.split())))
qry = " ".join(x for x in qry)      

allcounts = {} 
for descript in df['description']:
      tmp = dict(nltk.FreqDist(descript.split()))
      for key, value in tmp.items():
        if key not in allcounts:
            allcounts[key] = value
        else: 
            allcounts[key] = allcounts[key] + value
total = sum(allcounts.values())
df2=calc_docscore(df, qry) 
   
df2sort_by_score = df2.sort_values('score', ascending=False)
print (df2sort_by_score[1:20])
    
  




































