# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:58:01 2020

@author: s-minhas
"""

import operator
import pandas as pd
import re
import sklearn
from sklearn.decomposition import PCA
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import ngrams
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.collocations import *
from nltk.corpus import webtext
import numpy as np
import random
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity  
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


airline_tweets = pd.read_csv("C:/data/tweets/Tweets.csv")
airline_tweets.head()


airline_tweets.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.show()


airline_tweets.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])


airl= list(set(list(airline_tweets['airline'])))
pos=[]
neg=[]
neut=[]

for airlines in airl:
    filter = (airline_tweets['airline'] == airlines) & (airline_tweets['airline_sentiment'] == 'negative')
    num = len(airline_tweets[filter])
    neg.append(num)
    filter = (airline_tweets['airline'] == airlines) & (airline_tweets['airline_sentiment'] == 'positive')
    num = len(airline_tweets[filter])
    pos.append(num)
    filter = (airline_tweets['airline'] == airlines) & (airline_tweets['airline_sentiment'] == 'neutral')
    neut = len(airline_tweets[filter])


data = [pos,neg,neut]
  

fig, ax = plt.subplots()
ax.set_xticks(range(len(airl)))
ax.set_xticklabels(airl, rotation='vertical')
X = np.arange(6)
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)

plt.show()


features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values

print(features[1:5])
print(labels[1:5])

processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)
print (processed_feature[12])


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

predictions = text_classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))


























