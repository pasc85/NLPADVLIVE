# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:57:05 2020

@author: s-minhas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from glob import glob
import os,re,string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import PCA
plt.rcParams.update({'font.size': 7})
import operator
from sklearn.feature_extraction.text import TfidfVectorizer

import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score






movie_reviews = pd.read_csv("C:/data/imdb/dataset.csv",  encoding='iso-8859-1') 
descriptions = movie_reviews.iloc[:, 0].values

movie_reviews['Sentiment'].value_counts()


def clean (ptext):

    notwanted = ["<br />", "?", "\'" , ",","@", "!", "...", "\"", "\n"]
    for line in range(len(ptext)):
        ptext[line]= ptext[line].lower()
        for mark in notwanted:
            if mark in ptext[line]:
                ptext[line] = ptext[line].replace(mark, "")
    return ptext

  # Optionally remove stop words (false by default)
    
  
def remove_stopwords(pstop, ptext):
    
    stops = set(pstop.words("english"))
    for line in range(len(ptext)):
         current = ptext[line].lower().split(" ")
         for index, value  in enumerate(current):
             if value in stops:
                current[index] = current[index].replace(value, "")
         ptext[line] = " ".join(current)     
    return ptext
    

#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
      
movie_reviews['SentimentText'] = clean(descriptions)
movie_reviews['SentimentText'] = remove_stopwords(stopwords, movie_reviews.iloc[:, 0].values )
#Apply function on review column
movie_reviews['SentimentText']=movie_reviews['SentimentText'].apply(simple_stemmer)

movie_reviews['TextLen']=movie_reviews['SentimentText'].str.len()


sentpos=movie_reviews[movie_reviews['Sentiment'] == 1]
sentneg=movie_reviews[movie_reviews['Sentiment'] == 0]


box_plot_data=[sentpos['TextLen'],sentneg['TextLen']]
plt.boxplot(box_plot_data,patch_artist=True,labels=['positive', 'negative'])
plt.show()


m = movie_reviews.groupby("Sentiment")
# Summary statistic for both sentiments
m.describe().head()


#wordclouds

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(str(sentpos['SentimentText']))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")



wc = WordCloud(background_color="white", max_words=200, width=400, height=400, random_state=1).generate(str(sentneg['SentimentText']))
# to recolour the image
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")

#split the dataset  
#train dataset
train_reviews=movie_reviews.Sentiment[:20000]
train_sentiments=movie_reviews.SentimentText[:20000]
#test dataset
test_reviews=movie_reviews.Sentiment[20000:]
test_sentiments=movie_reviews.SentimentText[20000:]

print(train_reviews.shape,train_sentiments.shape)
print(test_reviews.shape,test_sentiments.shape)



#Tfidf vectorizer
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
#transformed train reviews
tv_train_reviews=tv.fit_transform(train_sentiments)
#transformed test reviews
tv_test_reviews=tv.transform(test_sentiments)
print('Tfidf_train:',tv_train_reviews.shape)
print('Tfidf_test:',tv_test_reviews.shape)


from sklearn import preprocessing
#labeling the sentient data
lb=preprocessing.LabelBinarizer()

sentiment_data=lb.fit_transform(movie_reviews['Sentiment'])

#transformed sentiment data
sentiment_data=lb.fit_transform(movie_reviews['Sentiment'])
print(sentiment_data.shape)

#Spliting the sentiment data
train_sentiments=sentiment_data[:20000]
test_sentiments=sentiment_data[20000:]
print(train_sentiments)
print(test_sentiments)


#training the model
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
#Fitting the model for tfidf features
lr_tfidf=lr.fit(tv_train_reviews,train_sentiments)
print(lr_tfidf)


##Predicting the model for tfidf features
lr_tfidf_predict=lr.predict(tv_test_reviews)
print(lr_tfidf_predict)

#Accuracy score for tfidf features
lr_tfidf_score=accuracy_score(test_sentiments,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)

#Classification report for tfidf features
lr_tfidf_report=classification_report(test_sentiments,lr_tfidf_predict,target_names=['Positive','Negative'])
print(lr_tfidf_report)






