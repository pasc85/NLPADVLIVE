# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:23:57 2020

@author: s-minhas
"""

import pandas as pd
import matplotlib.pyplot as plt


##read in data
df = pd.read_csv("C:/data/BBC/News_dataset.csv", sep=';')


df.shape

##look at top 6
df.head()


df['Content'].head()


df.loc[1, 'Content']




#get all data from category and ut in list
cat = list(df['Category'])

#get all  unique values
cat2 = set(cat)

#add counts to dictionary
dict1 = {}
for value in cat2:
    dict1[value] = cat.count(value)
    
print (dict1)



#plot contents of dictionary
plt.bar(range(len(dict1)), dict1.values(), align='center')
plt.xticks(range(len(dict1)), list(dict1.keys()))

plt.show()



#length of each news article
df['News_length'] = df['Content'].str.len()

# have a look at top 6
df['News_length'].head()


#get basic stats on column
df['News_length'].describe()

#how many articles with more than 10k words
df_more10k = df[df['News_length'] > 10000]
len(df_more10k)


#95 percent of values have a news length  of value given by quantile
quantile_95 = df['News_length'].quantile(0.95)
print (quantile_95)
df_95 = df[df['News_length'] < quantile_95]

#get only category and newslength
df_95_2 = df_95[['Category', 'News_length']]

#have a look at it
df_95_2.head()

# all categories
print (cat2)

#make box plot
dictc = {}

for value in cat2:
     
     news_len = df_95_2[df_95_2['Category'] == value]
     news_len = list (news_len['News_length'])
     dictc[value]= news_len

        
dictc['business'][0:6]


box_plot_data=[dictc['business'],dictc['entertainment'],dictc['politics'],dictc['sport'], dictc['tech']]
plt.boxplot(box_plot_data,patch_artist=True,labels=['business', 'entertainment','politics','sport','tech'])
plt.show()


import numpy as np
from collections import Counter
import random
import nltk
from nltk.stem import WordNetLemmatizer 
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import os
os.chdir("C:/IR Course/Adv -IR/")
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
plt.style.use('ggplot')


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

def lemmatiser (pdf, pcol):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_text_list = []
    
   
    
    for row in range(len(pdf)):
        
        
        # Create an empty list containing lemmatized words
        lemmatized_list = []
        
        # Save the text and its words into an object
        text = pdf.loc[row, pcol]
        #print(text)
       
        text_words = text.split(" ")
    
        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
            
        # Join the list
        lemmatized_text = " ".join(lemmatized_list)
        
        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)
    return lemmatized_text_list


df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ")
df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
# Lowercasing the text
df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()
# remove punctuation
df['Content_Parsed_3'] = pd.Series(remove_punctuation (list(df['Content_Parsed_2'])))
#remove possessive
df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")
df.head()


df['Content_Parsed_5'] = lemmatiser (df, 'Content_Parsed_4')

df['Content_Parsed_6'] = df['Content_Parsed_5']

#remove stopwords
df['Content_Parsed_6'] = pd.Series(remove_stopwords_from_tokens(list(df['Content_Parsed_6'])))

list_columns = ["File_Name", "Category", "Complete_Filename", "Content", "Content_Parsed_6"]
df = df[list_columns]

df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})

print(df.loc[3,'Content_Parsed'])

df.head()

category_codes = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sport': 3,
    'tech': 4
}

# Category mapping
df['Category_Code'] = df['Category']
df = df.replace({'Category_Code':category_codes})


## ensure no other category in dataframe

for index, row in df.iterrows():
    if row['Category_Code'] not in [0,1,2,3,4]:
         df = df.drop (index)

df.tail()


X_train, X_test, y_train, y_test = train_test_split(df['Content_Parsed'], 
                                                    df['Category_Code'], 
                                                    test_size=0.15, 
                                                    random_state=8)


ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300


tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
                        
                        
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
labels_train = np.array(labels_train, dtype=np.int)

#training data
print(features_train.shape)


features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
labels_test = np.array(labels_test, dtype=np.int)


#test data
print(features_test.shape)


for Product, category_id in sorted(category_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' category:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")


svc_0= svm.SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=True, random_state=8, shrinking=True, tol=0.001,
  verbose=False)


print('Parameters currently in use:\n')
print(svc_0.get_params())

svc_0.fit(features_train, labels_train)
svc_pred = svc_0.predict(features_test)


print("The training accuracy is: ")
print(accuracy_score(labels_train, svc_0.predict(features_train)))



print("Classification report")
print(classification_report(labels_test,svc_pred))



































