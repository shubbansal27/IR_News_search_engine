# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 00:00:53 2017

@author: SWD
"""
import sklearn.datasets
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report as clsr
from PreP import NLTKPreprocessor



#text_clf = Pipeline([('vect', CountVectorizer(max_features=40000)),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
#text_clf = Pipeline([('preprocessor', NLTKPreprocessor()),('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])
#text_clf = Pipeline([('preprocessor', NLTKPreprocessor()),('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])


train = sklearn.datasets.load_files("D:\\Mohit IR\\DataSets\\AG\\Train",encoding='utf-8', decode_error='ignore')
test = sklearn.datasets.load_files("D:\\Mohit IR\\DataSets\\AG\\Test",encoding='utf-8', decode_error='ignore')
#print(train.target)

#train = sklearn.datasets.load_files("D:\\Mohit IR\\DataSets\\bbc-fulltext\\bbc\\Train1",encoding='utf-8', decode_error='ignore')
#test = sklearn.datasets.load_files("D:\\Mohit IR\\DataSets\\bbc-fulltext\\bbc\\Test1",encoding='utf-8', decode_error='ignore')
#print(train.target)


#prePro = NLTKPreprocessor()
#train.data = prePro.transform(train.data)
#train.data = prePro.inverse_transform(train.data)


###train TF-IDF
text_clf = text_clf.fit(train.data,train.target)
predicted = text_clf.predict(test.data)
print(predicted)
print(np.mean(predicted == test.target)*100,'%')
print(clsr(test.target, predicted))
##
#

