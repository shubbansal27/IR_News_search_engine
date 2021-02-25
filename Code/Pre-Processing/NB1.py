from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np









twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_counts = count_vect.fit_transform(twenty_test.data)
print(X_test_counts.shape)
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
print(X_test_tfidf.shape)


predicted = clf.predict(X_test_tfidf)
print(np.mean(predicted == twenty_test.target))
