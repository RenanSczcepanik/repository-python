import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from wordcloud import WordCloud

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

import functions as functions

# predictor
X_col = 'tweet_text'
# classifier
y_col = 'sentiment' 

regex_pattern_hash = '#[\w]*'
regex_pattern_markup = '@[\w]*'



# Train3Classes.csv

train_ds = pd.read_csv('../portuguese-tweets-for-sentiment-analysis/trainingdatasets/Train500.csv', delimiter=';')

sentiment=['Negative','Positive']

train_ds["sentiment"] = train_ds["sentiment"].map({0: 'Negative', 1: 'Positive'})

X_train = train_ds.loc[:, "tweet_text"].values
y_train = train_ds.loc[:, "sentiment"].values

X_train = functions.replace_emoticons(X_train)
X_train = functions.remove_url(X_train)
X_train = functions.remove_regex(X_train, regex_pattern_hash)
X_train = functions.remove_regex(X_train, regex_pattern_markup)
X_train = functions.remove_char(X_train)
X_train_tokens = functions.tokenize_text(X_train)
X_train_tokens = functions.apply_standardization(X_train_tokens)
X_train_tokens = functions.remove_stopwords(X_train_tokens)
X_train_tokens = functions.apply_stemmer(X_train_tokens)

X_train = functions.untokenize_text(X_train_tokens)

vectorizer = CountVectorizer(ngram_range=(1,2))
X_train_vect = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vect, y_train)

result = cross_val_predict(model, X_train_vect, y_train, cv=20)

print (pd.crosstab(y_train, result, rownames=['Real'], colnames=['Predito'], margins=True), '')

print(metrics.accuracy_score(y_train,result))

print(metrics.classification_report(y_train,result,sentiment))








#X_train = functions.untokenize_text(X_train_tokens)


#for teste in X_train_tokens[0:10]:
#    print(teste)
#    


