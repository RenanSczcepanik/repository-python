import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

import functions as functions

# predictor
X_col = 'tweet_text'
# classifier
y_col = 'sentiment' 

regex_pattern_hash = '#[\w]*'
regex_pattern_markup = '@[\w]*'

emoticon_list = {':))': 'positive_emoticon', ':)': 'positive_emoticon',
    ':D': 'positive_emoticon', ':(': 'negative_emoticon', ':((': 'negative_emoticon'}
    
#emoticon_list = {':))': '', ':)': '',
#    ':D': '', ':(': '', ':((': '', '8)': ''}
std_list = {'eh': 'é', 'vc': 'você', 'vcs': 'vocês','tb': 'também', 'tbm': 'também', 'obg': 'obrigado', 'gnt': 'gente', 'q': 'que', 'n': 'não', 'cmg': 'comigo', 'p': 'para', 'ta': 'está', 'to': 'estou', 'vdd': 'verdade'}
stopword_list = functions.get_stopwords()
# Train3Classes.csv

train_ds = pd.read_csv('../portuguese-tweets-for-sentiment-analysis/trainingdatasets/Train500.csv', delimiter=';')

train_ds[y_col] = train_ds[y_col].map({0: 'Negative', 1: 'Positive'})

X_train = train_ds.loc[:, X_col].values
y_train = train_ds.loc[:, y_col].values

X_train = functions.remove_url(X_train)
X_train = functions.remove_regex(X_train, regex_pattern_hash)
X_train = functions.remove_regex(X_train, regex_pattern_markup)
X_train = functions.replace_emoticons(X_train, emoticon_list)
X_train_tokens = functions.tokenize_text(X_train)
X_train_tokens = functions.apply_standardization(X_train_tokens, std_list)
X_train_tokens = functions.remove_stopwords(X_train_tokens, stopword_list)
X_train_tokens = functions.apply_stemmer(X_train_tokens)




test_ds = pd.read_csv('../portuguese-tweets-for-sentiment-analysis/testdatasets/Test.csv', delimiter=';')
test_ds[y_col] = test_ds[y_col].map({0: 'Negative', 1: 'Positive'})

X_test = test_ds.loc[:, X_col].values
y_test = test_ds.loc[:, y_col].values

X_test = functions.remove_url(X_test)
X_test = functions.remove_regex(X_test, regex_pattern_hash)
X_test = functions.remove_regex(X_test, regex_pattern_markup)
X_test = functions.replace_emoticons(X_test, emoticon_list)
X_test_tokens = functions.tokenize_text(X_test)
X_test_tokens = functions.apply_standardization(X_test_tokens, std_list)
X_test_tokens = functions.remove_stopwords(X_test_tokens, stopword_list)
X_test_tokens = functions.apply_stemmer(X_test_tokens)






vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_vect)


model = MultinomialNB()
print("começou")
model.fit(X_train_tfidf, y_train)


new_corpus = [
        '@acme A alegria está na luta, na tentativa, no sofrimento envolvido e não na vitória propriamente dita!', 
        'A alegria evita mil males e prolonga a vida.',
        'não se deve maltratar os idosos, eles possuem muita sabedoria!',
        '#filmedevampiro tome muito cuidado com o dracula... :( www.filmedevampiro.com.br'
        ]

def _new_corpus(new_corpus):
    X_new = new_corpus
    # Remove urls from text (http(s), www)
    X_new = functions.remove_url(X_new)
    # Remove hashtags
    regex_pattern = '#[\w]*'
    X_new = functions.remove_regex(X_new, regex_pattern)
    
    # Remove notations
    regex_pattern = '@[\w]*'
    X_new = functions.remove_regex(X_new, regex_pattern)
    
    # Replace emoticons ":)) :) :D :(" to positive_emoticon or negative_emoticon or neutral_emoticon
    X_new = functions.replace_emoticons(X_new, emoticon_list)
    
    # Tokenize text
    X_new_tokens = functions.tokenize_text(X_new)
    
    # Object Standardization
    X_new_tokens = functions.apply_standardization(X_new_tokens, std_list)
    
    # remove stopwords
    X_new_tokens = functions.remove_stopwords(X_new_tokens, stopword_list)
    
    # Stemming (dimensionality reduction)
    X_new_tokens = functions.apply_stemmer(X_new_tokens)
    
    # Dataset preparation
    # Untokenize text (transform tokenized text into string list)
    X_new = functions.untokenize_text(X_new_tokens)
    
    X_new_vect = vectorizer.transform(X_new)
    
    X_new_tfidf = tfidf_transformer.transform(X_new_vect)
    
    standalone_predictions = model.predict(X_new_tfidf)
    
    for doc, prediction in zip(X_new, standalone_predictions):
        print('%r => %s' % (doc, prediction))




X_test_vect = vectorizer.transform(X_test)

X_test_tfidf = tfidf_transformer.transform(X_test_vect)

predictions = model.predict(X_test_tfidf)

matrix = metrics.confusion_matrix(y_test, predictions)
print(matrix)

print(model.classes_)

acc1 = np.mean(predictions == y_test)
acc2 = functions.get_accuracy(matrix)
print(acc1, acc2)

for doc, prediction, y in zip(X_test[0:10], predictions[0:10], y_test[0:10]):
    print('%r => %s [%s]' % (doc, prediction, y))

model_MNB = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
    
model_MNB.fit(X_train, y_train)
predictions_MNB = model_MNB.predict(X_test)

matrix = metrics.confusion_matrix(y_test, predictions_MNB)
acc = functions.get_accuracy(matrix)
print(acc)











