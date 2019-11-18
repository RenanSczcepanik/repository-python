import GetOldTweets3 as got
import pandas

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("kabumcombr")\
                                           .setSince("2019-05-01")\
                                           .setUntil("2019-11-10")\
                                           
tweets = got.manager.TweetManager.getTweets(tweetCriteria)

lista_id = []

lista_tweet = []

for tweet in tweets:
    lista_id.append(tweet.id)
    lista_tweet.append(tweet.text)



df = pandas.DataFrame(data={"id": lista_id,"tweet": lista_tweet})
df.to_csv("./tweets_kabum4.csv", sep=';',index=False)






sample_train = random.sample(X_train_tokens, 10000)
text_cloud = get_text_cloud(sample_train)

word_cloud = WordCloud(max_font_size = 100, width = 1520, height = 535)
word_cloud.generate(text_cloud)
plt.figure(figsize = (16, 9))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()


sample_train = random.sample(X_test_tokens, 1000)
text_cloud = get_text_cloud(sample_train)

word_cloud = WordCloud(max_font_size = 100, width = 1520, height = 535)
word_cloud.generate(text_cloud)
plt.figure(figsize = (16, 9))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()



series = train_ds['sentiment'].value_counts()
ax = series.plot(kind='bar', title='Number for each sentiment')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')
plt.show()

series = train_ds['query_used'].value_counts()
ax = series.plot(kind='bar', title='Number for each sentiment')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')
plt.show()



def _new_corpus(new_corpus):
    X_new = new_corpus.loc[:, "tweet"].values
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
    
    X_new = functions.remove_char(X_new)
    
    # Tokenize text
    #X_new_tokens = functions.tokenize_text(X_new)
    X_new_tokens = functions.tokenize_bigram(X_new)
    
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


new_corpus = pd.read_csv('tweets_kabum3.csv', delimiter=';')


#_new_corpus(new_corpus)

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








test_ds = pd.read_csv('../portuguese-tweets-for-sentiment-analysis/testdatasets/Test.csv', delimiter=';')
test_ds["sentiment"] = test_ds["sentiment"].map({0: 'Negative', 1: 'Positive'})

X_test = test_ds.loc[:, "tweet_text"].values
y_test = test_ds.loc[:, "sentiment"].values



X_test = functions.remove_url(X_test)
X_test = functions.remove_regex(X_test, regex_pattern_hash)
X_test = functions.remove_regex(X_test, regex_pattern_markup)
#X_test = functions.replace_emoticons(X_test)


X_test = functions.remove_char(X_test)


X_test_tokens = functions.tokenize_text(X_test)


X_test_tokens = functions.apply_standardization(X_test_tokens)
X_test_tokens = functions.remove_stopwords(X_test_tokens)
X_test_tokens = functions.apply_stemmer(X_test_tokens)

X_test = functions.untokenize_text(X_test_tokens)






X_test_vect = vectorizer.transform(X_test)
predictions = model.predict(X_test_vect)

matrix = metrics.confusion_matrix(y_test, predictions)
print(matrix)

print(model.classes_)

acc1 = np.mean(predictions == y_test)
acc2 = functions.get_accuracy(matrix)
print(acc1, acc2)

for doc, prediction, y in zip(X_test[0:10], predictions[0:10], y_test[0:10]):
    print('%r => %s [%s]' % (doc, prediction, y))
    

test_ds = pd.read_csv('../portuguese-tweets-for-sentiment-analysis/testdatasets/Test.csv', delimiter=';')
test_ds["sentiment"] = test_ds["sentiment"].map({0: 'Negative', 1: 'Positive'})

X_test = test_ds.loc[:, "tweet_text"].values
y_test = test_ds.loc[:, "sentiment"].values

freq_test = vectorizer.transform(X_test)
predictions = model.predict(freq_test)

np.mean(predictions == y_test)