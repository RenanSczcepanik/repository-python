import tweepy

consumer_key = 'FsDmR5tUBJpIgCYCnH6DcilDq'
consumer_secret = 'OOzQXLUrQhCMnBW1LF2Zv0eBYkg98XtskSa8a40CzzTLlKnUQ4'

access_token = '426490226-GBJUyV3P8S31tp5JZfDiGSmuO9yCVWeuNGhEK0Yi'
access_token_secret = 'ZBhXh6mKGy8ZH0EnQIaKXiSZsa0VcKxoqJTGRmuJVPraA'

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)

#print(api.me())
tweets = api.search('kabum')

for tweet in tweets:
    frase = tweet.text
    print('Tweet: {0}'.format(tweet.text))