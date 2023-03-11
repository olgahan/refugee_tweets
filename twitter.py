import snscrape
import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np

# Creating list to append tweet data to
attributes_container = []

# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('syrisch since:2014-07-05 until:2015-07-06').get_items()):
    if i>1000:
        break
    attributes_container.append([tweet.user.username, tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])
    
# Creating a dataframe to load the list
tweets_df = pd.DataFrame(attributes_container, columns=["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"])

#tweets_df["Tweet"].to_csv('tweets.csv')

#preprocessing

import preprocessor as p

#drop NAs and duplicates

train_df = train_df.dropna()
train_df = train_df.drop_duplicates()

train_df.head()

#use preprocessor

def preprocess_tweet(row):
    text = row['text']
    text = p.clean(text)
    return text

train_df['text'] = train_df.apply(preprocess_tweet, axis=1)

train_df.head()

#stopword removal

from gensim.parsing.preprocessing import remove_stopwords

def stopword_removal(row):
    text = row['text']
    text = remove_stopwords(text)
    return text

train_df['text'] = train_df.apply(stopword_removal, axis=1)

train_df.head()

#Remove extra white spaces, punctuation and apply lower casing

train_df['text'] = train_df['text'].str.lower().str.replace('[^\w\s]',' ').str.replace('\s\s+', ' ')

train_df.head()

