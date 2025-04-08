# Task 1: Tweet Extraction and Preprocessing

import tweepy
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# 1. Twitter API Authentication
API_KEY = "PoR6y74KFwRlu5cv3m1KREjMs"
API_SECRET = "MZDvOIYFbGQe2EfSRXOBrQ2iTydx9VdODzMlMA65uVuEFCUV1V"
BEARER = "AAAAAAAAAAAAAAAAAAAAAKzb0QEAAAAAVz3DQCfTlFR4GboIZe1c6lVJFG8%3DtYc7e4hQE66ugDCPCEvotKOtxfF2rr5my2viEH8jrjkcJGSeBE"

client = tweepy.Client(consumer_key=API_KEY, consumer_secret=API_SECRET, bearer_token=BEARER)

# 2. Define mental health related keywords
keywords = [
    "depressed", "depression", "suicidal", "suicide", "overdose", "relapse", 
    "addiction", "self harm", "overwhelmed", "hopeless", "need help", "alone"
]
query = "(" + " OR ".join(keywords) + ") -is:retweet lang:en"

# 3. Search recent tweets matching the query
response = client.search_recent_tweets(query=query, max_results=100,
                                       tweet_fields=["id", "created_at", "text", "public_metrics"])
tweets_data = []
if response.data:
    for tweet in response.data:
        metrics = tweet.public_metrics
        tweets_data.append({
            "id": tweet.id,
            "created_at": tweet.created_at.isoformat(),
            "text": tweet.text,
            "likes": metrics.get("like_count", 0),
            "retweets": metrics.get("retweet_count", 0),
            "replies": metrics.get("reply_count", 0)
        })

# Convert to DataFrame
df_tweets = pd.DataFrame(tweets_data)

# Save raw data
df_tweets.to_csv("tweets_raw.csv", index=False)

# 4. Text preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[@#]\w+', '', text)
    text = re.sub(r'[^0-9A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    cleaned = " ".join(tokens)
    return cleaned

df_tweets["clean_text"] = df_tweets["text"].apply(clean_text)

# Save cleaned data
df_tweets.to_csv("tweets_clean.csv", index=False)
print("Raw data saved to 'tweets_raw.csv'. Cleaned text saved to 'tweets_clean.csv'.")
