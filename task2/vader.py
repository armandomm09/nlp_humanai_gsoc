import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download VADER lexicon
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
df = pd.read_csv("tweets_clean.csv")

# Ensure all values are strings
df["clean_text"] = df["clean_text"].fillna("").astype(str)

# Compute sentiment score using VADER
df["sentiment_score"] = df["clean_text"].apply(lambda text: sia.polarity_scores(text)["compound"])

print(df[["clean_text", "sentiment_score"]])

def categorize_sentiment(score):
    if score >= 0.05:
        return "Positivo"
    elif score <= -0.05:
        return "Negativo"
    else:
        return "Neutral"

df["sentiment"] = df["sentiment_score"].apply(categorize_sentiment)

# TF-IDF to identify potentially significant terms
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(df["clean_text"])
feature_names = vectorizer.get_feature_names_out()
idf_scores = vectorizer.idf_

# Select top 15 rarest terms by IDF score
top_idx = idf_scores.argsort()[-15:]
high_idf_terms = [feature_names[i] for i in top_idx]

print("High TF-IDF terms (potentially high-risk):", high_idf_terms)

# Risk-related keywords (domain knowledge)
high_risk_terms = {"suicide", "suicidal", "kill myself", "end my life", "overdose", "selfharm"}
moderate_risk_terms = {"depressed", "depression", "relapse", "addiction", "hopeless", "overwhelmed"}

# Risk level classification
def classify_risk(text):
    text_lower = text.lower()
    for term in high_risk_terms:
        if term in text_lower:
            return "Alto"
    for term in moderate_risk_terms:
        if term in text_lower:
            return "Moderado"
    return "Bajo"

df["risk_level"] = df["text"].apply(classify_risk)

# Sentiment and risk distribution
sentiment_counts = df["sentiment"].value_counts()
risk_counts = df["risk_level"].value_counts()

print("\nSentiment Distribution:")
print(sentiment_counts.to_frame(name="Number of Tweets"))

print("\nRisk Level Distribution:")
print(risk_counts.to_frame(name="Number of Tweets"))

# Save results
df.to_csv("tweets_classified.csv", index=False)

# Bar chart for risk distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
risk_counts.plot(kind='bar', color=['red','orange','green'])
plt.title('Tweet Distribution by Risk Level')
plt.xlabel('Risk Level')
plt.ylabel('Number of Tweets')
plt.tight_layout()
plt.savefig("risk_distribution.png")
plt.show()
