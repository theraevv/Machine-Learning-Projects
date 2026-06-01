import re
import pandas as pd
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
import yfinance as yf
import datetime

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# Initialize the lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

tickers = ["TSLA"]
rows = []
 
for symbol in tickers:
    for article in yf.Ticker(symbol).get_news(count=5):
        c = article.get("content", article)
        rows.append({
            "ticker":    symbol,
            "title":     c.get("title", ""),
            "date": datetime.datetime.strptime(c["pubDate"], "%Y-%m-%dT%H:%M:%SZ") if "pubDate" in c else "",
        })
 
df = pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)

def pre_process(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    cleaned_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 2
    ]

    cleaned_text = ' '.join(cleaned_tokens)

    X = tfidf.transform([cleaned_text])
    pred = model.predict(X)

    return pred[0]

sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

df["sentiment"] = df["title"].apply(pre_process)
df["sentiment"] = df["sentiment"].map(sentiment_labels).fillna("Unknown")

print(df)