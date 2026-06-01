import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
from nlp_preprocess import pre_process

# Initialize the lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

text = ["why does like every man in borderlands have slicked back hair havenâ€™t you heard of bangs you stupid assholes"]


cleaned_text = pre_process(text[0])

print(r"Text:", text)
print(r"Cleaned Text:", cleaned_text)

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

X = tfidf.transform([cleaned_text])
pred = model.predict(X)

# Output the predicted sentiment, mapping numeric labels to human-readable form if necessary
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
pred_sentiment = sentiment_labels.get(pred[0], "Unknown")

print("Predicted Sentiment:", pred_sentiment)