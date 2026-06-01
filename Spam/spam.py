import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
import time


input = input("Enter Email Content: \n")


df = pd.DataFrame([[input]], columns=['Words'])

# pre-process input data

# Step 1: List of common spam words
spam_words = [
    'free', 'win', 'winner', 'prize', 'buy now', 'click', 'cash', 
    'offer', 'rich', 'money', 'urgent', 'claim', 'limited', 'act now',
    'guarantee', 'credit', 'cheap', 'discount', 'deal', '!'
]


# Step 2: Function to count spam words in each text
def count_spam_words(text):
    text = text.lower()
    return sum(1 for word in spam_words if word in text)


# Step 3: Apply both features to DataFrame
df['spam_word_count'] = df['Words'].apply(count_spam_words)
df['message_length'] = df['Words'].apply(len)  # Total character count

# Optional: word count too
df['word_count'] = df['Words'].apply(lambda x: len(x.split()))

# Show the updated DataFrame
df_FE = pd.DataFrame(df)

vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Step 1: Vectorize the 'Words' column
vectorize_words = vectorizer.transform(df_FE['Words'])

# Step 2: Convert to DataFrame with TF-IDF features
vectorize_words = pd.DataFrame(
    vectorize_words.toarray(),
    columns=vectorizer.get_feature_names_out()
)

# Step 3: Select other features from original DataFrame
new_df = df_FE[['spam_word_count', 'message_length', 'word_count']].reset_index(drop=True)

# Step 4: Concatenate TF-IDF features with the selected columns
new_df = pd.concat([vectorize_words, new_df], axis=1)

model = joblib.load('spam_classifier_model.pkl')

pred = model.predict(new_df)

print(f"The email you input is {pred[0]}")
time.sleep(5)