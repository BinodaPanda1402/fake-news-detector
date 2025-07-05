# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

# Step 1: Load dataset
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# Step 2: Add labels
fake['label'] = 0
real['label'] = 1

# Step 3: Combine and shuffle
data = pd.concat([fake, real])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# OPTIONAL SPEED BOOST (comment if full data needed)
# data = data[:10000]  # Fast train for dev testing

# Step 4: Use title + text together
data['content'] = data['title'] + " " + data['text']
X = data['content']
y = data['label']

# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Vectorize the text using fast TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1,1))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 7: Train fast Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 8: Evaluate
pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, pred)
print(f"✅ Model trained with accuracy: {acc*100:.2f}%")

# Step 9: Save model & vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("✅ Model and vectorizer saved to /model/")



