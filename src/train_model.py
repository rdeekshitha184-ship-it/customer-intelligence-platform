import pandas as pd
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Stopwords
stop_words = set(stopwords.words("english"))

reviews = []
labels = []

# Load first 50,000 reviews
with open("data/train.ft.txt", "r", encoding="utf-8") as file:

    for i in range(50000):

        line = file.readline()

        if not line:
            break

        parts = line.split(" ", 1)

        labels.append(parts[0])
        reviews.append(parts[1])

# DataFrame
df = pd.DataFrame({
    "label": labels,
    "review": reviews
})

# Convert labels
df["label"] = df["label"].map({
    "__label__1": 0,
    "__label__2": 1
})

# Clean text
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

df["cleaned_review"] = df["review"].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["cleaned_review"])

y = df["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib

joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel Saved Successfully!")