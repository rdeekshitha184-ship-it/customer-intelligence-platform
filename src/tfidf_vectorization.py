import pandas as pd
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Label Encoding
df["label"] = df["label"].map({
    "__label__1": 0,
    "__label__2": 1
})

# Cleaning Function
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Clean Reviews
df["cleaned_review"] = df["review"].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["cleaned_review"])

y = df["label"]

print("TF-IDF Shape:")
print(X.shape)

print("\nLabels Shape:")
print(y.shape)